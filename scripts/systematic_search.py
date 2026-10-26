"""
Systematic literature search for the Fed-ISIC2019 leakage / conformal project.

Runs four search blocks (B1-B4) on three free sources:
  - OpenAlex (broad coverage, close to Scopus/Web of Science)
  - PubMed (E-utilities)
  - arXiv (API)
Then removes duplicates and writes one Excel workbook:
  search_log   : source, block, exact query, date, hit count
  all_records  : every retrieved record
  screening    : de-duplicated records with empty decision columns
  prisma       : counts for the PRISMA flow diagram

Usage (Windows PowerShell):
  pip install requests pandas openpyxl
  python systematic_search.py --email your_email@example.com

The e-mail is sent only to OpenAlex and PubMed, which ask for one so they can
contact heavy users; it is not stored anywhere else.
"""
import argparse
import datetime as dt
import re
import time
import xml.etree.ElementTree as ET

import pandas as pd
import requests

YEARS = (2017, 2026)

# One Boolean query per block, written once and adapted per source below.
BLOCKS = {
    "B1_FL_skin": {
        "openalex": '("federated learning" OR federated) AND ("skin lesion" OR "skin lesions" OR dermoscopy '
                    'OR dermoscopic OR dermatoscopy OR melanoma OR ISIC OR HAM10000)',
        "pubmed": '("federated learning"[tiab] OR federated[tiab]) AND ("skin lesion*"[tiab] OR dermoscop*[tiab] '
                  'OR dermatoscop*[tiab] OR melanoma[tiab] OR ISIC[tiab] OR HAM10000[tiab])',
        "arxiv": '(abs:"federated learning" OR abs:federated) AND (abs:"skin lesion" OR abs:dermoscopy OR '
                 'abs:dermoscopic OR abs:melanoma OR abs:ISIC OR abs:HAM10000)',
    },
    "B2_leakage_derm": {
        "openalex": '("data leakage" OR duplicate OR duplicates OR "near-duplicate" OR "lesion-level") AND '
                    '(dermoscopy OR dermoscopic OR ISIC OR HAM10000 OR "skin lesion")',
        "pubmed": '("data leakage"[tiab] OR duplicate*[tiab] OR "near-duplicate*"[tiab]) AND '
                  '(dermoscop*[tiab] OR ISIC[tiab] OR HAM10000[tiab] OR "skin lesion*"[tiab])',
        "arxiv": '(abs:leakage OR abs:duplicate OR abs:duplicates) AND (abs:dermoscopy OR abs:ISIC OR '
                 'abs:HAM10000 OR abs:"skin lesion")',
    },
    "B3_conformal": {
        "openalex": '"conformal prediction" AND (federated OR "medical imaging" OR "medical image" OR '
                    'dermatology OR "skin lesion")',
        "pubmed": '"conformal prediction"[tiab] AND (federated[tiab] OR imaging[tiab] OR image[tiab] OR '
                  'dermatolog*[tiab] OR "skin lesion*"[tiab])',
        "arxiv": 'abs:"conformal prediction" AND (abs:federated OR abs:"medical imaging" OR abs:"medical image" '
                 'OR abs:dermatology)',
    },
    "B4_calibration_FL": {
        "openalex": '(calibration OR "expected calibration error" OR "uncertainty quantification") AND '
                    '"federated learning"',
        "pubmed": '(calibration[tiab] OR "uncertainty quantification"[tiab]) AND "federated learning"[tiab]',
        "arxiv": '(abs:calibration OR abs:"uncertainty quantification") AND abs:"federated learning"',
    },
}

LOG, RECORDS = [], []


def log(source, block, query, hits, note=""):
    LOG.append({"source": source, "block": block, "query": query, "hits": hits,
                "search_date": dt.date.today().isoformat(), "note": note})
    print(f"  {source:8s} {block:18s} hits={hits} {note}")


# ---------------------------------------------------------------- OpenAlex
def openalex(block, query, email, cap=2000):
    url = "https://api.openalex.org/works"
    params = {"filter": f"title_and_abstract.search:{query},publication_year:{YEARS[0]}-{YEARS[1]}",
              "per-page": 200, "cursor": "*", "mailto": email,
              "select": "id,doi,title,publication_year,primary_location,type,language"}
    total, got = None, 0
    while True:
        r = requests.get(url, params=params, timeout=60)
        r.raise_for_status()
        js = r.json()
        total = js["meta"]["count"] if total is None else total
        for w in js["results"]:
            src = ((w.get("primary_location") or {}).get("source") or {}).get("display_name")
            RECORDS.append({"source": "OpenAlex", "block": block, "title": w.get("title"),
                            "year": w.get("publication_year"), "venue": src, "doi": w.get("doi"),
                            "type": w.get("type"), "language": w.get("language"), "id": w.get("id")})
            got += 1
        cur = js["meta"].get("next_cursor")
        if not cur or got >= min(total, cap):
            break
        params["cursor"] = cur
        time.sleep(0.2)
    log("OpenAlex", block, query, total, "" if got >= total else f"downloaded first {got}")


# ---------------------------------------------------------------- PubMed
def pubmed(block, query, email):
    base = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/"
    term = f"({query}) AND ({YEARS[0]}:{YEARS[1]}[dp])"
    r = requests.get(base + "esearch.fcgi", params={"db": "pubmed", "term": term, "retmax": 2000,
                                                     "retmode": "json", "email": email}, timeout=60)
    r.raise_for_status()
    js = r.json()["esearchresult"]
    ids, total = js.get("idlist", []), int(js.get("count", 0))
    for i in range(0, len(ids), 200):
        s = requests.get(base + "esummary.fcgi", params={"db": "pubmed", "id": ",".join(ids[i:i + 200]),
                                                          "retmode": "json", "email": email}, timeout=60)
        s.raise_for_status()
        res = s.json()["result"]
        for pid in res.get("uids", []):
            d = res[pid]
            doi = next((a["value"] for a in d.get("articleids", []) if a.get("idtype") == "doi"), None)
            RECORDS.append({"source": "PubMed", "block": block, "title": d.get("title"),
                            "year": (d.get("pubdate") or "")[:4], "venue": d.get("fulljournalname"),
                            "doi": f"https://doi.org/{doi}" if doi else None, "type": ",".join(d.get("pubtype", [])),
                            "language": ",".join(d.get("lang", [])), "id": f"PMID:{pid}"})
        time.sleep(0.4)
    log("PubMed", block, term, total)


# ---------------------------------------------------------------- arXiv
def arxiv(block, query, cap=1000):
    ns = {"a": "http://www.w3.org/2005/Atom", "o": "http://a9.com/-/spec/opensearch/1.1/"}
    start, total = 0, None
    while True:
        r = requests.get("http://export.arxiv.org/api/query",
                         params={"search_query": query, "start": start, "max_results": 200}, timeout=60)
        r.raise_for_status()
        root = ET.fromstring(r.content)
        total = int(root.find("o:totalResults", ns).text) if total is None else total
        entries = root.findall("a:entry", ns)
        for e in entries:
            year = int(e.find("a:published", ns).text[:4])
            if not (YEARS[0] <= year <= YEARS[1]):
                continue
            RECORDS.append({"source": "arXiv", "block": block,
                            "title": " ".join(e.find("a:title", ns).text.split()), "year": year,
                            "venue": "arXiv", "doi": None, "type": "preprint", "language": "en",
                            "id": e.find("a:id", ns).text})
        start += 200
        if not entries or start >= min(total, cap):
            break
        time.sleep(3)  # arXiv asks for 3 s between calls
    log("arXiv", block, query, total, "hits before year filter")


# ---------------------------------------------------------------- de-duplication
def norm_title(t):
    return re.sub(r"[^a-z0-9]", "", (t or "").lower())[:120]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--email", required=True)
    ap.add_argument("--out", default="systematic_search.xlsx")
    a = ap.parse_args()

    for block, q in BLOCKS.items():
        print(block)
        for name, fn in [("OpenAlex", lambda: openalex(block, q["openalex"], a.email)),
                         ("PubMed", lambda: pubmed(block, q["pubmed"], a.email)),
                         ("arXiv", lambda: arxiv(block, q["arxiv"]))]:
            try:
                fn()
            except Exception as ex:  # keep going if one source fails
                log(name, block, q[name.lower()], None, f"ERROR: {ex}")

    rec = pd.DataFrame(RECORDS)
    rec["key"] = rec["doi"].fillna("").str.lower().str.replace("https://doi.org/", "", regex=False)
    rec.loc[rec["key"] == "", "key"] = rec["title"].map(norm_title)
    dedup = (rec.sort_values(["source"])
                .groupby("key", as_index=False)
                .agg(title=("title", "first"), year=("year", "first"), venue=("venue", "first"),
                     doi=("doi", "first"), type=("type", "first"),
                     sources=("source", lambda s: ", ".join(sorted(set(s)))),
                     blocks=("block", lambda s: ", ".join(sorted(set(s))))))
    for col in ["stage1_title_abstract (include/exclude)", "stage1_reason",
                "stage2_full_text (include/exclude)", "stage2_reason",
                "uses_Fed-ISIC2019 (y/n)", "split_type", "notes"]:
        dedup[col] = ""

    prisma = pd.DataFrame({
        "stage": ["Records identified (all sources)", "Records after duplicates removed",
                  "Records screened (title/abstract)", "Excluded at title/abstract",
                  "Full texts assessed", "Excluded at full text", "Studies included",
                  "Added by snowballing"],
        "count": [len(rec), len(dedup), len(dedup), "", "", "", "", ""]})

    with pd.ExcelWriter(a.out) as xw:
        pd.DataFrame(LOG).to_excel(xw, sheet_name="search_log", index=False)
        rec.drop(columns="key").to_excel(xw, sheet_name="all_records", index=False)
        dedup.drop(columns="key").to_excel(xw, sheet_name="screening", index=False)
        prisma.to_excel(xw, sheet_name="prisma", index=False)
    print(f"\nRecords: {len(rec)} | after de-duplication: {len(dedup)} | saved {a.out}")


if __name__ == "__main__":
    main()
