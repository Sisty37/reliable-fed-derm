"""
Full-text screening helper (data extraction, stage 3).

For every record in fulltext_queue.xlsx it
  1. looks for a legal open-access PDF (OpenAlex best OA location; arXiv by DOI or title),
  2. downloads it into fulltext_pdfs/ (skips files already downloaded),
  3. extracts the text and records whether, and where, the paper mentions
     Fed-ISIC2019 / FLamby, ISIC 2019, HAM10000, lesion- or patient-level splits,
     image-level / random splits, calibration (ECE, temperature scaling),
     conformal prediction / coverage, and real centers,
  4. writes fulltext_extraction.xlsx with one row per record and short text snippets.

The script only flags passages; the include/exclude decision is made by reading.

Usage (Windows PowerShell, from the repo folder):
  pip install requests pandas openpyxl pypdf
  python scripts\\fulltext_extract.py fulltext_queue.xlsx --email you@example.com
"""
import argparse
import os
import re
import time
import xml.etree.ElementTree as ET

import pandas as pd
import requests
from pypdf import PdfReader

HEAD = {"User-Agent": "reliable-fed-derm systematic review (mailto:{email})"}

PATTERNS = {
    "fedisic_flamby": r"Fed-?ISIC\s?-?2019|FLamby",
    "isic2019": r"ISIC[\s-]?2019",
    "ham10000": r"HAM-?10000|HAM10k",
    "lesion_or_patient_split": r"lesion[\s_-]?(id|level|wise|grouped|disjoint)|patient[\s-]?(level|wise|disjoint|grouped)|group\s?k-?fold|grouped (split|cross)",
    "image_or_random_split": r"random(ly)? (split|divid|partition)|image[\s-]level split|train_test_split|stratified (split|by center)|80\s?[:/]\s?20|8\s?:\s?2|70\s?[:/]\s?30",
    "leakage_duplicates": r"leak(age)?|duplicate|same lesion|near-duplicate",
    "calibration": r"calibrat|\bECE\b|expected calibration error|temperature scaling|reliability diagram|Brier",
    "conformal_coverage": r"conformal|prediction set|marginal coverage|coverage guarantee",
    "real_centers": r"six (centers|centres|clients|hospitals|institutions)|6 (centers|centres|clients)|natural(ly)? (split|partition)|real[- ]world (split|partition|clients|centers)",
}


def oa_pdf_url(doi, title, email):
    """Return (url, source) of an open-access PDF, or (None, reason)."""
    if doi:
        r = requests.get(f"https://api.openalex.org/works/doi:{doi}",
                         params={"mailto": email, "select": "best_oa_location,locations,ids"}, timeout=60)
        if r.status_code == 200:
            js = r.json()
            locs = [js.get("best_oa_location")] + (js.get("locations") or [])
            for loc in locs:
                if loc and loc.get("pdf_url"):
                    return loc["pdf_url"], "OpenAlex OA"
        m = re.search(r"10\.48550/arxiv\.(\d{4}\.\d{4,5})", doi, re.I)
        if m:
            return f"https://arxiv.org/pdf/{m.group(1)}", "arXiv (DOI)"
    # arXiv by title
    words = re.sub(r"[^A-Za-z0-9 ]", " ", str(title)).split()[:10]
    if words:
        q = " AND ".join(f"ti:{w}" for w in words)
        r = requests.get("http://export.arxiv.org/api/query",
                         params={"search_query": q, "max_results": 3}, timeout=60)
        time.sleep(3)
        if r.status_code == 200:
            ns = {"a": "http://www.w3.org/2005/Atom"}
            norm = lambda t: re.sub(r"[^a-z0-9]", "", str(t).lower())[:70]
            for e in ET.fromstring(r.content).findall("a:entry", ns):
                if norm(e.find("a:title", ns).text) == norm(title):
                    aid = e.find("a:id", ns).text.rsplit("/", 1)[-1]
                    return f"https://arxiv.org/pdf/{aid}", "arXiv (title)"
    return None, "no open-access PDF found"


def download(url, path, email):
    r = requests.get(url, headers={k: v.format(email=email) for k, v in HEAD.items()},
                     timeout=90, allow_redirects=True)
    if r.status_code != 200 or not r.content[:5] == b"%PDF-":
        return f"download failed (HTTP {r.status_code}, not a PDF)" if r.status_code == 200 else f"HTTP {r.status_code}"
    with open(path, "wb") as f:
        f.write(r.content)
    return "ok"


def pdf_text(path):
    try:
        return " ".join((p.extract_text() or "") for p in PdfReader(path).pages)
    except Exception as ex:
        return f"__ERROR__ {ex}"


def snippets(text, pat, n=2, w=110):
    out = []
    for m in re.finditer(pat, text, re.I):
        s = text[max(0, m.start() - w): m.end() + w].replace("\n", " ")
        out.append("..." + re.sub(r"\s+", " ", s) + "...")
        if len(out) >= n:
            break
    return " | ".join(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("queue")
    ap.add_argument("--email", required=True)
    ap.add_argument("--out", default="fulltext_extraction.xlsx")
    ap.add_argument("--pdf-dir", default="fulltext_pdfs")
    a = ap.parse_args()
    os.makedirs(a.pdf_dir, exist_ok=True)

    q = pd.read_excel(a.queue).fillna("")
    rows = []
    for k, rec in q.iterrows():
        rid = str(rec["row_id"])
        path = os.path.join(a.pdf_dir, re.sub(r"[^A-Za-z0-9_-]", "_", rid) + ".pdf")
        status, src, url = "", "", ""
        try:
            if os.path.exists(path):
                status, src = "ok", "already downloaded"
            else:
                url, src = oa_pdf_url(str(rec["doi"]).strip(), rec["title"], a.email)
                status = download(url, path, a.email) if url else "no PDF"
        except Exception as ex:
            status = f"ERROR {ex}"
        row = {"row_id": rid, "category": rec["category"], "title": rec["title"], "year": rec["year"],
               "doi": rec["doi"], "pdf_status": status, "pdf_source": src, "pdf_url": url}
        if status == "ok":
            text = pdf_text(path)
            row["pages_chars"] = len(text)
            for name, pat in PATTERNS.items():
                hits = len(re.findall(pat, text, re.I))
                row[name] = hits
                row[name + "_snippet"] = snippets(text, pat) if hits else ""
        rows.append(row)
        print(f"{k + 1}/{len(q)}  {status[:28]:28s} {str(rec['title'])[:60]}")
        time.sleep(0.5)

    df = pd.DataFrame(rows)
    for c in ["fulltext_decision (include/exclude)", "fulltext_reason", "uses_Fed-ISIC2019 (y/n)",
              "split_unit", "reports_calibration (y/n)", "reports_coverage (y/n)", "notes"]:
        df[c] = ""
    df.to_excel(a.out, index=False)
    ok = (df.pdf_status == "ok").sum()
    print(f"\nPDFs obtained: {ok}/{len(df)} | saved {a.out} (PDFs in {a.pdf_dir}/)")


if __name__ == "__main__":
    main()
