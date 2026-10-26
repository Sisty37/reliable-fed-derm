"""
Fetch abstracts for the stage-1 includes so they can be screened (stage 2).

Reads the 'stage1_includes' sheet of the screened workbook, looks each record up
in OpenAlex (by DOI, otherwise by exact title) and, for arXiv-only records, in the
arXiv API. Writes a new workbook with an 'abstracts' sheet and empty decision columns.

Usage (Windows PowerShell, from the repo folder):
  python scripts\\fetch_abstracts.py systematic_search_2026-10-26_screened.xlsx --email you@example.com
"""
import argparse
import re
import time
import xml.etree.ElementTree as ET

import pandas as pd
import requests

OA = "https://api.openalex.org/works"


def inv_to_text(inv):
    if not inv:
        return None
    pos = [(p, w) for w, ps in inv.items() for p in ps]
    return " ".join(w for _, w in sorted(pos))


def norm(t):
    return re.sub(r"[^a-z0-9]", "", str(t).lower())


def from_openalex(doi, title, email):
    sel = "id,doi,title,abstract_inverted_index,primary_location"
    if isinstance(doi, str) and doi.strip():
        d = doi.replace("https://doi.org/", "")
        r = requests.get(f"{OA}/doi:{d}", params={"mailto": email, "select": sel}, timeout=60)
        if r.status_code == 200:
            return inv_to_text(r.json().get("abstract_inverted_index")), "OpenAlex (DOI)"
    q = re.sub(r"[,:()|]", " ", str(title))[:200]
    r = requests.get(OA, params={"filter": f"title.search:{q}", "per-page": 5, "mailto": email, "select": sel},
                     timeout=60)
    if r.status_code == 200:
        for w in r.json().get("results", []):
            if norm(w.get("title"))[:80] == norm(title)[:80]:
                txt = inv_to_text(w.get("abstract_inverted_index"))
                if txt:
                    return txt, "OpenAlex (title)"
    return None, None


def from_arxiv(title):
    q = re.sub(r"[^A-Za-z0-9 ]", " ", str(title))
    q = " AND ".join(f"ti:{w}" for w in q.split()[:12])
    r = requests.get("http://export.arxiv.org/api/query", params={"search_query": q, "max_results": 3}, timeout=60)
    if r.status_code != 200:
        return None, None
    ns = {"a": "http://www.w3.org/2005/Atom"}
    for e in ET.fromstring(r.content).findall("a:entry", ns):
        if norm(e.find("a:title", ns).text)[:80] == norm(title)[:80]:
            return " ".join(e.find("a:summary", ns).text.split()), "arXiv"
    return None, None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("workbook")
    ap.add_argument("--email", required=True)
    ap.add_argument("--out", default="stage2_abstracts.xlsx")
    a = ap.parse_args()

    inc = pd.read_excel(a.workbook, sheet_name="stage1_includes")
    abstracts, where = [], []
    for k, row in inc.iterrows():
        txt, src = None, None
        try:
            txt, src = from_openalex(row["doi"], row["title"], a.email)
            if not txt:
                time.sleep(3)
                txt, src = from_arxiv(row["title"])
        except Exception as ex:
            src = f"ERROR: {ex}"
        abstracts.append(txt)
        where.append(src or "not found")
        print(f"{k + 1}/{len(inc)}  {where[-1]:18s} {str(row['title'])[:70]}")
        time.sleep(0.3)
    inc["abstract"] = abstracts
    inc["abstract_source"] = where
    for c in ["stage2_abstract (include/exclude)", "stage2_reason", "uses_Fed-ISIC2019 (y/n)", "notes"]:
        inc[c] = ""
    inc.to_excel(a.out, sheet_name="abstracts", index=False)
    found = sum(x is not None for x in abstracts)
    print(f"\nAbstracts found: {found}/{len(inc)} | saved {a.out}")


if __name__ == "__main__":
    main()
