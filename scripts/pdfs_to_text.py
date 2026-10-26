"""
Convert the downloaded full-text PDFs to plain text and pack them into one zip
so the full texts can be read and coded in the review.

Usage (from the repo folder):
  python scripts\\pdfs_to_text.py
Creates fulltext_txt/ and fulltext_txt.zip
"""
import os
import zipfile

from pypdf import PdfReader

SRC, DST, ZIP = "fulltext_pdfs", "fulltext_txt", "fulltext_txt.zip"
os.makedirs(DST, exist_ok=True)
n = 0
for name in sorted(os.listdir(SRC)):
    if not name.lower().endswith(".pdf"):
        continue
    try:
        text = "\n".join((p.extract_text() or "") for p in PdfReader(os.path.join(SRC, name)).pages)
    except Exception as ex:
        text = f"__ERROR__ {ex}"
    with open(os.path.join(DST, name[:-4] + ".txt"), "w", encoding="utf-8") as f:
        f.write(text)
    n += 1
    print(n, name)
with zipfile.ZipFile(ZIP, "w", zipfile.ZIP_DEFLATED) as z:
    for name in sorted(os.listdir(DST)):
        z.write(os.path.join(DST, name), name)
print(f"\n{n} text files -> {ZIP} ({os.path.getsize(ZIP) // 1024} KB)")
