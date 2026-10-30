"""
Download and preprocess the Fed-ISIC2019 images on Windows (or any OS) without wget/unzip.

Steps
  1. download ISIC_2019_Training_Input.zip (about 9 GB) from the ISIC S3 bucket used by FLamby,
     with resume support (re-run the script if the download breaks);
  2. read only the 23,247 images listed in splits/split_official.csv straight from the zip;
  3. preprocess exactly as FLamby's resize_images.py (commit edacf54): shorter edge resized to
     224 px with PIL bilinear, aspect ratio kept, colour constancy (power 6), saved as JPEG
     with PIL defaults;
  4. write a manifest (file name, size, SHA-256) and check that all 23,247 images exist.

Usage (from the repo folder):
  python scripts\\prepare_fedisic.py --data-dir data\\fedisic
  python scripts\\prepare_fedisic.py --data-dir data\\fedisic --delete-zip    # free 9 GB after success

Licence: ISIC 2019 is CC-BY-NC 4.0; HAM10000 has its own terms. Accept both before use:
  https://challenge.isic-archive.com/data/#2019
  https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/DBW86T
"""
import argparse
import hashlib
import io
import os
import re
import sys
import time
import zipfile
from multiprocessing import Pool

import numpy as np
import pandas as pd
import requests
from PIL import Image

URL = "https://isic-archive.s3.amazonaws.com/challenges/2019/ISIC_2019_Training_Input.zip"
# FLamby (commit edacf54) used https://isic-challenge-data.s3.amazonaws.com/2019/ISIC_2019_Training_Input.zip,
# which now returns a "moved" page pointing to the URL above.
SIZE = 224


# ---------------------------------------------------------------- FLamby preprocessing (verbatim logic)
def color_constancy(img, power=6):
    img_dtype = img.dtype
    img = img.astype("float32")
    img_power = np.power(img, power)
    rgb_vec = np.power(np.mean(img_power, (0, 1)), 1 / power)
    rgb_norm = np.sqrt(np.sum(np.power(rgb_vec, 2.0)))
    rgb_vec = rgb_vec / rgb_norm
    rgb_vec = 1 / (rgb_vec * np.sqrt(3))
    img = np.multiply(img, rgb_vec)
    return img.astype(img_dtype)


def resize_and_cc(img):
    ratio = float(SIZE) / min(img.size)
    new_size = tuple(int(x * ratio) for x in img.size)
    img = img.resize(new_size, resample=Image.BILINEAR)
    return Image.fromarray(color_constancy(np.array(img)))


# ---------------------------------------------------------------- download with resume
def resolve(url):
    """Follow HTML 'moved' pages (meta refresh) until a real file URL is reached."""
    for _ in range(5):
        h = requests.head(url, timeout=60, allow_redirects=True)
        ctype = h.headers.get("Content-Type", "")
        if "html" not in ctype and int(h.headers.get("Content-Length", 0) or 0) > 1000:
            return url
        page = requests.get(url, timeout=60).text
        m = re.search(r'url=([^"\'>\s]+)', page)
        if not m:
            return url
        print(f"file moved -> {m.group(1)}")
        url = m.group(1)
    return url


def download(url, dest):
    """Download with resume; prints the server's answer if something goes wrong."""
    url = resolve(url)
    head = requests.head(url, timeout=60, allow_redirects=True)
    total = int(head.headers.get("Content-Length", 0) or 0)
    print(f"server answer to HEAD: HTTP {head.status_code}, size {total / 1e9:.2f} GB")
    have = os.path.getsize(dest) if os.path.exists(dest) else 0
    if have and not zipfile.is_zipfile(dest) and (not total or have > total):
        os.remove(dest)
        have = 0
    if total and have == total and zipfile.is_zipfile(dest):
        print(f"zip already complete ({total / 1e9:.2f} GB)")
        return
    print(f"downloading, resuming at {have / 1e9:.2f} GB")
    headers = {"Range": f"bytes={have}-"} if have else {}
    with requests.get(url, headers=headers, stream=True, timeout=120, allow_redirects=True) as r:
        if r.status_code not in (200, 206):
            sys.exit(f"download refused: HTTP {r.status_code}\n{r.text[:500]}")
        total = total or int(r.headers.get("Content-Length", 0) or 0) + (have if r.status_code == 206 else 0)
        mode = "ab" if have and r.status_code == 206 else "wb"
        done = have if mode == "ab" else 0
        t0 = time.time()
        with open(dest, mode) as f:
            for chunk in r.iter_content(chunk_size=8 * 1024 * 1024):
                f.write(chunk)
                done += len(chunk)
                rate = (done - have) / max(time.time() - t0, 1) / 1e6
                sys.stdout.write(f"\r  {done / 1e9:.2f} / {total / 1e9:.2f} GB  ({rate:.1f} MB/s)   ")
                sys.stdout.flush()
    print()
    size = os.path.getsize(dest)
    if size < 1_000_000_000 or not zipfile.is_zipfile(dest):
        with open(dest, "rb") as f:
            head_bytes = f.read(300)
        sys.exit(f"downloaded file is not the ISIC zip ({size} bytes). First bytes:\n{head_bytes!r}\n"
                 "If the download stopped early, run the same command again to resume.")


# ---------------------------------------------------------------- parallel preprocessing
_ZIP = None


def _init(zip_path):
    global _ZIP
    _ZIP = zipfile.ZipFile(zip_path)


def _work(job):
    member, out_path = job
    if os.path.exists(out_path):
        return "skip"
    with _ZIP.open(member) as f:
        img = Image.open(io.BytesIO(f.read())).convert("RGB")
    resize_and_cc(img).save(out_path)
    return "ok"


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        h.update(f.read())
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default=os.path.join("data", "fedisic"))
    ap.add_argument("--split", default=os.path.join("splits", "split_official.csv"))
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    ap.add_argument("--delete-zip", action="store_true")
    a = ap.parse_args()

    os.makedirs(a.data_dir, exist_ok=True)
    zip_path = os.path.join(a.data_dir, "ISIC_2019_Training_Input.zip")
    out_dir = os.path.join(a.data_dir, "ISIC_2019_Training_Input_preprocessed")
    os.makedirs(out_dir, exist_ok=True)

    names = pd.read_csv(a.split)["image"].astype(str).tolist()
    print(f"{len(names)} images listed in {a.split}")

    download(URL, zip_path)
    with zipfile.ZipFile(zip_path) as z:
        members = {os.path.splitext(os.path.basename(m))[0]: m for m in z.namelist() if m.lower().endswith(".jpg")}
    missing = [n for n in names if n not in members]
    if missing:
        sys.exit(f"{len(missing)} images not found in the zip, e.g. {missing[:3]}")

    jobs = [(members[n], os.path.join(out_dir, n + ".jpg")) for n in names]
    t0, done = time.time(), 0
    with Pool(a.workers, initializer=_init, initargs=(zip_path,)) as pool:
        for _ in pool.imap_unordered(_work, jobs, chunksize=32):
            done += 1
            if done % 500 == 0 or done == len(jobs):
                print(f"  preprocessed {done}/{len(jobs)}  ({time.time() - t0:.0f} s)")

    rows = []
    for n in names:
        p = os.path.join(out_dir, n + ".jpg")
        with Image.open(p) as im:
            w, h = im.size
        rows.append({"image": n, "width": w, "height": h, "sha256": sha256(p)})
    man = pd.DataFrame(rows)
    man.to_csv(os.path.join(a.data_dir, "manifest_preprocessed.csv"), index=False)
    bad = man[~man[["width", "height"]].min(axis=1).isin([SIZE - 1, SIZE])]  # int() rounding, as in FLamby
    print(f"\nimages: {len(man)} | shorter edge {SIZE - 1}-{SIZE}: {len(man) - len(bad)} | manifest saved")
    if len(man) == len(names) and bad.empty:
        print("ALL OK")
        if a.delete_zip:
            os.remove(zip_path)
            print("zip deleted")
    else:
        print("CHECK: some images are missing or have an unexpected size")


if __name__ == "__main__":
    main()
