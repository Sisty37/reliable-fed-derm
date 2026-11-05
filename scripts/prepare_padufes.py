"""
PAD-UFES-20 for the E6 domain-shift stress test (protocol v0.11, Sections 2 and 5).

  * Reads data/padufes20/metadata.csv (2,298 images, 1,641 lesions, 1,373 patients) and the PNG images
    in data/padufes20/images/imgs_part_*/.
  * Preprocesses every image exactly like Fed-ISIC2019 (FLamby resize_images.py, as in
    prepare_fedisic.py): RGB, shorter edge resized to 224 px (bilinear, aspect kept), colour constancy
    (power 6), saved as JPEG in data/padufes20/preprocessed/<image>.jpg.
  * Maps the six PAD-UFES-20 diagnoses to the Fed-ISIC2019 class indices
    (MEL 0, NV 1, BCC 2, AK 3, BKL 4, DF 5, VASC 6, SCC 7): MEL->0, NEV->1, BCC->2, ACK->3, SEK->4, SCC->7.
  * Writes splits/padufes20.csv with columns image, patient_id, lesion_id, diagnostic, target,
    center (= 0, a placeholder) and fold (= test), so fedderm.data.IsicDataset can read it.

Usage (from the repo folder):  python scripts\\prepare_padufes.py
"""
import glob
import os
import sys

import pandas as pd
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from prepare_fedisic import resize_and_cc  # noqa: E402

MAP = {"MEL": 0, "NEV": 1, "BCC": 2, "ACK": 3, "SEK": 4, "SCC": 7}


def main():
    root = os.path.join("data", "padufes20")
    out_dir = os.path.join(root, "preprocessed")
    os.makedirs(out_dir, exist_ok=True)
    meta = pd.read_csv(os.path.join(root, "metadata.csv"))
    paths = {os.path.basename(p): p for p in glob.glob(os.path.join(root, "images", "imgs_part_*", "*.png"))}
    assert len(meta) == 2298 and meta.img_id.is_unique, f"expected 2,298 unique images, got {len(meta)}"
    missing = [i for i in meta.img_id if i not in paths]
    assert not missing, f"{len(missing)} images listed in metadata.csv not found, e.g. {missing[:3]}"
    assert set(meta.diagnostic) <= set(MAP), f"unknown diagnoses {set(meta.diagnostic) - set(MAP)}"

    for k, img_id in enumerate(meta.img_id):
        dst = os.path.join(out_dir, img_id.replace(".png", ".jpg"))
        if not os.path.exists(dst):
            resize_and_cc(Image.open(paths[img_id]).convert("RGB")).save(dst)
        if (k + 1) % 500 == 0:
            print(f"  {k + 1}/{len(meta)} images preprocessed")

    df = pd.DataFrame({"image": meta.img_id.str.replace(".png", "", regex=False),
                       "patient_id": meta.patient_id, "lesion_id": meta.lesion_id,
                       "diagnostic": meta.diagnostic, "target": meta.diagnostic.map(MAP),
                       "center": 0, "fold": "test"})
    os.makedirs("splits", exist_ok=True)
    df.to_csv(os.path.join("splits", "padufes20.csv"), index=False)
    print(f"saved splits/padufes20.csv: {len(df)} images, {df.lesion_id.nunique()} lesions, "
          f"{df.patient_id.nunique()} patients")
    print("images per diagnosis: " + ", ".join(f"{k} {v}" for k, v in df.diagnostic.value_counts().items()))
    n_multi = (df.groupby("lesion_id").patient_id.nunique() > 1).sum()
    print(f"lesions with more than one patient ID: {n_multi} (must be 0)")


if __name__ == "__main__":
    main()
