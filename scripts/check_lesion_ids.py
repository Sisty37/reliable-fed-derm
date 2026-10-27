"""
Check whether lesion IDs are real or per-image placeholders, per center.
If a center has (almost) one image per lesion AND its IDs look generated per image,
its leakage rate is a lower bound.

Usage (from the repo folder):  python scripts\\check_lesion_ids.py splits\\split_official.csv
"""
import re
import sys

import pandas as pd

df = pd.read_csv(sys.argv[1] if len(sys.argv) > 1 else "splits/split_official.csv")
df["prefix"] = df.lesion_id.astype(str).str.extract(r"^([A-Za-z]+)_?", expand=False).fillna("?")
df["id_is_image"] = df.lesion_id.astype(str).str.replace("IMG_", "", regex=False) == df.image.astype(str)
rows = []
for c, g in df.groupby("center"):
    per = g.groupby("lesion_id").size()
    rows.append({
        "center": c, "images": len(g), "lesions": per.size,
        "images_per_lesion": round(len(g) / per.size, 3),
        "lesions_with_2plus_images": int((per >= 2).sum()),
        "id_equals_image_name": int(g.id_is_image.sum()),
        "id_prefixes": dict(g.prefix.value_counts().head(4)),
    })
out = pd.DataFrame(rows)
pd.set_option("display.width", 200)
print(out.to_string(index=False))
print("\nTotal:", len(df), "images |", df.lesion_id.nunique(), "lesions |",
      int(df.id_is_image.sum()), "IDs equal to the image name (placeholders)")
