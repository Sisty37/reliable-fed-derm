"""Write SHA-256 checksums of all split files so later changes can be detected."""
import hashlib
from pathlib import Path

root = Path(__file__).resolve().parents[1] / "splits"
lines = []
for f in sorted(root.glob("*.csv")):
    lines.append(f"{hashlib.sha256(f.read_bytes()).hexdigest()}  {f.name}")
(root / "SHA256SUMS.txt").write_text("\n".join(lines) + "\n")
print(f"{len(lines)} split files hashed -> {root / 'SHA256SUMS.txt'}")
