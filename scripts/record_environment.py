"""
Records the software environment and checksums of the external inputs (review: reproducibility).

Writes
  requirements-lock.txt          exact versions of every installed Python package (pip freeze)
  results/Environment.json       Python, OS, PyTorch/CUDA/GPU, git commit, and SHA-256 of
                                 the PanDerm-Base checkpoint, the EfficientNet-B0 weights cache (if found),
                                 the Fed-ISIC2019 metadata files and the PAD-UFES-20 metadata

Usage (from the repo folder, in the environment used for the experiments):
  python scripts\\record_environment.py
  python scripts\\record_environment.py --panderm-ckpt D:\\path\\to\\panderm_bb_data6_checkpoint-499.pth
"""
import argparse
import glob
import hashlib
import json
import os
import platform
import subprocess
import sys


def sha256(path, chunk=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(chunk), b""):
            h.update(b)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--panderm-ckpt", default=os.path.join("..", "models", "panderm", "panderm_bb_data6_checkpoint-499.pth"))
    a = ap.parse_args()

    env = {"python": sys.version, "platform": platform.platform(), "machine": platform.machine()}
    try:
        env["git_commit"] = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        env["git_commit"] = None
    try:
        import torch
        env["torch"] = torch.__version__
        env["cuda"] = torch.version.cuda
        env["cudnn"] = torch.backends.cudnn.version()
        env["gpu"] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    except Exception as e:
        env["torch"] = f"not importable: {e}"
    for mod in ("numpy", "pandas", "scipy", "timm", "torchvision", "sklearn", "flamby"):
        try:
            env[mod] = __import__(mod).__version__
        except Exception:
            env[mod] = None

    files = {}
    candidates = [a.panderm_ckpt]
    candidates += glob.glob(os.path.join(os.path.expanduser("~"), ".cache", "torch", "hub", "checkpoints", "*efficientnet*"))
    candidates += glob.glob(os.path.join("data", "fedisic", "*.csv"))
    candidates += glob.glob(os.path.join("data", "padufes20", "*.csv")) + glob.glob(os.path.join("data", "pad*", "metadata.csv"))
    for p in candidates:
        if os.path.isfile(p):
            files[os.path.normpath(p)] = {"sha256": sha256(p), "bytes": os.path.getsize(p)}
            print(f"  {files[os.path.normpath(p)]['sha256']}  {p}")
        else:
            print(f"  not found: {p}")
    env["sha256"] = files

    lock = subprocess.check_output([sys.executable, "-m", "pip", "freeze"], text=True)
    open("requirements-lock.txt", "w", encoding="utf-8").write(lock)
    os.makedirs("results", exist_ok=True)
    json.dump(env, open(os.path.join("results", "Environment.json"), "w"), indent=2)
    print(f"saved requirements-lock.txt ({len(lock.splitlines())} packages) and results/Environment.json")


if __name__ == "__main__":
    main()
