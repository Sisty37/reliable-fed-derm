"""Evaluation on per-center and pooled test sets (balanced accuracy as in FLamby)."""
import numpy as np
import torch
from sklearn.metrics import balanced_accuracy_score


@torch.no_grad()
def predict(model, loader, device, amp=True):
    model.eval()
    amp = amp and device.type == "cuda"
    logits, ys = [], []
    for x, y in loader:
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp):
            out = model(x.to(device, non_blocking=True))
        logits.append(out.float().cpu().numpy())
        ys.append(y.numpy())
    return np.concatenate(logits), np.concatenate(ys)


def bacc(y, logits):
    return float(balanced_accuracy_score(y, logits.argmax(1)))


def summarize(per_center):
    """per_center: dict center -> (logits, y). Returns metrics dict."""
    res = {f"bacc_center_{c}": bacc(y, lg) for c, (lg, y) in per_center.items()}
    allg = np.concatenate([lg for lg, _ in per_center.values()])
    ally = np.concatenate([y for _, y in per_center.values()])
    res["bacc_pooled"] = bacc(ally, allg)
    res["bacc_mean_of_centers"] = float(np.mean([res[f"bacc_center_{c}"] for c in per_center]))
    return res
