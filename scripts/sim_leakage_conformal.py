"""
Synthetic simulation: how lesion-level leakage distorts accuracy and
conformal coverage estimates.

Each lesion has a latent position (class mean + lesion-specific offset);
each image of a lesion is that position plus small noise. A flexible model
can memorise lesion identity, which is what leakage rewards.

For each split type we train, calibrate split-conformal (LAC score) at
90% target, and report:
  - accuracy and coverage measured on the split's own test set ("reported")
  - accuracy and coverage on brand-new lesions ("deployment")
"""
import numpy as np
from sklearn.ensemble import RandomForestClassifier

rng0 = np.random.default_rng(0)
K, D = 4, 10                      # classes, feature dimension
CLASS_SEP, LESION_SD, IMG_SD = 1.0, 1.0, 0.25
IMGS_PER_LESION = [1, 2, 3, 4]    # sampled uniformly (FLamby BCN ~3.4/lesion)
ALPHA = 0.10
class_means = rng0.normal(0, CLASS_SEP, size=(K, D))


def make_lesions(n_lesions, rng, start_id=0):
    X, y, g = [], [], []
    for i in range(n_lesions):
        c = rng.integers(K)
        mu = class_means[c] + rng.normal(0, LESION_SD, D)
        for _ in range(rng.choice(IMGS_PER_LESION)):
            X.append(mu + rng.normal(0, IMG_SD, D)); y.append(c); g.append(start_id + i)
    return np.array(X), np.array(y), np.array(g)


def split(y, g, how, rng, fracs=(0.6, 0.2, 0.2)):
    if how == "image":
        idx = rng.permutation(len(y))
    else:  # lesion-grouped
        les = rng.permutation(np.unique(g))
        idx = np.concatenate([np.where(g == l)[0] for l in les])
    n = len(idx); a = int(fracs[0] * n); b = a + int(fracs[1] * n)
    return idx[:a], idx[a:b], idx[b:]


def conformal(model, Xc, yc, alpha):
    p = model.predict_proba(Xc)
    s = 1 - p[np.arange(len(yc)), yc]
    n = len(s)
    q = np.quantile(s, min(1, np.ceil((n + 1) * (1 - alpha)) / n), method="higher")
    return q


def evaluate(model, q, X, y):
    p = model.predict_proba(X)
    sets = (1 - p) <= q
    cov = sets[np.arange(len(y)), y].mean()
    return (p.argmax(1) == y).mean(), cov, sets.sum(1).mean()


rows = []
for seed in range(20):
    rng = np.random.default_rng(seed + 1)
    X, y, g = make_lesions(1500, rng)
    Xn, yn, _ = make_lesions(1500, rng, start_id=10**6)   # new lesions
    for how in ["image", "lesion"]:
        tr, ca, te = split(y, g, how, rng)
        m = RandomForestClassifier(n_estimators=200, min_samples_leaf=1, random_state=seed, n_jobs=-1)
        m.fit(X[tr], y[tr])
        q = conformal(m, X[ca], y[ca], ALPHA)
        acc_r, cov_r, sz_r = evaluate(m, q, X[te], y[te])
        acc_d, cov_d, sz_d = evaluate(m, q, Xn, yn)
        rows.append((how, acc_r, cov_r, sz_r, acc_d, cov_d, sz_d))

rows = np.array(rows, dtype=object)
print(f"Target coverage {1-ALPHA:.0%}; 20 seeds; mean (sd)\n")
print(f"{'split':8s} {'reported acc':>14s} {'reported cov':>14s} {'set size':>9s} | "
      f"{'deploy acc':>12s} {'deploy cov':>12s} {'set size':>9s}")
for how in ["image", "lesion"]:
    r = rows[rows[:, 0] == how][:, 1:].astype(float)
    m, s = r.mean(0), r.std(0)
    print(f"{how:8s} {m[0]:8.3f} ({s[0]:.3f}) {m[1]:8.3f} ({s[1]:.3f}) {m[2]:9.2f} | "
          f"{m[3]:7.3f} ({s[3]:.3f}) {m[4]:7.3f} ({s[4]:.3f}) {m[5]:9.2f}")


# Dose-response: vary the number of images per lesion (10 seeds each)
print("\nDose-response over images per lesion (image-level split; 10 seeds):")
for ipl in [[1], [1, 2], [1, 2, 3, 4], [2, 3, 4, 5, 6]]:
    IMGS_PER_LESION = ipl
    out = []
    for seed in range(10):
        rng = np.random.default_rng(seed + 1)
        X, y, g = make_lesions(1500, rng)
        Xn, yn, _ = make_lesions(1500, rng, start_id=10**6)
        tr, ca, te = split(y, g, "image", rng)
        m = RandomForestClassifier(n_estimators=150, random_state=seed, n_jobs=-1).fit(X[tr], y[tr])
        q = conformal(m, X[ca], y[ca], ALPHA)
        a, b = evaluate(m, q, X[te], y[te]), evaluate(m, q, Xn, yn)
        out.append([a[0], a[1], b[0], b[1]])
    r = np.mean(out, 0)
    print(f"  mean {np.mean(ipl):.1f} images/lesion: reported acc {r[0]:.3f}, cov {r[1]:.3f} | "
          f"new lesions acc {r[2]:.3f}, cov {r[3]:.3f}")
