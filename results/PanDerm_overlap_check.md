# Model B (PanDerm): pretraining-overlap check (30 Oct 2026, before E1)

Protocol v0.10, Section "Pretraining overlap": before E1 we check the PanDerm supplementary material
and ask the authors. If ISIC 2019 or HAM10000 images were used, model B is reported as possibly
contaminated and analysis 2f relies on the model A contrast only.

## What was checked

- PanDerm paper, arXiv 2410.15038 v1 and v3 (Methods, "Pretraining dataset"; Discussion).
- PanDerm GitHub README (github.com/SiyuanYan1/PanDerm).
- The Nature Medicine version (s41591-025-03747-y) could not be opened from our environment.
- **The authors were not contacted** (not possible for us). This deviates from the protocol.

## Findings

1. Pretraining used 2,149,706 images from 11 sources: in-house cohorts (MYM, HOP, MMT, ACEMID, NSSI,
   Edu1, Edu2; mainly Australia and New Zealand) and public sets ISIC2024 (352,034 tiles), TCGA-SKCM
   and UAH89k.
2. ISIC 2019, HAM10000, BCN20000 and PAD-UFES-20 are **not listed** as pretraining sources.
3. The paper gives **no deduplication step and no statement** that downstream evaluation data were
   excluded from pretraining. HAM10000 is one of PanDerm's own downstream evaluation sets.
4. Part of HAM10000 comes from a skin cancer practice in Queensland, Australia, and the in-house MMT
   set comes from clinics across Australia and New Zealand. Overlap of patients (not necessarily of
   images) cannot be ruled out from the published information.

## Decision (fixed before any model B run)

Overlap is **not shown but not ruled out**. Because the authors could not be asked, we apply the
protocol's conservative branch:

- Model B is reported as **"pretraining overlap not ruled out"** everywhere it appears.
- Analysis 2f (full fine-tuning vs frozen encoder) relies on the **model A vs model A-frozen**
  contrast; model B's result is reported next to it as supporting, not as evidence on its own.
- The primary outcome (Gate 2) uses model A only, so this decision does not touch it.
- Model B stays in E1, E2, E5 and E7 as planned.
