# Post hoc training-budget controls (review points 50-51; protocol deviations table, 8 Nov 2026).
# Not a pre-specified test: results are reported as descriptive sensitivity analyses.
#   1. FedAvg-FT local fine-tuning steps: 1, 10, 30 (protocol: 3) for model A; 1, 10, 30, 100 for the frozen encoders (3 is added for A-frozen, which had no FedAvg-FT run)
#   2. Matched-budget control for the frozen encoders: FedAvg with 27 rounds (protocol: 9), pooled and
#      local-only with 5 and 60 epochs (protocol: 20; the 20-epoch runs are added where missing)
# Seeds 0-2, image-level and lesion-grouped splits. Needs the existing FedAvg runs (fedavg_r9) of each split.
# From the repo folder:  powershell -ExecutionPolicy Bypass -File scripts\run_budget_ft.ps1
# Finished runs are skipped, so the script can be restarted after an interruption.
# Then:  python scripts\budget_analysis.py
$ErrorActionPreference = "Stop"
function Done($split, $tag) {
    $name = [System.IO.Path]::GetFileNameWithoutExtension($split)
    return (Test-Path "runs\$name\$tag\metrics.json")
}
foreach ($s in 0..2) {
    foreach ($kind in "image", "lesion") {
        $split = "splits\e1\resplit_${kind}_seed${s}_val.csv"
        # 1a. model A FedAvg-FT steps (GPU, a few minutes each)
        foreach ($st in 1, 10, 30) {
            if (-not (Done $split "fedavgft${st}_r9_s$s")) {
                python scripts\finetune_fedavg.py --split $split --seed $s --steps $st
                if ($LASTEXITCODE -ne 0) { throw "failed: model A fedavgft $st seed $s $kind" }
            }
        }
        foreach ($enc in "effb0", "panderm") {
            # 1b. frozen encoders, FedAvg-FT steps
            foreach ($st in 1, 3, 10, 30, 100) {
                $ft = if ($st -eq 3) { "fedavgft" } else { "fedavgft$st" }
                if (-not (Done $split "${enc}frozen_${ft}_r9_s$s")) {
                    python scripts\train_linear.py --encoder $enc --method fedavgft --ft-steps $st --seed $s --split $split
                    if ($LASTEXITCODE -ne 0) { throw "failed: $enc fedavgft $st seed $s $kind" }
                }
            }
            # 2. matched-budget control
            if (-not (Done $split "${enc}frozen_fedavg_r27_s$s")) {
                python scripts\train_linear.py --encoder $enc --method fedavg --rounds 27 --seed $s --split $split
                if ($LASTEXITCODE -ne 0) { throw "failed: $enc fedavg r27 seed $s $kind" }
            }
            foreach ($m in "pooled", "local") {
                foreach ($ep in 5, 20, 60) {
                    if (-not (Done $split "${enc}frozen_${m}_e${ep}_s$s")) {
                        python scripts\train_linear.py --encoder $enc --method $m --epochs $ep --seed $s --split $split
                        if ($LASTEXITCODE -ne 0) { throw "failed: $enc $m e$ep seed $s $kind" }
                    }
                }
            }
        }
    }
}
Write-Host "budget / fine-tuning runs done; now run: python scripts\budget_analysis.py"
