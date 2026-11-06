# E8 model A (fine-tuned EfficientNet-B0, GPU, about 22 min per run, 24 runs = about 9 h).
# From the repo folder:  powershell -ExecutionPolicy Bypass -File scripts\run_e8_modelA.ps1
# Needs splits\e8 (python scripts\make_e8_splits.py). Seed by seed, so finished seeds are usable early.
# Finished runs are skipped, so the script can be restarted after an interruption.
$ErrorActionPreference = "Stop"
foreach ($s in 0..2) {
    foreach ($arm in 1, 2) {
        $split = "splits\e8\sibexp_lesion_seed${s}_arm${arm}.csv"
        $dir = "runs\sibexp_lesion_seed${s}_arm${arm}"
        foreach ($m in "fedavg", "fedper", "pooled", "local") {
            $tag = if ($m -in "pooled", "local") { "${m}_e20_s$s" } else { "${m}_r9_s$s" }
            if (-not (Test-Path "$dir\$tag\metrics.json")) {
                python scripts\train_fl.py --method $m --seed $s --split $split
                if ($LASTEXITCODE -ne 0) { throw "E8 model A failed: $m seed $s arm $arm" }
            }
        }
    }
}
Write-Host "E8 model A runs done"
