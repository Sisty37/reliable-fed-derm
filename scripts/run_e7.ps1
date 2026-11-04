# E7 runs (protocol v0.11, E7). From the repo folder:  powershell -ExecutionPolicy Bypass -File scripts\run_e7.ps1
# Model B (FedPer-style head on PanDerm embeddings, minutes) first, then model A FedPer (GPU, hours).
# Finished runs are skipped, so the script can be restarted after an interruption.
$ErrorActionPreference = "Stop"
foreach ($kind in "image", "lesion") {
    foreach ($s in 0..4) {
        $split = "splits\e1\resplit_${kind}_seed${s}_val.csv"
        $outB = "runs\resplit_${kind}_seed${s}_val\pandermfrozen_fedper_r9_s$s\metrics.json"
        if (-not (Test-Path $outB)) {
            python scripts\train_linear.py --encoder panderm --method fedper --seed $s --split $split
            if ($LASTEXITCODE -ne 0) { throw "model B failed: $kind seed $s" }
        }
    }
}
foreach ($kind in "image", "lesion") {
    foreach ($s in 0..4) {
        $split = "splits\e1\resplit_${kind}_seed${s}_val.csv"
        $outA = "runs\resplit_${kind}_seed${s}_val\fedper_r9_s$s\metrics.json"
        if (-not (Test-Path $outA)) {
            python scripts\train_fl.py --method fedper --seed $s --split $split
            if ($LASTEXITCODE -ne 0) { throw "model A failed: $kind seed $s" }
        }
    }
}
Write-Host "E7 runs done"
