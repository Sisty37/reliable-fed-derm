# E8 (frozen encoders on the sibling-exposure splits) and E9 (capacity-matched two-layer heads for model B).
# Small GPU load; can run in a second terminal while run_e8_modelA.ps1 is running. About 7 h in total.
# From the repo folder:  powershell -ExecutionPolicy Bypass -File scripts\run_e8e9_linear.ps1
# Finished runs are skipped, so the script can be restarted after an interruption.
$ErrorActionPreference = "Stop"
function Run-Linear($enc, $m, $s, $split, $head, $tag) {
    $name = [System.IO.Path]::GetFileNameWithoutExtension($split)
    if (-not (Test-Path "runs\$name\$tag\metrics.json")) {
        python scripts\train_linear.py --encoder $enc --method $m --seed $s --split $split --head $head
        if ($LASTEXITCODE -ne 0) { throw "failed: $enc $m $head seed $s $split" }
    }
}
# E9: model B, two-layer heads for FedAvg, pooled and local (FedPer already has one), both split types, seeds 0-4
foreach ($kind in "image", "lesion") {
    foreach ($s in 0..4) {
        $split = "splits\e1\resplit_${kind}_seed${s}_val.csv"
        Run-Linear panderm pooled $s $split mlp "pandermfrozen_pooledmlp_e20_s$s"
        Run-Linear panderm local  $s $split mlp "pandermfrozen_localmlp_e20_s$s"
        Run-Linear panderm fedavg $s $split mlp "pandermfrozen_fedavgmlp_r9_s$s"
    }
}
# E8: frozen encoders on the sibling-exposure crossover splits, seeds 0-2, arms 1-2
foreach ($s in 0..2) {
    foreach ($arm in 1, 2) {
        $split = "splits\e8\sibexp_lesion_seed${s}_arm${arm}.csv"
        foreach ($enc in "effb0", "panderm") {
            Run-Linear $enc pooled $s $split linear "${enc}frozen_pooled_e20_s$s"
            Run-Linear $enc local  $s $split linear "${enc}frozen_local_e20_s$s"
            Run-Linear $enc fedavg $s $split linear "${enc}frozen_fedavg_r9_s$s"
        }
        Run-Linear panderm fedper $s $split linear "pandermfrozen_fedper_r9_s$s"
    }
}
Write-Host "E8/E9 linear runs done"
