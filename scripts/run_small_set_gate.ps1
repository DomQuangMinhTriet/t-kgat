# Small-set gate of decision 0016 (items 5, 6, 8, 9): run every model of configs/model/small/ on lastfm-w3,
# validation only, then write docs/small-set-gate.md. Run from the repo root in PowerShell:
#   .\scripts\run_small_set_gate.ps1
#   .\scripts\run_small_set_gate.ps1 -Device cuda      # if a GPU is available
# It is safe to stop (Ctrl+C) and start again: KGAT runs use --resume, and BPR-MF rows already in the registry are
# not logged twice. Nothing here reads a test file. Order: fast models first, so a problem shows up early.
param([string]$Device = "auto", [string]$Data = "configs/data/lastfm_w3.yaml")
$ErrorActionPreference = "Stop"
$tags = @(
    "bprmf_w3_none", "bprmf_w3_cosine",
    "kgat_nokg_w3_none", "kgat_nokg_w3_cosine",
    "kgat_w3_none", "kgat_w3_cosine",
    "kgat_pruned_w3_none", "kgat_pruned_w3_cosine"
)
foreach ($tag in $tags) {
    Write-Host "=== $tag ($(Get-Date -Format HH:mm:ss)) ==="
    $args_ = @("scripts/run.py", "--data", $Data, "--model", "configs/model/small/$tag.yaml", "--val-only", "--device", $Device)
    if ($tag -like "kgat*") { $args_ += "--resume" }
    python @args_
    if ($LASTEXITCODE -ne 0) { throw "run.py failed for $tag" }
}
python scripts/15_small_set_gate.py --data $Data
Write-Host "Done: see docs/small-set-gate.md."
