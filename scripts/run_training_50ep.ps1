# IRD V1.5 — 50-Epoch Local Production Training Launcher
# Hardware: AMD Radeon RX 7700 XT 12GB (Windows 11 / ROCm HIP)
$ErrorActionPreference = "Stop"

$env:MIOPEN_FIND_MODE = "2"
$env:PYTHONUNBUFFERED = "1"

$projectRoot = "c:\Users\limbk\OneDrive\Desktop\YOLO\IndianRoadDetector"
Set-Location $projectRoot

$rocmPython = "E:\ComfyUI_windows_portable\python_embeded\python.exe"
$outputDir = Join-Path $projectRoot "experiments\custom_model\final_training_50ep"

if (-not (Test-Path $outputDir)) {
    New-Item -ItemType Directory -Path $outputDir -Force | Out-Null
}

$logFile = Join-Path $outputDir "training_live.log"

Write-Host "==================================================================" -ForegroundColor Cyan
Write-Host "     IRD V1.5 - 50-EPOCH LOCAL PRODUCTION TRAINING RUN           " -ForegroundColor Green
Write-Host "==================================================================" -ForegroundColor Cyan
Write-Host "GPU:         AMD Radeon RX 7700 XT 12GB" -ForegroundColor Yellow
Write-Host "Python:      $rocmPython" -ForegroundColor Yellow
Write-Host "Batch Size:  12 (Verified safe: 6.67 GB peak VRAM, 4.94 GB headroom)" -ForegroundColor Yellow
Write-Host "Epochs:      50" -ForegroundColor Yellow
Write-Host "Output:      $outputDir" -ForegroundColor Yellow
Write-Host "Log:         $logFile" -ForegroundColor Yellow
Write-Host "==================================================================" -ForegroundColor Cyan

& $rocmPython -u scripts/train_custom.py `
    --data-dir data/indian_road_yolo `
    --output-dir experiments/custom_model/final_training_50ep `
    --epochs 50 `
    --batch-size 12 `
    --lr 0.001 `
    --device cuda `
    --workers 0 `
    --matcher-version topk_adaptive_v2 `
    --class-balanced-loss `
    --use-atd `
    --use-ssdp `
    --use-fgbr `
    --use-quality `
    --use-cdg `
    --use-aux-one2one | Tee-Object -FilePath $logFile
