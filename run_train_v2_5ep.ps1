# ==============================================================================
# IRD V2 Production Training Launcher - 5 Epochs Full Dataset (~646k Frames)
# Target Hardware: AMD Radeon RX 7700 XT 12GB (Windows ROCm HIP)
# ==============================================================================
$ErrorActionPreference = 'Stop'

$env:MIOPEN_FIND_MODE = '2'
$env:PYTHONUNBUFFERED = '1'

$projectRoot = 'C:\Users\limbk\OneDrive\Desktop\YOLO\IndianRoadDetector'
Set-Location $projectRoot

$rocmPython = 'E:\ComfyUI_windows_portable\python_embeded\python.exe'

Write-Host '==================================================================' -ForegroundColor Cyan
Write-Host '     IRD V2 - 5-EPOCH FULL-DATASET STREAMING TRAINING RUN        ' -ForegroundColor Green
Write-Host '==================================================================' -ForegroundColor Cyan
Write-Host 'Hardware:   AMD Radeon RX 7700 XT 12GB (ROCm)' -ForegroundColor Yellow
Write-Host "Python:     $rocmPython" -ForegroundColor Yellow
Write-Host 'Epochs:     5' -ForegroundColor Yellow
Write-Host 'Batch Size: 8' -ForegroundColor Yellow
Write-Host 'Dataset:    thirdeyelabs/indian-road-dataset (646 Shards Streaming)' -ForegroundColor Yellow
Write-Host 'Validation: 25 Benchmark Clips Excluded (100% Leakage Protected)' -ForegroundColor Yellow
Write-Host 'Output:     experiments/custom_model/v2_full_646k' -ForegroundColor Yellow
Write-Host '==================================================================' -ForegroundColor Cyan

& $rocmPython scripts/train_streaming_v2.py `
    --epochs 5 `
    --batch-size 8 `
    --lr 0.001 `
    --weight-decay 0.0001 `
    --img-size 640 `
    --cache-dir 'data/cache_shards' `
    --max-cached-shards 12 `
    --prefetch-ahead 2 `
    --workers 2 `
    --output-dir 'experiments/custom_model/v2_full_646k' `
    --smoke-batches 100

Write-Host '==================================================================' -ForegroundColor Cyan
Write-Host '   RUNNING AUTHORITATIVE COCO EVALUATION ON 1,719 VAL IMAGES     ' -ForegroundColor Green
Write-Host '==================================================================' -ForegroundColor Cyan

& $rocmPython scripts/evaluate_ird.py `
    --weights 'experiments/custom_model/v2_full_646k/checkpoints/ird_v2_full_best.pt' `
    --data 'data/indian_road_yolo/data.yaml' `
    --output-json 'experiments/custom_model/v2_full_646k/eval_benchmark_results.json' `
    --conf 0.25 `
    --iou 0.40 `
    --decoder-version 'v2_smooth' `
    --device 'cuda'
