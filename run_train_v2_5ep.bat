@echo off
set MIOPEN_FIND_MODE=2
set PYTHONUNBUFFERED=1
cd /d "C:\Users\limbk\OneDrive\Desktop\YOLO\IndianRoadDetector"

set ROCM_PYTHON=E:\ComfyUI_windows_portable\python_embeded\python.exe

echo ==================================================================
echo      IRD V2 - 5-EPOCH FULL-DATASET STREAMING TRAINING RUN
echo ==================================================================
echo Hardware:   AMD Radeon RX 7700 XT 12GB (ROCm)
echo Epochs:     5
echo Batch Size: 8
echo Dataset:    thirdeyelabs/indian-road-dataset (646 Shards Streaming)
echo Validation: 25 Benchmark Clips Excluded (100%% Leakage Protected)
echo Output:     experiments/custom_model/v2_full_646k
echo ==================================================================

"%ROCM_PYTHON%" scripts/train_streaming_v2.py --epochs 5 --batch-size 8 --lr 0.001 --weight-decay 0.0001 --img-size 640 --cache-dir "data/cache_shards" --max-cached-shards 12 --prefetch-ahead 2 --workers 2 --output-dir "experiments/custom_model/v2_full_646k" --smoke-batches 100

echo ==================================================================
echo   RUNNING AUTHORITATIVE COCO EVALUATION ON 1,719 VAL IMAGES
echo ==================================================================
"%ROCM_PYTHON%" scripts/evaluate_ird.py --weights "experiments/custom_model/v2_full_646k/checkpoints/ird_v2_full_best.pt" --data "data/indian_road_yolo/data.yaml" --output-json "experiments/custom_model/v2_full_646k/eval_benchmark_results.json" --conf 0.25 --iou 0.40 --decoder-version "v2_smooth" --device "cuda"

pause
