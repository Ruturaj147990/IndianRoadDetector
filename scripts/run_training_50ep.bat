@echo off
set MIOPEN_FIND_MODE=2
set PYTHONUNBUFFERED=1
cd /d "c:\Users\limbk\OneDrive\Desktop\YOLO\IndianRoadDetector"

set ROCM_PYTHON=E:\ComfyUI_windows_portable\python_embeded\python.exe

echo ==================================================================
echo      IRD V1.5 - 50-EPOCH LOCAL PRODUCTION TRAINING RUN
echo ==================================================================
echo GPU: AMD Radeon RX 7700 XT 12GB
echo Batch Size: 12 (Safe VRAM Headroom: 4.94 GB)
echo Epochs: 50
echo ==================================================================

"%ROCM_PYTHON%" -u scripts/train_custom.py --data-dir data/indian_road_yolo --output-dir experiments/custom_model/final_training_50ep --epochs 50 --batch-size 12 --lr 0.001 --device cuda --workers 0 --matcher-version topk_adaptive_v2 --class-balanced-loss --use-atd --use-ssdp --use-fgbr --use-quality --use-cdg --use-aux-one2one
pause
