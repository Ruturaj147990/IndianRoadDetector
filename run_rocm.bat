@echo off
set ROCM_PYTHON=E:\ComfyUI_windows_portable\python_embeded\python.exe

if exist "%ROCM_PYTHON%" (
    "%ROCM_PYTHON%" %*
) else (
    echo [WARNING] ROCm Python at %ROCM_PYTHON% not found. Falling back to system python...
    python %*
)
