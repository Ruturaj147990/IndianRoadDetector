# PowerShell launcher for IRD ROCm GPU execution on AMD Radeon RX 7700 XT
$rocm_python = "E:\ComfyUI_windows_portable\python_embeded\python.exe"

if (Test-Path $rocm_python) {
    & $rocm_python $args
} else {
    Write-Warning "ROCm Python at $rocm_python not found. Falling back to system python..."
    & python $args
}
