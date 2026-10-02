param(
    [switch]$KeepBuild
)

$ErrorActionPreference = "Stop"

$root = Split-Path -Parent $MyInvocation.MyCommand.Path
$python = Join-Path $root "tools\python312_portable\python.exe"
$dist = Join-Path $root "dist"
$name = "星布谷地钓鱼助手"

if (-not (Test-Path $python)) {
    throw "Portable Python not found: $python"
}
if (-not (Test-Path (Join-Path $root "runs\fishing_yolo11n_v1\weights\best.pt"))) {
    throw "YOLO model not found"
}

Push-Location $root
$pyinstallerCache = Join-Path $root "build\.pyinstaller-cache"
New-Item -ItemType Directory -Force -Path $pyinstallerCache | Out-Null
$previousPyInstallerConfig = $env:PYINSTALLER_CONFIG_DIR
$env:PYINSTALLER_CONFIG_DIR = $pyinstallerCache
$pyinstallerArgs = @(
    "--noconfirm",
    "--clean",
    "--windowed",
    "--uac-admin",
    "--name", $name,
    "--icon", "assets\app_icon.ico",
    "--add-data", "runs\fishing_yolo11n_v1\weights\best.pt;models",
    "--add-data", "templates;templates",
    "--add-data", "assets\app_icon.ico;assets",
    "--hidden-import", "opencv_feed_zero",
    "--collect-all", "dxcam",
    "--hidden-import", "PySide6.QtCore",
    "--hidden-import", "PySide6.QtGui",
    "--hidden-import", "PySide6.QtWidgets",
    "--hidden-import", "dxcam._libs.d3d11",
    "--hidden-import", "dxcam._libs.dxgi",
    "--hidden-import", "dxcam._libs.user32",
    "--hidden-import", "torchvision._C_stable",
    "--hidden-import", "torchvision.image_stable",
    "--hidden-import", "torchvision.extension",
    "--hidden-import", "torchvision.ops",
    "--hidden-import", "torchvision.ops.boxes",
    "--hidden-import", "lap",
    "--exclude-module", "tensorboard",
    "--exclude-module", "torch.utils.tensorboard",
    "--exclude-module", "triton",
    "--exclude-module", "polars",
    "--exclude-module", "matplotlib",
    "--exclude-module", "networkx",
    "--exclude-module", "fontTools",
    "--exclude-module", "pandas",
    "--exclude-module", "scipy",
    "--exclude-module", "tensorflow",
    "--exclude-module", "keras",
    "--exclude-module", "onnx",
    "--exclude-module", "onnxruntime",
    "--exclude-module", "onnxslim",
    "--exclude-module", "sympy",
    "--exclude-module", "ml_dtypes",
    "--exclude-module", "protobuf",
    "--exclude-module", "tkinter",
    "run_app.py"
)
$pyinstallerArgs = @("--onedir") + $pyinstallerArgs
$originalPath = $env:PATH
try {
    # Poppler ships ICU DLLs with version-suffixed exports that conflict with
    # Qt's ICU imports. Keep that external directory out of PyInstaller's
    # dependency search while still allowing the bundled package hooks to run.
    $env:PATH = (($originalPath -split ';') | Where-Object {
        $_ -and ($_ -notmatch '(?i)\\poppler(\\|$)')
    }) -join ';'
    & $python -m PyInstaller @pyinstallerArgs
    $buildExitCode = $LASTEXITCODE
}
finally {
    $env:PATH = $originalPath
    if ($null -eq $previousPyInstallerConfig) {
        Remove-Item Env:PYINSTALLER_CONFIG_DIR -ErrorAction SilentlyContinue
    } else {
        $env:PYINSTALLER_CONFIG_DIR = $previousPyInstallerConfig
    }
}
Pop-Location

if ($buildExitCode -ne 0) {
    throw "PyInstaller failed with exit code $buildExitCode"
}

Write-Host ""
Write-Host "Build complete: $dist\$name\$name.exe"

$releaseDir = Join-Path $dist $name
$releaseTorchLib = Join-Path $releaseDir "_internal\torch\lib"
$optionalCudaDlls = @(
    "cusolverMg64_11.dll",
    "curand64_10.dll",
    "cufftw64_11.dll",
    "nvperf_host.dll",
    "nvToolsExt64_1.dll",
    "libiompstubs5md.dll",
    "caffe2_nvrtc.dll"
)
$removedCudaDlls = @()
foreach ($dll in $optionalCudaDlls) {
    $path = Join-Path $releaseTorchLib $dll
    if (Test-Path -LiteralPath $path) {
        Remove-Item -LiteralPath $path -Force
        $removedCudaDlls += $dll
    }
}
if ($removedCudaDlls.Count -gt 0) {
    Write-Host "Removed optional CUDA DLLs: $($removedCudaDlls -join ', ')"
}

$bundledIcu = Get-ChildItem -LiteralPath (Join-Path $releaseDir "_internal") -Recurse -File -Filter "icu*.dll"
if ($bundledIcu) {
    $paths = ($bundledIcu | ForEach-Object { $_.FullName }) -join [Environment]::NewLine
    throw "Unexpected ICU DLLs were bundled. Remove the external ICU source before distributing:`n$paths"
}

if (-not $KeepBuild) {
    $buildDir = [IO.Path]::GetFullPath((Join-Path $root "build\$name"))
    $rootDir = [IO.Path]::GetFullPath($root).TrimEnd('\') + '\'
    if (-not $buildDir.StartsWith($rootDir, [StringComparison]::OrdinalIgnoreCase)) {
        throw "Refusing to remove build directory outside project root: $buildDir"
    }
    if (Test-Path -LiteralPath $buildDir) {
        Remove-Item -LiteralPath $buildDir -Recurse -Force
        Write-Host "Removed intermediate build directory: $buildDir"
    }
}

Write-Host "Portable folder: $releaseDir"
