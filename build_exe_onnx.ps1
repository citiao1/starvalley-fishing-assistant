param(
    [ValidateSet("directml", "cuda")]
    [string]$Provider = "directml",
    [switch]$KeepBuild
)

$ErrorActionPreference = "Stop"

$root = Split-Path -Parent $MyInvocation.MyCommand.Path
$python = Join-Path $root "tools\python312_portable\python.exe"
$deps = if ($Provider -eq "directml") {
    "D:\Temp\starvalley_ort_dml_deps"
} else {
    "D:\Temp\starvalley_ort120_deps"
}
$dist = Join-Path $root "dist"
$name = "星布谷地钓鱼助手_ONNX_$Provider"

if (-not (Test-Path -LiteralPath $python)) {
    throw "Portable Python not found: $python"
}
if (-not (Test-Path -LiteralPath $deps)) {
    throw "ONNX Runtime dependency directory not found: $deps"
}
if (-not (Test-Path -LiteralPath (Join-Path $root "models\best.onnx"))) {
    throw "ONNX model not found"
}

Push-Location $root
$pyinstallerArgs = @(
    "--noconfirm",
    "--clean",
    "--onedir",
    "--windowed",
    "--uac-admin",
    "--name", $name,
    "--paths", $deps,
    "--icon", "assets\app_icon.ico",
    "--add-data", "models\best.onnx;models",
    "--add-data", "templates;templates",
    "--add-data", "assets\app_icon.ico;assets",
    "--hidden-import", "onnxruntime",
    "--hidden-import", "onnxruntime.capi.onnxruntime_pybind11_state",
    "--collect-all", "onnxruntime",
    "--hidden-import", "opencv_feed_zero",
    "--collect-all", "dxcam",
    "--hidden-import", "PySide6.QtCore",
    "--hidden-import", "PySide6.QtGui",
    "--hidden-import", "PySide6.QtWidgets",
    "--hidden-import", "dxcam._libs.d3d11",
    "--hidden-import", "dxcam._libs.dxgi",
    "--hidden-import", "dxcam._libs.user32",
    "--exclude-module", "torch",
    "--exclude-module", "torchvision",
    "--exclude-module", "ultralytics",
    "--exclude-module", "onnx",
    "--exclude-module", "onnxslim",
    "--exclude-module", "sympy",
    "--exclude-module", "matplotlib",
    "--exclude-module", "pandas",
    "--exclude-module", "scipy",
    "run_app_onnx.py"
)

$originalPath = $env:PATH
try {
    $env:PATH = (($originalPath -split ';') | Where-Object {
        $_ -and ($_ -notmatch '(?i)\\poppler(\\|$)')
    }) -join ';'
    $env:FISHING_ASSISTANT_BACKEND = "onnx"
    $env:FISHING_ASSISTANT_ONNX_PROVIDER = $Provider
    & $python -m PyInstaller @pyinstallerArgs
    $buildExitCode = $LASTEXITCODE
}
finally {
    $env:PATH = $originalPath
    Remove-Item Env:FISHING_ASSISTANT_BACKEND -ErrorAction SilentlyContinue
    Remove-Item Env:FISHING_ASSISTANT_ONNX_PROVIDER -ErrorAction SilentlyContinue
}
Pop-Location

if ($buildExitCode -ne 0) {
    throw "PyInstaller failed with exit code $buildExitCode"
}

$releaseDir = Join-Path $dist $name
Write-Host "ONNX build complete: $releaseDir"

if (-not $KeepBuild) {
    $buildDir = [IO.Path]::GetFullPath((Join-Path $root "build\$name"))
    $rootDir = [IO.Path]::GetFullPath($root).TrimEnd('\') + '\'
    if (-not $buildDir.StartsWith($rootDir, [StringComparison]::OrdinalIgnoreCase)) {
        throw "Refusing to remove build directory outside project root: $buildDir"
    }
    if (Test-Path -LiteralPath $buildDir) {
        Remove-Item -LiteralPath $buildDir -Recurse -Force
    }
}
