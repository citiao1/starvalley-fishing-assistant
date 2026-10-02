# -*- mode: python ; coding: utf-8 -*-
from PyInstaller.utils.hooks import collect_all

datas = [('runs/fishing_yolo11n_v1/weights/best.pt', 'models'), ('templates', 'templates')]
binaries = []
hiddenimports = ['opencv_feed_zero', 'PySide6.QtCore', 'PySide6.QtGui', 'PySide6.QtWidgets', 'dxcam._libs.d3d11', 'dxcam._libs.dxgi', 'dxcam._libs.user32', 'torchvision._C_stable', 'torchvision.image_stable', 'torchvision.extension', 'lap']
tmp_ret = collect_all('dxcam')
datas += tmp_ret[0]; binaries += tmp_ret[1]; hiddenimports += tmp_ret[2]
tmp_ret = collect_all('torchvision')
datas += tmp_ret[0]; binaries += tmp_ret[1]; hiddenimports += tmp_ret[2]


a = Analysis(
    ['run_app.py'],
    pathex=[],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=['tensorboard', 'torch.utils.tensorboard', 'triton', 'polars', 'matplotlib', 'networkx', 'fontTools', 'pandas', 'scipy', 'tkinter'],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='星布谷地钓鱼助手_CPU',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    uac_admin=True,
)
coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='星布谷地钓鱼助手_CPU',
)
