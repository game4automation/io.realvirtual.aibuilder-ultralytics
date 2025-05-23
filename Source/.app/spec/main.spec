# -*- mode: python ; coding: utf-8 -*-

from PyInstaller.utils.hooks import collect_data_files

import ultralytics
data_files = collect_data_files('ultralytics')


from PyInstaller.utils.hooks import collect_dynamic_libs
binaries = collect_dynamic_libs('onnxruntime', destdir='onnxruntime/capi')

a = Analysis(
    ['../src/main.py'],
    pathex=[],
    binaries=binaries,
    datas=data_files,
    hiddenimports=[
        'tkinter', 
        'onnx', 
        'onnxslim',
        'onnxruntime', 
        'onnxruntime.capi._pybind_state',
        'ultralytics' 
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=True,  # Use True to skip archiving for faster builds
    optimize=0,
)

pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='AiBuilder',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon='../src/logo.ico'
)
coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='main',
)
