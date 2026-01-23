# -*- mode: python ; coding: utf-8 -*-

import os
import sys
# We will use PyInstaller helper functions to find the package data/lib files correctly.
from PyInstaller.utils.hooks import collect_data_files, collect_submodules, collect_dynamic_libs, collect_all
# ----------------------

# --- Configuration ---
APP_NAME = "Background Remover"
SCRIPT_FILE = "BACKGROUND_REMOVER_AK.py"
ICON_FILE = "assets/icon.icns"
BUNDLE_ID = "com.anindyakarmaker.backgroundremoverak"

# --- Path Definitions ---
# Manually define the rembg models path (this part is still correct)
rembg_models_path = os.path.join(os.path.expanduser("~"), ".u2net")

# Check if the models directory actually exists
if not os.path.isdir(rembg_models_path):
    raise FileNotFoundError(
        f"The rembg models directory was not found at: {rembg_models_path}\n"
        "Please run the Python script once (python BACKGROUND_REMOVER_AK.py) to "
        "allow rembg to download the necessary models, then try compiling again."
    )

# --- Robust dependency collection using collect_all ---
# This helper collects submodules, data files, and binaries all at once.
# It is the most reliable way to ensure complex packages like rembg and onnxruntime are bundled.

rembg_extra = collect_all('rembg')
onnx_extra = collect_all('onnxruntime')
torch_audio_extra = collect_all('torchaudio')
click_extra = collect_all('click')
onnx_lib_extra = collect_all('onnx')
pooch_extra = collect_all('pooch')
pymatting_extra = collect_all('pymatting') # Essential for alpha matting
scipy_extra = collect_all('scipy') # Essential submodules for pymatting

datas = [(rembg_models_path, '_internal/rembg_models')]
datas.extend(collect_data_files('PySide6'))
datas.extend(rembg_extra[0])
datas.extend(onnx_extra[0])
datas.extend(torch_audio_extra[0])
datas.extend(click_extra[0])
datas.extend(onnx_lib_extra[0])
datas.extend(pooch_extra[0])
datas.extend(pymatting_extra[0])
datas.extend(scipy_extra[0])

binaries = collect_dynamic_libs('onnxruntime')
binaries.extend(rembg_extra[1])
binaries.extend(onnx_extra[1])
binaries.extend(torch_audio_extra[1])
binaries.extend(click_extra[1])
binaries.extend(onnx_lib_extra[1])
binaries.extend(pooch_extra[1])
binaries.extend(pymatting_extra[1])
binaries.extend(scipy_extra[1])

# Explicitly add the Python dynamic library (required for some Conda environments on MacOS)
python_lib = os.path.join(sys.prefix, 'lib', 'libpython3.10.dylib')
if os.path.exists(python_lib):
    binaries.append((python_lib, '.'))
else:
    # Fallback to absolute path if sys.prefix doesn't match
    fallback_lib = '/opt/anaconda3/envs/bgremover/lib/libpython3.10.dylib'
    if os.path.exists(fallback_lib):
        binaries.append((fallback_lib, '.'))

hiddenimports = [
    'rembg',
    'onnxruntime',
    'torchaudio',
    'PySide6.QtCore',
    'PySide6.QtWidgets',
    'PySide6.QtGui',
    'pooch',
    'pymatting',
    'scipy.ndimage',
    'click',
    'onnx',
]
hiddenimports.extend(rembg_extra[2])
hiddenimports.extend(onnx_extra[2])
hiddenimports.extend(torch_audio_extra[2])
hiddenimports.extend(click_extra[2])
hiddenimports.extend(onnx_lib_extra[2])
hiddenimports.extend(pooch_extra[2])
hiddenimports.extend(pymatting_extra[2])
hiddenimports.extend(scipy_extra[2])
# ----------------------------------------------------

# --- PyInstaller Analysis ---
a = Analysis(
    [SCRIPT_FILE],
    pathex=[],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    runtime_hooks=['runtime_hook.py'],
    excludes=[
        'PyQt5', 'PyQt6', 'tkinter', 'matplotlib', 'pandas', 'notebook', 'jupyter',
        'numpy.testing', 'doctest', 'unittest',

        # onnxruntime providers we are not using (we only want CPU).
        # This prevents bundling large GPU-specific binaries.
        'onnxruntime.providers.cuda', 'onnxruntime.providers.tensorrt',
        'onnxruntime.providers.dnnl', 'onnxruntime.providers.openvino',
        'onnxruntime.providers.directml',
    ],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=None,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=None)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name=APP_NAME,
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,
    disable_windowed_traceback=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=ICON_FILE
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name=APP_NAME
)

app = BUNDLE(
    coll,
    name=f'{APP_NAME}.app',
    icon=ICON_FILE,
    bundle_identifier=BUNDLE_ID,
    info_plist={
        'NSHighResolutionCapable': 'True',
        'LSMinimumSystemVersion': '10.15',
        'CFBundleShortVersionString': '1.0.0',
        'CFBundleVersion': '1.0.0',
        'NSHumanReadableCopyright': 'Copyright © 2025 Anindya Karmaker. All rights reserved.',
    }
)