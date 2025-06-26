# -*- mode: python ; coding: utf-8 -*-

import os
import sys
# --- THIS IS THE FIX ---
# We will use a PyInstaller helper function to find the PySide6 data files correctly.
from PyInstaller.utils.hooks import collect_data_files
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

# --- Construct the list of data files ---
# Start with our manual data (the rembg models)
datas = [(rembg_models_path, '_internal/rembg_models')]

# --- THIS IS THE FIX (continued) ---
# Now, use the PyInstaller helper to automatically find and add all necessary
# PySide6 files (plugins, translations, etc.). This works reliably with Anaconda.
datas.extend(collect_data_files('PySide6'))
# -----------------------------------

# --- PyInstaller Analysis ---
a = Analysis(
    [SCRIPT_FILE],
    pathex=[],
    binaries=[],
    # Use the 'datas' list we constructed above
    datas=datas,
    hiddenimports=[
        'rembg.sessions.u2net',
        'rembg.sessions.u2netp',
        'rembg.sessions.u2net_human_seg',
        'rembg.sessions.silueta',
        'rembg.sessions.isnet_general_use',
        'rembg.sessions.isnet_anime',
        'onnxruntime.capi.onnxruntime_inference_sessions',
    ],
    hookspath=[],
    runtime_hooks=['runtime_hook.py'],
    excludes=[],
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