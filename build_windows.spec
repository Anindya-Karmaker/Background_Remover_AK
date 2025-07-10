# -*- mode: python ; coding: utf-8 -*-

import os
from PyInstaller.utils.hooks import collect_data_files

# --- Configuration ---
APP_NAME = "Background Remover"
SCRIPT_FILE = "BACKGROUND_REMOVER_AK.py"
ICON_FILE = "assets/icon.ico"

# --- Path Definitions ---
rembg_models_path = os.path.join(os.path.expanduser("~"), ".u2net")

if not os.path.isdir(rembg_models_path):
    raise FileNotFoundError(
        f"The rembg models directory was not found at: {rembg_models_path}\n"
        "Please run the Python script once to download the models, then try compiling again."
    )

# --- Construct the list of data files ---
datas = [(rembg_models_path, '_internal/rembg_models')]

# Filter PySide6 files to include only those that actually exist, preventing errors
pyside_data_files = collect_data_files('PySide6')
for source_path, dest_path in pyside_data_files:
    if os.path.exists(source_path):
        datas.append((source_path, dest_path))

# --- PyInstaller Analysis ---
a = Analysis(
    [SCRIPT_FILE],
    pathex=[],
    binaries=[],
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
    # --- CORRECTED HOOK FILENAME ---
    runtime_hooks=['runtime_hook.py'],
    excludes=[],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=None,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=None)

# --- SINGLE FILE EXE CONFIGURATION ---
# The EXE object now includes all data and binaries. The COLLECT block is removed.
exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.zipfiles,
    a.datas,
    [],
    name=APP_NAME,
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    runtime_tmpdir=None,
    console=False, # This creates a GUI app without a command-line window
    icon=ICON_FILE
)