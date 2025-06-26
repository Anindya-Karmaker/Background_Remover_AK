# -*- mode: python ; coding: utf-8 -*-

import os
import sys
import subprocess
from PySide6 import QtCore
from rembg import const

# --- Configuration ---
APP_NAME = "Background Remover"
SCRIPT_FILE = "BACKGROUND_REMOVER_AK.py"
ICON_FILE = "assets/icon.ico"
VERSION_FILE = "version_file.txt"

# --- Code Signing Configuration (EDIT THESE) ---
# Set to True to enable code signing. If False, the signing step will be skipped.
ENABLE_SIGNING = True
# Full path to your .pfx certificate file.
CERT_PATH = "C:\\certs\\my_certificate.pfx"
# The password for your certificate. It's safer to load this from an environment variable.
# In CMD: set CERT_PASS=your_password
# In PowerShell: $env:CERT_PASS="your_password"
CERT_PASS = os.environ.get("CERT_PASS")
# URL of the timestamp server. This is a common, free one.
TIMESTAMP_URL = "http://timestamp.sectigo.com"

# --- Helper function to find signtool.exe ---
def find_signtool():
    """Finds the path to signtool.exe from the Windows SDK."""
    # Common base paths for the Windows Kits
    base_paths = [
        os.path.join(os.environ.get("ProgramFiles(x86)", ""), "Windows Kits", "10", "bin"),
        os.path.join(os.environ.get("ProgramFiles", ""), "Windows Kits", "10", "bin"),
    ]
    for base in base_paths:
        if os.path.isdir(base):
            # The SDK versions are in subdirectories, find the latest one
            versions = sorted([d for d in os.listdir(base) if d.startswith("10.")], reverse=True)
            for v in versions:
                tool_path = os.path.join(base, v, "x64", "signtool.exe")
                if os.path.exists(tool_path):
                    print(f"Found signtool.exe at: {tool_path}")
                    return tool_path
    return None

# Get the directory where PySide6 stores its plugins
pyside_library = os.path.join(os.path.dirname(QtCore.__file__), "plugins")

# Get the directory where rembg stores its models
rembg_models_path = const.U2NET_HOME

# --- PyInstaller Analysis ---
a = Analysis(
    [SCRIPT_FILE],
    pathex=[],
    binaries=[],
    datas=[
        # Bundle the PySide6 Qt platform plugin for Windows
        (os.path.join(pyside_library, "platforms", "qwindows.dll"), "_internal\\PySide6\\plugins\\platforms"),
        (os.path.join(pyside_library, "styles", "qwindowsvistastyle.dll"), "_internal\\PySide6\\plugins\\styles"),
        # --- Bundle all downloaded rembg models ---
        (rembg_models_path, '_internal\\rembg_models')
    ],
    hiddenimports=[
        'rembg.sessions.u2net', 'rembg.sessions.u2netp', 'rembg.sessions.u2net_human_seg',
        'rembg.sessions.silueta', 'rembg.sessions.isnet_general_use', 'rembg.sessions.isnet_anime',
        'onnxruntime.capi.onnxruntime_inference_sessions',
        'scipy.special._cdflib', # Often needed by SciPy
    ],
    hookspath=[],
    # Add our custom runtime hook
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
    console=False,  # --- This creates a windowed app (no console) ---
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=ICON_FILE,
    version=VERSION_FILE, # --- Embed version info from our file ---
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name=APP_NAME,
)

# --- Post-Build Code Signing Step ---
if ENABLE_SIGNING:
    print("--- Starting Code Signing ---")
    signtool_path = find_signtool()
    if not signtool_path:
        raise FileNotFoundError("signtool.exe not found. Is the Windows SDK installed?")
    if not os.path.exists(CERT_PATH):
        raise FileNotFoundError(f"Certificate not found at: {CERT_PATH}")
    if not CERT_PASS:
        raise ValueError("Certificate password not set. Use 'set CERT_PASS=your_password'.")

    # The final executable is inside the COLLECT directory
    exe_path_to_sign = os.path.join(distpath, APP_NAME, f"{APP_NAME}.exe")
    
    command = [
        signtool_path,
        "sign",
        "/f", CERT_PATH,
        "/p", CERT_PASS,
        "/tr", TIMESTAMP_URL,
        "/td", "sha256",
        "/fd", "sha256",
        "/v", # Verbose output
        exe_path_to_sign,
    ]
    
    print(f"Signing command: {' '.join(command)}")
    try:
        subprocess.check_call(command)
        print("--- Code Signing Successful ---")
    except subprocess.CalledProcessError as e:
        print(f"--- Code Signing FAILED: {e} ---")
        # Fail the build if signing fails
        sys.exit(1)