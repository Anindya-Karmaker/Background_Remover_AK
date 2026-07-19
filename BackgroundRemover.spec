# -*- mode: python ; coding: utf-8 -*-
#
# PyInstaller spec for Background Remover AK (Professional Edition)
#
# Build:
#   pip install -r requirements.txt pyinstaller
#   pyinstaller --noconfirm BackgroundRemover.spec
#
# Output:
#   macOS   -> dist/BackgroundRemover.app  (+ dist/BackgroundRemover/)
#   Windows -> dist/BackgroundRemover/BackgroundRemover.exe
#   Linux   -> dist/BackgroundRemover/BackgroundRemover
#
# Set ONEFILE = True below for a single-file executable (slower startup).

import sys
import os
from PyInstaller.utils.hooks import collect_all

ONEFILE = False                      # True -> single executable
APP_NAME = "BackgroundRemover"
SCRIPT   = "BACKGROUND_REMOVER_AK_BETA.py"

# --- Application icon (platform-appropriate) ---
# macOS wants .icns, Windows wants .ico. Both live in assets/.
if sys.platform == "darwin":
    ICON = os.path.join("assets", "icon.icns")
elif sys.platform.startswith("win"):
    ICON = os.path.join("assets", "icon.ico")
else:
    ICON = os.path.join("assets", "icon.ico")
if not os.path.exists(ICON):
    print(f"[spec] warning: icon '{ICON}' not found; building without an icon")
    ICON = None

# --- Optional code-signing (macOS) --------------------------------------
# Export these before building to produce a signed .app that passes
# Gatekeeper (then notarize with notarize_macos.sh):
#   export CODESIGN_IDENTITY="Developer ID Application: Your Name (TEAMID)"
# Entitlements are read from entitlements.plist when present.
CODESIGN_IDENTITY = os.environ.get("CODESIGN_IDENTITY") or None
ENTITLEMENTS = "entitlements.plist" if os.path.exists("entitlements.plist") else None

# --- Gather data files, binaries and hidden imports for tricky packages ---
# Ship the assets folder so the app can load its window icon at runtime
# (resource_path() reads assets/ from sys._MEIPASS when frozen).
datas, binaries, hiddenimports = [], [], []
if os.path.isdir("assets"):
    datas += [("assets", "assets")]
for pkg in ("rembg", "onnxruntime", "scipy", "cv2", "PIL", "numpy", "requests"):
    try:
        d, b, h = collect_all(pkg)
        datas += d
        binaries += b
        hiddenimports += h
    except Exception as e:
        print(f"[spec] warning: could not collect '{pkg}': {e}")

hiddenimports += [
    "scipy.ndimage",
    "onnxruntime.capi._pybind_state",
    "PySide6.QtSvg",
]

block_cipher = None

a = Analysis(
    [SCRIPT],
    pathex=[],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=["tkinter", "matplotlib", "PyQt5", "PyQt6"],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

if ONEFILE:
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
        upx_exclude=[],
        runtime_tmpdir=None,
        console=False,
        disable_windowed_traceback=False,
        argv_emulation=(sys.platform == "darwin"),
        target_arch=None,
        codesign_identity=CODESIGN_IDENTITY,
        entitlements_file=ENTITLEMENTS,
        icon=ICON,
    )
    coll = None
else:
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
        argv_emulation=(sys.platform == "darwin"),
        target_arch=None,
        codesign_identity=CODESIGN_IDENTITY,
        entitlements_file=ENTITLEMENTS,
        icon=ICON,
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

# --- macOS .app bundle ---
if sys.platform == "darwin":
    app = BUNDLE(
        exe if ONEFILE else coll,
        name=f"{APP_NAME}.app",
        icon=ICON,
        bundle_identifier="com.anindyakarmaker.BackgroundRemover",
        info_plist={
            "CFBundleName": "Background Remover AK",
            "CFBundleDisplayName": "Background Remover AK",
            "CFBundleShortVersionString": "3.0",
            "NSHighResolutionCapable": True,
        },
    )
