# --- BACKGROUND REMOVER AK (Professional Edition) ---
#
# A professional-grade background removal & image editing tool.
#
# Features:
#   - Branded startup splash screen with staged loading.
#   - Modern, professional themed UI (Fusion + custom QSS).
#   - AI background removal (rembg) with alpha-matting controls.
#   - Manual refinement: Keep/Remove brushes, Magic Wand, color removal, crop.
#   - Perspective correction (4-point quadrilateral & 6-point curvature).
#   - Image overlay / compositing (opacity, scale, rotate, blend modes).
#   - Export to PDF (A4/Letter/Original, orientation, DPI, margins).
#   - Non-destructive background fill, undo/redo history, zoom & fit.

import sys
import io
import os
import math
import time
import base64
import tempfile

import numpy as np

from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QLabel, QVBoxLayout, QHBoxLayout,
    QPushButton, QFileDialog, QSplitter, QMenu, QMenuBar,
    QMessageBox, QColorDialog, QSpinBox, QDoubleSpinBox, QSizePolicy,
    QCheckBox, QComboBox, QFormLayout, QDialog, QStatusBar, QRadioButton,
    QGroupBox, QProgressDialog, QScrollArea, QTabWidget, QSlider,
    QSplashScreen, QStyle, QDialogButtonBox, QGridLayout, QFrame,
    QListWidget, QListWidgetItem, QAbstractItemView, QProgressBar,
    QTextBrowser, QLineEdit
)
from PySide6.QtGui import (
    QPixmap, QImage, QPainter, QPen, QBrush, QColor, QCursor, QIcon,
    QKeySequence, QAction, QActionGroup, QDesktopServices, QLinearGradient,
    QFont, QPainterPath, QPolygon, QTransform
)
from PySide6.QtCore import (
    Qt, QPoint, QPointF, QRect, QBuffer, QByteArray, QSize, Signal, QUrl,
    QMimeData, QTimer, QThread, QSettings
)

# --- Optional / heavy imports (loaded lazily during splash) ---
REMBG_AVAILABLE = False
remove_bg = None

try:
    from scipy.ndimage import label as scipy_label
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False

try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False

from PIL import Image, ImageDraw, ImageQt, ImageFilter

APP_NAME = "Background Remover"
APP_TAGLINE = "Professional Edition"
APP_VERSION = "3.0"
APP_AUTHOR = "Anindya Karmaker"
APP_ORG = "AnindyaKarmaker"          # QSettings organisation key
ACCENT = "#2d7ff9"
ACCENT_DARK = "#1e63cf"
GITHUB_URL = "https://github.com/Anindya-Karmaker/Background_Remover_AK"
# GitHub "latest release" endpoint used by the optional update checker.
UPDATE_API_URL = "https://api.github.com/repos/Anindya-Karmaker/Background_Remover_AK/releases/latest"

# Standard page sizes in millimetres (portrait: width, height)
PAGE_SIZES_MM = {
    "A3": (297, 420),
    "A4": (210, 297),
    "A5": (148, 210),
    "A6": (105, 148),
    "Letter": (215.9, 279.4),
    "Legal": (215.9, 355.6),
    "Tabloid": (279.4, 431.8),
}

# Platform-appropriate UI font (avoids missing-font warnings)
if sys.platform == "darwin":
    UI_FONT = "Helvetica Neue"
elif sys.platform.startswith("win"):
    UI_FONT = "Segoe UI"
else:
    UI_FONT = "DejaVu Sans"


def get_app_dir():
    """Directory of the app (next to the executable when frozen)."""
    if getattr(sys, "frozen", False):
        return os.path.dirname(sys.executable)
    return os.path.dirname(os.path.abspath(__file__))


def resource_path(*parts):
    """Absolute path to a bundled read-only resource.

    Works both when running from source and when frozen by PyInstaller,
    where bundled data lives under sys._MEIPASS.
    """
    base = getattr(sys, "_MEIPASS", None) or os.path.dirname(os.path.abspath(__file__))
    return os.path.join(base, *parts)


def get_app_icon():
    """The application QIcon loaded from the bundled assets folder.

    Prefers the multi-resolution .ico (best on Windows/Linux); falls back to
    the .icns. Cached so it is built only once. Returns an empty QIcon if the
    asset is missing so callers never have to guard against None.
    """
    global _APP_ICON
    if _APP_ICON is not None:
        return _APP_ICON
    icon = QIcon()
    for name in ("icon.ico", "icon.icns", "icon.png"):
        path = resource_path("assets", name)
        if os.path.exists(path):
            candidate = QIcon(path)
            if not candidate.isNull():
                icon = candidate
                break
    _APP_ICON = icon
    return _APP_ICON


_APP_ICON = None


def _version_tuple(v):
    """Parse a version string like 'v3.1.0' / '3.0 BETA' into a comparable
    tuple of integers, ignoring any non-numeric suffix."""
    import re
    nums = re.findall(r"\d+", str(v))
    return tuple(int(n) for n in nums) if nums else (0,)


def _is_writable(directory):
    try:
        os.makedirs(directory, exist_ok=True)
        test = os.path.join(directory, ".write_test")
        with open(test, "w"):
            pass
        os.remove(test)
        return True
    except Exception:
        return False


def get_models_dir():
    """A 'models' folder next to the app if writable, else a user folder."""
    preferred = os.path.join(get_app_dir(), "models")
    if _is_writable(preferred):
        return preferred
    fallback = os.path.join(os.path.expanduser("~"), ".BackgroundRemoverAK", "models")
    os.makedirs(fallback, exist_ok=True)
    return fallback


# rembg reads U2NET_HOME to decide where to store / load model files.
MODELS_DIR = get_models_dir()
os.environ["U2NET_HOME"] = MODELS_DIR

# Available rembg models. The birefnet-* and sam models give noticeably
# cleaner edges (hair/fur, fine detail) than the classic u2net family but
# are larger downloads. rembg fetches each on first use into U2NET_HOME.
REMBG_MODELS = [
    "u2net", "u2netp", "u2net_human_seg", "u2net_cloth_seg", "silueta",
    "isnet-general-use", "isnet-anime",
    "birefnet-general", "birefnet-general-lite", "birefnet-portrait",
    "birefnet-massive", "sam",
]

# OpenCV (dnn_superres) CNN super-resolution models. Each entry maps a
# scale factor to (local filename, download URL). Models are fetched on
# demand into MODELS_DIR; when none is available the code falls back to
# high-quality Lanczos + sharpening.
#
#   FSRCNN  — tiny & fast, good for a quick clean upscale        (×2/×3/×4)
#   ESPCN   — fast, a touch sharper than FSRCNN                  (×2/×3/×4)
#   EDSR    — highest quality, large model / slower              (×2/×3/×4)
#   LapSRN  — good quality, supports very large factors          (×2/×4/×8)
SR_MODELS = {
    "FSRCNN": {
        "label": "FSRCNN — fast & light",
        "arch": "fsrcnn",
        "scales": {
            2: ("FSRCNN_x2.pb", "https://github.com/Saafke/FSRCNN_Tensorflow/raw/master/models/FSRCNN_x2.pb"),
            3: ("FSRCNN_x3.pb", "https://github.com/Saafke/FSRCNN_Tensorflow/raw/master/models/FSRCNN_x3.pb"),
            4: ("FSRCNN_x4.pb", "https://github.com/Saafke/FSRCNN_Tensorflow/raw/master/models/FSRCNN_x4.pb"),
        },
    },
    "ESPCN": {
        "label": "ESPCN — fast, crisp",
        "arch": "espcn",
        "scales": {
            2: ("ESPCN_x2.pb", "https://github.com/fannymonori/TF-ESPCN/raw/master/export/ESPCN_x2.pb"),
            3: ("ESPCN_x3.pb", "https://github.com/fannymonori/TF-ESPCN/raw/master/export/ESPCN_x3.pb"),
            4: ("ESPCN_x4.pb", "https://github.com/fannymonori/TF-ESPCN/raw/master/export/ESPCN_x4.pb"),
        },
    },
    "EDSR": {
        "label": "EDSR — best quality (large)",
        "arch": "edsr",
        "scales": {
            2: ("EDSR_x2.pb", "https://github.com/Saafke/EDSR_Tensorflow/raw/master/models/EDSR_x2.pb"),
            3: ("EDSR_x3.pb", "https://github.com/Saafke/EDSR_Tensorflow/raw/master/models/EDSR_x3.pb"),
            4: ("EDSR_x4.pb", "https://github.com/Saafke/EDSR_Tensorflow/raw/master/models/EDSR_x4.pb"),
        },
    },
    "LapSRN": {
        "label": "LapSRN — detailed, up to ×8",
        "arch": "lapsrn",
        "scales": {
            2: ("LapSRN_x2.pb", "https://github.com/fannymonori/TF-LapSRN/raw/master/export/LapSRN_x2.pb"),
            4: ("LapSRN_x4.pb", "https://github.com/fannymonori/TF-LapSRN/raw/master/export/LapSRN_x4.pb"),
            8: ("LapSRN_x8.pb", "https://github.com/fannymonori/TF-LapSRN/raw/master/export/LapSRN_x8.pb"),
        },
    },
}

# Human label used in the UI for the plain (non-AI) resampling path.
LANCZOS_LABEL = "Lanczos — high quality (no download)"

# Real-ESRGAN runs through onnxruntime (already a dependency of rembg) and
# gives the most detailed, photo-realistic upscales. The .onnx model is
# fetched on first use; if the download is unavailable the code falls back
# to Lanczos so the app always produces a result. You can also drop your own
# ONNX file with this name into MODELS_DIR to use it offline.
REALESRGAN_MODELS = {
    "RealESRGAN-x4": {
        "label": "Real-ESRGAN ×4 — photo detail",
        "scale": 4,
        "file": "realesr-general-x4v3.onnx",
        # Small (~5 MB) general-purpose Real-ESRGAN model. If the download is
        # blocked you can drop this exact filename into MODELS_DIR to use it
        # offline; the app falls back to Lanczos when it is unavailable.
        "url": "https://huggingface.co/Samo629/real-esrgan-onnx/resolve/main/realesr-general-x4v3.onnx",
    },
}


def preferred_ort_providers(use_gpu=True):
    """Ordered onnxruntime execution providers, preferring hardware
    acceleration (CUDA / Apple CoreML / DirectML) when available."""
    try:
        import onnxruntime as ort
        avail = set(ort.get_available_providers())
    except Exception:
        return None
    if use_gpu:
        order = ["CUDAExecutionProvider", "CoreMLExecutionProvider",
                 "DmlExecutionProvider", "CPUExecutionProvider"]
    else:
        order = ["CPUExecutionProvider"]
    picked = [p for p in order if p in avail]
    return picked or None


def model_is_downloaded(model_name):
    return os.path.exists(os.path.join(MODELS_DIR, f"{model_name}.onnx"))


def ensure_rembg():
    """Attempt to import rembg lazily. Returns True on success."""
    global REMBG_AVAILABLE, remove_bg
    if REMBG_AVAILABLE:
        return True
    try:
        from rembg import remove as _remove
        remove_bg = _remove
        REMBG_AVAILABLE = True
    except Exception as e:
        REMBG_AVAILABLE = False
        print(f"Info: rembg unavailable ({e}). AI removal disabled.")
    return REMBG_AVAILABLE


# ======================================================================
# Helper functions
# ======================================================================

def pil_to_qpixmap(pil_image):
    if pil_image is None:
        return QPixmap()
    try:
        return QPixmap.fromImage(ImageQt.ImageQt(pil_image.convert("RGBA")))
    except Exception as e:
        print(f"Error converting PIL to QPixmap: {e}")
        return QPixmap()


def qimage_to_pil(qimage):
    if qimage.isNull():
        return None
    buffer = QBuffer()
    buffer.open(QBuffer.OpenModeFlag.ReadWrite)
    qimage.save(buffer, 'PNG')
    return Image.open(io.BytesIO(buffer.data())).convert('RGBA')


def flatten_image(pil_image, bg_color=(255, 255, 255)):
    """Composite an RGBA image over a solid background, returning RGB."""
    img = pil_image.convert('RGBA')
    background = Image.new('RGB', img.size, bg_color)
    background.paste(img, mask=img.getchannel('A'))
    return background


def create_checkerboard(width, height, grid_size=40):
    pixmap = QPixmap(width, height)
    pixmap.fill(Qt.GlobalColor.transparent)
    painter = QPainter(pixmap)
    color1, color2 = QColor(200, 200, 200), QColor(230, 230, 230)
    for y in range(0, height, grid_size):
        for x in range(0, width, grid_size):
            painter.setBrush(color1 if (x // grid_size + y // grid_size) % 2 == 0 else color2)
            painter.drawRect(x, y, grid_size, grid_size)
    painter.end()
    return pixmap


def create_brush_cursor(diameter, color):
    diameter = max(1, int(diameter))
    pix_size = max(32, diameter + 4)
    pixmap = QPixmap(pix_size, pix_size)
    pixmap.fill(Qt.GlobalColor.transparent)
    painter = QPainter(pixmap)
    painter.setRenderHint(QPainter.RenderHint.Antialiasing)
    center, radius = pix_size / 2.0, diameter / 2.0
    painter.setPen(QColor(0, 0, 0))
    painter.setBrush(Qt.BrushStyle.NoBrush)
    painter.drawEllipse(QPoint(int(center), int(center)), int(radius), int(radius))
    fill_color = QColor(color.red(), color.green(), color.blue(), 100)
    painter.setPen(Qt.PenStyle.NoPen)
    painter.setBrush(fill_color)
    painter.drawEllipse(QPoint(int(center), int(center)), int(radius), int(radius))
    painter.end()
    return QCursor(pixmap, int(center), int(center))


def create_splash_pixmap(width=560, height=340):
    """Build a branded splash-screen pixmap."""
    pixmap = QPixmap(width, height)
    painter = QPainter(pixmap)
    painter.setRenderHint(QPainter.RenderHint.Antialiasing)

    grad = QLinearGradient(0, 0, width, height)
    grad.setColorAt(0.0, QColor("#12213f"))
    grad.setColorAt(0.55, QColor("#1b3a6b"))
    grad.setColorAt(1.0, QColor(ACCENT))
    painter.fillRect(0, 0, width, height, QBrush(grad))

    # subtle accent panel
    painter.setPen(Qt.PenStyle.NoPen)
    painter.setBrush(QColor(255, 255, 255, 18))
    painter.drawRoundedRect(28, 28, width - 56, height - 56, 16, 16)

    # Logo mark
    painter.setBrush(QColor(255, 255, 255, 235))
    painter.drawRoundedRect(52, 60, 64, 64, 14, 14)
    painter.setPen(QPen(QColor(ACCENT), 0))
    painter.setBrush(QColor(ACCENT))
    painter.drawEllipse(QPoint(84, 92), 18, 18)

    # Title
    painter.setPen(QColor("#ffffff"))
    title_font = QFont(UI_FONT, 30, QFont.Weight.Bold)
    painter.setFont(title_font)
    painter.drawText(QRect(136, 56, width - 180, 44),
                     Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft, APP_NAME)

    sub_font = QFont(UI_FONT, 13)
    painter.setFont(sub_font)
    painter.setPen(QColor(220, 232, 255))
    painter.drawText(QRect(138, 100, width - 180, 28),
                     Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft,
                     f"{APP_TAGLINE}  ·  v{APP_VERSION}")

    # Feature line
    painter.setPen(QColor(200, 215, 245))
    painter.setFont(QFont(UI_FONT, 10))
    painter.drawText(QRect(54, 150, width - 108, 24),
                     Qt.AlignmentFlag.AlignLeft,
                     "AI removal · Perspective correction · Overlay · PDF export")

    # Developer credit (bottom of the card)
    painter.setPen(QColor(190, 208, 240))
    painter.setFont(QFont(UI_FONT, 10, QFont.Weight.DemiBold))
    painter.drawText(QRect(54, height - 70, width - 108, 22),
                     Qt.AlignmentFlag.AlignLeft,
                     f"Developed by {APP_AUTHOR}")

    painter.end()
    return pixmap


# ======================================================================
# Perspective correction (OpenCV)
# ======================================================================

def _tps_kernel(r2):
    """Thin-plate-spline radial basis U(r) = r^2 * log(r), given r^2."""
    out = np.zeros_like(r2)
    nz = r2 > 1e-12
    out[nz] = 0.5 * r2[nz] * np.log(r2[nz])
    return out


def _tps_fit(control, values):
    """Solve TPS weights mapping `control` points to scalar `values`."""
    n = control.shape[0]
    diff = control[:, None, :] - control[None, :, :]
    r2 = np.sum(diff ** 2, axis=2)
    K = _tps_kernel(r2)
    P = np.hstack([np.ones((n, 1)), control])
    L = np.zeros((n + 3, n + 3))
    L[:n, :n] = K
    L[:n, n:] = P
    L[n:, :n] = P.T
    Y = np.concatenate([values, np.zeros(3)])
    return np.linalg.solve(L, Y)


def _tps_eval(params, control, grid_xy):
    n = control.shape[0]
    w, a = params[:n], params[n:]
    diff = grid_xy[:, None, :] - control[None, :, :]
    r2 = np.sum(diff ** 2, axis=2)
    U = _tps_kernel(r2)
    return a[0] + a[1] * grid_xy[:, 0] + a[2] * grid_xy[:, 1] + U.dot(w)


def _tps_warp_image(image, src_pts, dst_pts):
    """
    Warp `image` so that content at `src_pts` moves to `dst_pts`.
    Pure-NumPy TPS + cv2.remap (no opencv-contrib required).
    """
    h, w = image.shape[:2]
    dst = dst_pts.astype(np.float64)
    src = src_pts.astype(np.float64)
    # Inverse map: for each destination pixel, where to sample from source.
    px = _tps_fit(dst, src[:, 0])
    py = _tps_fit(dst, src[:, 1])
    ys, xs = np.mgrid[0:h, 0:w]
    grid = np.stack([xs.ravel(), ys.ravel()], axis=1).astype(np.float64)
    map_x = _tps_eval(px, dst, grid).reshape(h, w).astype(np.float32)
    map_y = _tps_eval(py, dst, grid).reshape(h, w).astype(np.float32)
    return cv2.remap(image, map_x, map_y, interpolation=cv2.INTER_LANCZOS4,
                     borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0, 0))


def perspective_correct(pil_image, points):
    """
    Correct perspective of an image given ordered corner points.

    points order:
        4-point: [Top-Left, Top-Right, Bottom-Right, Bottom-Left]
        6-point: 4 corners as above, then [Top-Middle, Bottom-Middle]
                 which are used to straighten vertical curvature via TPS.
    Returns an RGBA PIL image.
    """
    if not CV2_AVAILABLE:
        raise RuntimeError("OpenCV (cv2) is required for perspective correction.")

    img = np.array(pil_image.convert('RGBA'))
    pts = np.array(points, dtype=np.float32)
    tl, tr, br, bl = pts[0], pts[1], pts[2], pts[3]

    width_top = np.linalg.norm(tr - tl)
    width_bottom = np.linalg.norm(br - bl)
    height_left = np.linalg.norm(bl - tl)
    height_right = np.linalg.norm(br - tr)
    W = max(int(round(max(width_top, width_bottom))), 1)
    H = max(int(round(max(height_left, height_right))), 1)

    src = np.array([tl, tr, br, bl], dtype=np.float32)
    dst = np.array([[0, 0], [W - 1, 0], [W - 1, H - 1], [0, H - 1]], dtype=np.float32)
    M = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(
        img, M, (W, H), flags=cv2.INTER_LANCZOS4,
        borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0, 0))

    if len(points) >= 6:
        # Map the mid-edge points through the same perspective transform,
        # then use a Thin-Plate-Spline to pin them to the ideal mid-edges.
        mids = np.array([points[4], points[5]], dtype=np.float32).reshape(-1, 1, 2)
        warped_mids = cv2.perspectiveTransform(mids, M).reshape(-1, 2)
        src6 = np.array([
            [0, 0], [W - 1, 0], [W - 1, H - 1], [0, H - 1],
            warped_mids[0], warped_mids[1]
        ], dtype=np.float32)
        dst6 = np.array([
            [0, 0], [W - 1, 0], [W - 1, H - 1], [0, H - 1],
            [(W - 1) / 2.0, 0], [(W - 1) / 2.0, H - 1]
        ], dtype=np.float32)
        try:
            warped = _tps_warp_image(warped, src6, dst6)
        except Exception as e:
            print(f"6-point TPS warp failed, using 4-point result: {e}")

    return Image.fromarray(warped, 'RGBA')


# ======================================================================
# AI enhancement — sharpen & upscale
# ======================================================================

def sharpen_image(pil_image, amount=150, radius=2.0, threshold=2, report=None):
    """Detail-enhance with an unsharp mask while preserving transparency."""
    if report:
        report("Sharpening image…")
    rgba = pil_image.convert("RGBA")
    r, g, b, a = rgba.split()
    rgb = Image.merge("RGB", (r, g, b))
    sharp = rgb.filter(ImageFilter.UnsharpMask(
        radius=float(radius), percent=int(amount), threshold=int(threshold)))
    sr, sg, sb = sharp.split()
    return Image.merge("RGBA", (sr, sg, sb, a))


def sr_available():
    """True if OpenCV's dnn_superres CNN upscaler is usable."""
    return CV2_AVAILABLE and hasattr(cv2, "dnn_superres")


def sr_model_scales(model):
    """Sorted list of scale factors a given SR model supports."""
    entry = SR_MODELS.get(model)
    return sorted(entry["scales"]) if entry else []


def sr_model_is_downloaded(model, scale):
    """True if the .pb file for (model, scale) already exists locally."""
    entry = SR_MODELS.get(model)
    if not entry or scale not in entry["scales"]:
        return False
    fname, _ = entry["scales"][scale]
    return os.path.exists(os.path.join(MODELS_DIR, fname))


def ensure_sr_model(model, scale, report=None):
    """Return a local path for the (model, scale) SR network, downloading it
    on first use. Returns None if unavailable or the download fails."""
    entry = SR_MODELS.get(model)
    if not entry or scale not in entry["scales"]:
        return None
    fname, url = entry["scales"][scale]
    path = os.path.join(MODELS_DIR, fname)
    if os.path.exists(path):
        return path
    try:
        import requests
        if report:
            report(f"Downloading {model} ×{scale} upscaler… (first use only)")
        resp = requests.get(url, timeout=180, stream=True)
        resp.raise_for_status()
        tmp = path + ".part"
        with open(tmp, "wb") as f:
            for chunk in resp.iter_content(chunk_size=65536):
                if chunk:
                    f.write(chunk)
        os.replace(tmp, path)
        return path
    except Exception as e:
        print(f"AI upscaler download failed ({e}); using high-quality fallback.")
        return None


def _lanczos_upscale(rgb, new_size, report, scale, sharpen=True):
    """High-quality Lanczos resampling with an optional light sharpen."""
    if report:
        report(f"Upscaling ×{scale} (Lanczos, high quality)…")
    out = rgb.resize(new_size, Image.LANCZOS)
    if sharpen:
        out = out.filter(ImageFilter.UnsharpMask(radius=2, percent=80, threshold=2))
    return out


def _denoise_rgb(rgb, report=None):
    """Light colour denoise before upscaling (OpenCV NLM). No-op without cv2."""
    if not CV2_AVAILABLE:
        return rgb
    try:
        if report:
            report("Denoising…")
        bgr = cv2.cvtColor(np.array(rgb), cv2.COLOR_RGB2BGR)
        out = cv2.fastNlMeansDenoisingColored(bgr, None, 3, 3, 7, 21)
        return Image.fromarray(cv2.cvtColor(out, cv2.COLOR_BGR2RGB))
    except Exception as e:
        print(f"Denoise skipped ({e}).")
        return rgb


def _dnn_sr_upsample(sr, bgr, scale, tile=0):
    """Run a cv2.dnn_superres model, optionally tiling large images to bound
    memory. Tiles use an overlap so seams don't show after stitching."""
    h, w = bgr.shape[:2]
    if tile <= 0 or (w <= tile and h <= tile):
        return sr.upsample(bgr)
    overlap = 16
    out = np.zeros((h * scale, w * scale, 3), dtype=bgr.dtype)
    for y in range(0, h, tile):
        for x in range(0, w, tile):
            x0, y0 = max(0, x - overlap), max(0, y - overlap)
            x1, y1 = min(w, x + tile + overlap), min(h, y + tile + overlap)
            patch = sr.upsample(bgr[y0:y1, x0:x1])
            # crop the overlap margins back off, in output coordinates
            cx0, cy0 = (x - x0) * scale, (y - y0) * scale
            ex, ey = min(x + tile, w), min(y + tile, h)
            cw, ch = (ex - x) * scale, (ey - y) * scale
            out[y * scale:y * scale + ch, x * scale:x * scale + cw] = \
                patch[cy0:cy0 + ch, cx0:cx0 + cw]
    return out


def ensure_realesrgan_model(key, report=None):
    """Local path to a Real-ESRGAN .onnx, downloading on first use."""
    entry = REALESRGAN_MODELS.get(key)
    if not entry:
        return None
    path = os.path.join(MODELS_DIR, entry["file"])
    if os.path.exists(path):
        return path
    try:
        import requests
        if report:
            report("Downloading Real-ESRGAN model… (first use, ~65 MB)")
        resp = requests.get(entry["url"], timeout=300, stream=True)
        resp.raise_for_status()
        tmp = path + ".part"
        with open(tmp, "wb") as f:
            for chunk in resp.iter_content(chunk_size=131072):
                if chunk:
                    f.write(chunk)
        os.replace(tmp, path)
        return path
    except Exception as e:
        print(f"Real-ESRGAN download failed ({e}); using high-quality fallback.")
        return None


def _realesrgan_upscale(rgb, key, report=None, use_gpu=True, tile=256):
    """Upscale an RGB PIL image ×4 with Real-ESRGAN via onnxruntime, tiling to
    keep memory bounded. Returns None on any failure (caller falls back)."""
    model_path = ensure_realesrgan_model(key, report)
    if not model_path:
        return None
    try:
        import onnxruntime as ort
        providers = preferred_ort_providers(use_gpu) or ["CPUExecutionProvider"]
        sess = ort.InferenceSession(model_path, providers=providers)
        inp_name = sess.get_inputs()[0].name
        scale = REALESRGAN_MODELS[key]["scale"]
        if report:
            report(f"Running Real-ESRGAN ×{scale}…")
        arr = np.asarray(rgb, dtype=np.float32) / 255.0        # H,W,3 RGB
        h, w, _ = arr.shape
        out = np.zeros((h * scale, w * scale, 3), dtype=np.float32)
        pad = 16
        step = max(32, tile)
        for y in range(0, h, step):
            for x in range(0, w, step):
                x0, y0 = max(0, x - pad), max(0, y - pad)
                x1, y1 = min(w, x + step + pad), min(h, y + step + pad)
                patch = arr[y0:y1, x0:x1, :]
                inp = np.transpose(patch, (2, 0, 1))[None, ...]   # 1,3,h,w
                res = sess.run(None, {inp_name: inp})[0]
                res = np.clip(res[0], 0.0, 1.0)
                res = np.transpose(res, (1, 2, 0))                # h,w,3
                cx0, cy0 = (x - x0) * scale, (y - y0) * scale
                ex, ey = min(x + step, w), min(y + step, h)
                cw, ch = (ex - x) * scale, (ey - y) * scale
                out[y * scale:y * scale + ch, x * scale:x * scale + cw] = \
                    res[cy0:cy0 + ch, cx0:cx0 + cw]
        return Image.fromarray((out * 255.0 + 0.5).astype(np.uint8))
    except Exception as e:
        print(f"Real-ESRGAN upscale failed ({e}); using high-quality fallback.")
        return None


# Above this many source pixels, CNN upscaling is tiled to bound memory.
SR_TILE_THRESHOLD = 1_000_000
SR_TILE_SIZE = 512


def upscale_image(pil_image, scale=2, model="EDSR", use_ai=True,
                  sharpen=False, denoise=False, use_gpu=True, report=None):
    """Upscale by an integer factor, preserving transparency.

    `model` selects the upscaler: a dnn_superres network (SR_MODELS), a
    Real-ESRGAN ONNX model (REALESRGAN_MODELS), or None/Lanczos. Large images
    are processed in tiles. `denoise` runs a light pre-pass; `sharpen` adds a
    post unsharp mask; `use_gpu` allows hardware acceleration for Real-ESRGAN.
    Any failure degrades gracefully to high-quality Lanczos so a result is
    always produced."""
    scale = int(scale)
    rgba = pil_image.convert("RGBA")
    r, g, b, a = rgba.split()
    rgb = Image.merge("RGB", (r, g, b))
    if denoise:
        rgb = _denoise_rgb(rgb, report)
    w, h = rgb.size
    new_size = (max(1, w * scale), max(1, h * scale))
    tile = SR_TILE_SIZE if (w * h) > SR_TILE_THRESHOLD else 0

    sr_rgb = None

    # --- Real-ESRGAN (onnxruntime) path ---
    if use_ai and model in REALESRGAN_MODELS:
        sr_rgb = _realesrgan_upscale(rgb, model, report, use_gpu=use_gpu,
                                     tile=(SR_TILE_SIZE if tile else 256))

    # --- OpenCV dnn_superres path ---
    entry = SR_MODELS.get(model)
    if sr_rgb is None and use_ai and sr_available() and entry and scale in entry["scales"]:
        model_path = ensure_sr_model(model, scale, report)
        if model_path:
            try:
                if report:
                    report(f"Running {model} super-resolution ×{scale}…")
                try:
                    sr = cv2.dnn_superres.DnnSuperResImpl_create()
                except AttributeError:
                    sr = cv2.dnn_superres.DnnSuperResImpl.create()
                sr.readModel(model_path)
                sr.setModel(entry["arch"], scale)
                bgr = cv2.cvtColor(np.array(rgb), cv2.COLOR_RGB2BGR)
                out = _dnn_sr_upsample(sr, bgr, scale, tile=tile)
                sr_rgb = Image.fromarray(cv2.cvtColor(out, cv2.COLOR_BGR2RGB))
            except Exception as e:
                print(f"AI upscale failed ({e}); using high-quality fallback.")
                sr_rgb = None

    if sr_rgb is None:
        sr_rgb = _lanczos_upscale(rgb, new_size, report, scale)

    if sr_rgb.size != new_size:
        sr_rgb = sr_rgb.resize(new_size, Image.LANCZOS)
    if sharpen:
        if report:
            report("Sharpening result…")
        sr_rgb = sr_rgb.filter(ImageFilter.UnsharpMask(radius=1.6, percent=90, threshold=2))
    up_a = a.resize(new_size, Image.LANCZOS)
    return Image.merge("RGBA", (*sr_rgb.split(), up_a))


# ======================================================================
# Vector (SVG) export
# ======================================================================

def _svg_header(w, h):
    return (f'<?xml version="1.0" encoding="UTF-8" standalone="no"?>\n'
            f'<svg xmlns="http://www.w3.org/2000/svg" '
            f'xmlns:xlink="http://www.w3.org/1999/xlink" '
            f'width="{w}" height="{h}" viewBox="0 0 {w} {h}">\n')


def _png_data_uri(rgba):
    buf = io.BytesIO()
    rgba.save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("ascii")


def _alpha_contours(rgba, threshold, simplify):
    """Vectorise the opaque silhouette into a list of point-lists."""
    alpha = np.array(rgba.split()[-1])
    mask = (alpha >= int(threshold)).astype(np.uint8) * 255
    contours, _ = cv2.findContours(mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    paths = []
    for cnt in contours:
        if len(cnt) < 3:
            continue
        approx = cv2.approxPolyDP(cnt, float(simplify), True)
        pts = approx.reshape(-1, 2)
        if len(pts) >= 3:
            paths.append(pts)
    return paths


def export_to_svg(path, pil_image, opts):
    """Write `pil_image` to an SVG file.

    opts:
      mode           -- "embed" (raster wrapped in SVG) or "outline"
      fill_color     -- (r,g,b) solid fill for outline mode, or None to clip
                        the embedded raster to the traced silhouette
      alpha_threshold-- opacity cut-off for tracing (0-255)
      simplify       -- contour simplification tolerance in pixels
    """
    rgba = pil_image.convert("RGBA")
    w, h = rgba.size
    mode = opts.get("mode", "embed")

    if mode == "outline" and CV2_AVAILABLE:
        paths = _alpha_contours(rgba, opts.get("alpha_threshold", 128),
                                opts.get("simplify", 1.5))
        d = " ".join(
            "M " + " L ".join(f"{int(x)} {int(y)}" for x, y in pts) + " Z"
            for pts in paths)
        fill = opts.get("fill_color")
        parts = [_svg_header(w, h)]
        if not d:
            # Nothing opaque to trace — fall back to embedding.
            parts.append(f'  <image width="{w}" height="{h}" x="0" y="0" '
                         f'xlink:href="{_png_data_uri(rgba)}"/>\n')
        elif fill is None:
            parts.append(f'  <defs><clipPath id="cut" clipPathUnits="userSpaceOnUse">'
                         f'<path d="{d}" clip-rule="evenodd"/></clipPath></defs>\n')
            parts.append(f'  <image width="{w}" height="{h}" x="0" y="0" '
                         f'clip-path="url(#cut)" xlink:href="{_png_data_uri(rgba)}"/>\n')
        else:
            col = "#%02x%02x%02x" % tuple(fill)
            parts.append(f'  <path d="{d}" fill="{col}" fill-rule="evenodd"/>\n')
        parts.append("</svg>\n")
        svg = "".join(parts)
    else:
        svg = (_svg_header(w, h) +
               f'  <image width="{w}" height="{h}" x="0" y="0" '
               f'xlink:href="{_png_data_uri(rgba)}"/>\n</svg>\n')

    with open(path, "w", encoding="utf-8") as f:
        f.write(svg)


# ======================================================================
# Interactive image label
# ======================================================================

class InteractiveLabel(QLabel):
    MODE_NONE, MODE_KEEP, MODE_REMOVE, MODE_CROP, MODE_WAND, MODE_PERSPECTIVE = range(6)

    interaction_started = Signal()
    stroke_committed = Signal(QPixmap)
    wand_point_selected = Signal(QPoint)
    perspective_changed = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.current_mode = self.MODE_NONE
        self.drawing, self.cropping = False, False
        self.last_point = QPoint()
        self.crop_start_point, self.crop_end_point = QPoint(), QPoint()
        self.crop_rect_visual = None
        self.base_pixmap, self.overlay_pixmap = QPixmap(), QPixmap()
        self.wand_preview_pixmap = QPixmap()
        self.wand_selection_mask = None

        # Perspective correction state
        self.perspective_points = []          # list[QPoint] in image coords
        self.perspective_required = 4
        self._perspective_drag_index = None

        # Alignment grid overlay (for rotation / straightening)
        self.show_grid = False
        self.grid_spacing = 50          # in image pixels

        self.zoom_level = 1.0
        self.brush_size = 10
        self.scroll_area = None

        self.setMouseTracking(True)
        self.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Ignored)

    def set_scroll_area(self, scroll_area):
        self.scroll_area = scroll_area

    def set_display_pixmap(self, pixmap):
        self.base_pixmap = pixmap if pixmap else QPixmap()
        if not self.base_pixmap.isNull():
            if self.overlay_pixmap.isNull() or self.overlay_pixmap.size() != self.base_pixmap.size():
                self.overlay_pixmap = QPixmap(self.base_pixmap.size())
                self.overlay_pixmap.fill(Qt.GlobalColor.transparent)
                self.wand_preview_pixmap = QPixmap(self.base_pixmap.size())
                self.wand_preview_pixmap.fill(Qt.GlobalColor.transparent)
            self.set_zoom(self.zoom_level)
        else:
            self.overlay_pixmap = QPixmap()
            self.wand_preview_pixmap = QPixmap()
        self.update()

    def clear_overlay(self):
        if not self.overlay_pixmap.isNull():
            self.overlay_pixmap.fill(Qt.GlobalColor.transparent)
            self.update()

    def get_overlay_pixmap(self):
        return self.overlay_pixmap.copy()

    def set_zoom(self, level):
        self.zoom_level = max(0.05, level)
        if not self.base_pixmap.isNull():
            self.resize(self.base_pixmap.size() * self.zoom_level)
        self.update_cursor()
        self.update()

    def fit_to_view(self):
        if self.base_pixmap.isNull() or not self.scroll_area:
            return
        vp_size = self.scroll_area.viewport().size() - QSize(2, 2)
        img_size = self.base_pixmap.size()
        if img_size.width() == 0 or img_size.height() == 0:
            return
        w_ratio = vp_size.width() / img_size.width()
        h_ratio = vp_size.height() / img_size.height()
        self.set_zoom(min(w_ratio, h_ratio))

    def set_mode(self, mode):
        if self.current_mode == self.MODE_WAND and mode != self.MODE_WAND:
            self.clear_wand_selection()
        if self.current_mode == self.MODE_PERSPECTIVE and mode != self.MODE_PERSPECTIVE:
            pass  # keep points so user can still Apply

        self.current_mode = mode if self.current_mode != mode else self.MODE_NONE
        if self.current_mode != self.MODE_CROP:
            self.crop_rect_visual = None
        self.update_cursor()
        self.update()

    def clear_interaction_state(self):
        self.crop_rect_visual, self.drawing, self.cropping = None, False, False
        self.clear_overlay()
        self.clear_wand_selection()
        self.clear_perspective()

    def clear_wand_selection(self):
        if not self.wand_preview_pixmap.isNull():
            self.wand_preview_pixmap.fill(Qt.GlobalColor.transparent)
        self.wand_selection_mask = None
        self.update()

    def set_wand_preview(self, mask_pil):
        self.clear_wand_selection()
        if mask_pil is None:
            return
        self.wand_selection_mask = mask_pil
        preview_color = QColor(0, 150, 255, 100)
        mask_np = np.array(mask_pil)
        color_img_np = np.zeros((mask_np.shape[0], mask_np.shape[1], 4), dtype=np.uint8)
        color_img_np[:, :, 0] = preview_color.red()
        color_img_np[:, :, 1] = preview_color.green()
        color_img_np[:, :, 2] = preview_color.blue()
        color_img_np[:, :, 3] = (mask_np / 255.0 * preview_color.alpha()).astype(np.uint8)
        preview_pil = Image.fromarray(color_img_np, 'RGBA')
        self.wand_preview_pixmap = pil_to_qpixmap(preview_pil)
        self.update()

    # --- Perspective helpers ---
    def set_perspective_required(self, n):
        self.perspective_required = n
        self.clear_perspective()

    def clear_perspective(self):
        self.perspective_points = []
        self._perspective_drag_index = None
        self.update()
        self.perspective_changed.emit()

    def get_perspective_points(self):
        return [(p.x(), p.y()) for p in self.perspective_points]

    def _nearest_perspective_index(self, img_point, radius_img):
        for i, p in enumerate(self.perspective_points):
            if (p - img_point).manhattanLength() <= radius_img * 2 and \
               (QPointF(p - img_point)).manhattanLength() >= 0:
                dx, dy = p.x() - img_point.x(), p.y() - img_point.y()
                if dx * dx + dy * dy <= radius_img * radius_img:
                    return i
        return None

    def get_crop_rect(self):
        if not self.crop_rect_visual or self.base_pixmap.isNull():
            return None
        img_x = self.crop_rect_visual.x() / self.zoom_level
        img_y = self.crop_rect_visual.y() / self.zoom_level
        img_w = self.crop_rect_visual.width() / self.zoom_level
        img_h = self.crop_rect_visual.height() / self.zoom_level
        return QRect(int(img_x), int(img_y), int(img_w), int(img_h)).normalized()

    def map_to_image(self, view_point):
        if self.base_pixmap.isNull() or self.zoom_level == 0:
            return QPoint(0, 0)
        return view_point / self.zoom_level

    def set_brush_size(self, size):
        self.brush_size = max(1, size)
        self.update_cursor()

    def set_grid(self, enabled, spacing=None):
        self.show_grid = bool(enabled)
        if spacing is not None:
            self.grid_spacing = max(5, int(spacing))
        self.update()

    def _draw_grid(self, painter):
        if self.base_pixmap.isNull():
            return
        w, h = self.width(), self.height()
        step = max(4.0, self.grid_spacing * self.zoom_level)
        painter.setPen(QPen(QColor(0, 0, 0, 55), 1, Qt.PenStyle.SolidLine))
        x = step
        while x < w:
            painter.drawLine(int(x), 0, int(x), h)
            x += step
        y = step
        while y < h:
            painter.drawLine(0, int(y), w, int(y))
            y += step
        # Emphasise centre cross to help align horizontals / verticals
        painter.setPen(QPen(QColor(ACCENT), 1, Qt.PenStyle.DashLine))
        painter.drawLine(w // 2, 0, w // 2, h)
        painter.drawLine(0, h // 2, w, h // 2)

    def update_cursor(self):
        if not self.isEnabled():
            return self.setCursor(Qt.CursorShape.ArrowCursor)
        cursor_size = self.brush_size * self.zoom_level
        if self.current_mode == self.MODE_KEEP:
            self.setCursor(create_brush_cursor(cursor_size, QColor(0, 255, 0)))
        elif self.current_mode == self.MODE_REMOVE:
            self.setCursor(create_brush_cursor(cursor_size, QColor(255, 0, 0)))
        elif self.current_mode in (self.MODE_CROP, self.MODE_WAND, self.MODE_PERSPECTIVE):
            self.setCursor(Qt.CursorShape.CrossCursor)
        else:
            self.setCursor(Qt.CursorShape.ArrowCursor)

    def mousePressEvent(self, event):
        if not self.isEnabled() or self.base_pixmap.isNull() or event.button() != Qt.MouseButton.LeftButton:
            return super().mousePressEvent(event)

        self.interaction_started.emit()
        if self.current_mode in (self.MODE_KEEP, self.MODE_REMOVE):
            self.drawing = True
            self.last_point = event.pos()
            self._draw_on_overlay(self.last_point, self.last_point)
        elif self.current_mode == self.MODE_CROP:
            self.cropping = True
            self.crop_start_point = self.crop_end_point = event.pos()
            self.update()
        elif self.current_mode == self.MODE_WAND:
            self.wand_point_selected.emit(self.map_to_image(event.pos()))
        elif self.current_mode == self.MODE_PERSPECTIVE:
            img_pt = self.map_to_image(event.pos())
            radius_img = max(10, 14 / max(self.zoom_level, 0.05))
            idx = self._nearest_perspective_index(img_pt, radius_img)
            if idx is not None:
                self._perspective_drag_index = idx
            elif len(self.perspective_points) < self.perspective_required:
                self.perspective_points.append(img_pt)
                self.perspective_changed.emit()
            self.update()

    def mouseMoveEvent(self, event):
        if not self.isEnabled() or not (event.buttons() & Qt.MouseButton.LeftButton):
            return
        if self.drawing:
            self._draw_on_overlay(self.last_point, event.pos())
            self.last_point = event.pos()
        elif self.cropping:
            self.crop_end_point = event.pos()
            self.update()
        elif self.current_mode == self.MODE_PERSPECTIVE and self._perspective_drag_index is not None:
            self.perspective_points[self._perspective_drag_index] = self.map_to_image(event.pos())
            self.perspective_changed.emit()
            self.update()

    def mouseReleaseEvent(self, event):
        if not self.isEnabled() or event.button() != Qt.MouseButton.LeftButton:
            return
        if self.drawing:
            self.drawing = False
            self.stroke_committed.emit(self.overlay_pixmap)
        elif self.cropping:
            self.cropping = False
            self.crop_rect_visual = QRect(self.crop_start_point, self.crop_end_point).normalized()
            self.update()
        elif self.current_mode == self.MODE_PERSPECTIVE:
            self._perspective_drag_index = None

    def _draw_on_overlay(self, start_point, end_point):
        painter = QPainter(self.overlay_pixmap)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        start_img_point = self.map_to_image(start_point)
        end_img_point = self.map_to_image(end_point)
        color = QColor(0, 255, 0, 180) if self.current_mode == self.MODE_KEEP else QColor(255, 0, 0, 180)
        pen = QPen(color, self.brush_size, Qt.PenStyle.SolidLine,
                   Qt.PenCapStyle.RoundCap, Qt.PenJoinStyle.RoundJoin)
        painter.setPen(pen)
        if start_point == end_point:
            painter.drawPoint(start_img_point)
        else:
            painter.drawLine(start_img_point, end_img_point)
        painter.end()
        self.update()

    def _draw_perspective(self, painter):
        if not self.perspective_points:
            return
        z = self.zoom_level
        view_pts = [QPoint(int(p.x() * z), int(p.y() * z)) for p in self.perspective_points]

        # Draw quad connecting first 4 corners
        if len(view_pts) >= 2:
            painter.setPen(QPen(QColor(ACCENT), 2, Qt.PenStyle.DashLine))
            painter.setBrush(Qt.BrushStyle.NoBrush)
            corner_pts = view_pts[:min(4, len(view_pts))]
            if len(corner_pts) >= 4:
                painter.drawPolygon(QPolygon(corner_pts))
            else:
                for i in range(len(corner_pts) - 1):
                    painter.drawLine(corner_pts[i], corner_pts[i + 1])

        labels = ["1", "2", "3", "4", "5", "6"]
        for i, vp in enumerate(view_pts):
            is_mid = i >= 4
            fill = QColor("#ff8c00") if is_mid else QColor(ACCENT)
            painter.setPen(QPen(QColor(255, 255, 255), 2))
            painter.setBrush(fill)
            painter.drawEllipse(vp, 9, 9)
            painter.setPen(QColor(255, 255, 255))
            painter.setFont(QFont(UI_FONT, 8, QFont.Weight.Bold))
            painter.drawText(QRect(vp.x() - 9, vp.y() - 9, 18, 18),
                             Qt.AlignmentFlag.AlignCenter, labels[i])

    def paintEvent(self, event):
        painter = QPainter(self)
        if self.base_pixmap.isNull():
            painter.setPen(QColor(150, 150, 150))
            painter.setFont(QFont(UI_FONT, 13))
            painter.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter,
                             "Load, Paste, or Drag & Drop an Image")
            return

        painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)
        target_rect = self.rect()
        painter.drawPixmap(target_rect, self.base_pixmap, self.base_pixmap.rect())
        if not self.overlay_pixmap.isNull():
            painter.drawPixmap(target_rect, self.overlay_pixmap, self.overlay_pixmap.rect())
        if not self.wand_preview_pixmap.isNull():
            painter.drawPixmap(target_rect, self.wand_preview_pixmap, self.wand_preview_pixmap.rect())
        if self.cropping or self.crop_rect_visual:
            pen = QPen(QColor(0, 100, 255, 220), 2, Qt.PenStyle.DashLine)
            painter.setPen(pen)
            painter.setBrush(Qt.BrushStyle.NoBrush)
            rect = self.crop_rect_visual if not self.cropping else QRect(
                self.crop_start_point, self.crop_end_point).normalized()
            painter.drawRect(rect)
        if self.show_grid:
            self._draw_grid(painter)
        if self.current_mode == self.MODE_PERSPECTIVE or self.perspective_points:
            self._draw_perspective(painter)


# ======================================================================
# Overlay / compositing dialog
# ======================================================================

class CompositorCanvas(QWidget):
    """Interactive canvas: click to select a layer, drag to move / overlap,
    drag corner handles to resize. Reports changes back to the dialog."""

    selection_changed = Signal(int)   # -1 = none
    layer_changed = Signal()          # geometry of current layer changed

    HANDLE = 8                        # half-size of a resize handle (view px)

    def __init__(self, dialog):
        super().__init__(dialog)
        self.dialog = dialog
        self.base_size = QSize(1, 1)
        self.selected_index = None
        self._mode = None             # 'move' | 'resize'
        self._drag_off = (0, 0)
        self._resize_start = None
        self.setMinimumSize(460, 420)
        self.setMouseTracking(True)
        self.setStyleSheet("background:#2b2b2b; border:1px solid #444; border-radius:8px;")

    # --- geometry mapping (image <-> view) ---
    def _geometry(self):
        bw, bh = self.base_size.width(), self.base_size.height()
        W, H = self.width(), self.height()
        pad = 12
        disp = min((W - 2 * pad) / max(1, bw), (H - 2 * pad) / max(1, bh))
        disp = max(0.01, disp)
        dw, dh = bw * disp, bh * disp
        return disp, (W - dw) / 2.0, (H - dh) / 2.0

    def _to_image(self, pt):
        disp, ox, oy = self._geometry()
        return ((pt.x() - ox) / disp, (pt.y() - oy) / disp)

    def _layer_view_rect(self, layer):
        disp, ox, oy = self._geometry()
        x, y, ow, oh = self.dialog.layer_bbox(layer)
        return QRect(int(ox + x * disp), int(oy + y * disp),
                     max(1, int(ow * disp)), max(1, int(oh * disp)))

    def _handle_rects(self, rect):
        h = self.HANDLE
        return {
            "tl": QRect(rect.left() - h, rect.top() - h, 2 * h, 2 * h),
            "tr": QRect(rect.right() - h, rect.top() - h, 2 * h, 2 * h),
            "bl": QRect(rect.left() - h, rect.bottom() - h, 2 * h, 2 * h),
            "br": QRect(rect.right() - h, rect.bottom() - h, 2 * h, 2 * h),
        }

    def _hit_handle(self, pos):
        if self.selected_index is None:
            return None
        layer = self.dialog.layers[self.selected_index]
        rect = self._layer_view_rect(layer)
        for name, hr in self._handle_rects(rect).items():
            if hr.contains(pos):
                return name
        return None

    def _hit_layer(self, px, py):
        for i in range(len(self.dialog.layers) - 1, -1, -1):
            x, y, ow, oh = self.dialog.layer_bbox(self.dialog.layers[i])
            if x <= px <= x + ow and y <= py <= y + oh:
                return i
        return None

    def set_selected(self, idx):
        self.selected_index = idx if (idx is not None and idx >= 0) else None
        self.update()

    # --- mouse ---
    def mousePressEvent(self, event):
        if event.button() != Qt.MouseButton.LeftButton:
            return
        pos = event.position().toPoint()
        handle = self._hit_handle(pos)
        if handle is not None:
            layer = self.dialog.layers[self.selected_index]
            x, y, ow, oh = self.dialog.layer_bbox(layer)
            self._mode = "resize"
            # anchor = opposite corner (image coords)
            anchor = {
                "br": (x, y), "tr": (x, y + oh),
                "bl": (x + ow, y), "tl": (x + ow, y + oh),
            }[handle]
            self._resize_start = {"handle": handle, "anchor": anchor,
                                  "scale0": layer["scale"], "w0": ow, "h0": oh}
            return
        px, py = self._to_image(event.position())
        idx = self._hit_layer(px, py)
        self.set_selected(idx)
        self.selection_changed.emit(-1 if idx is None else idx)
        if idx is not None:
            layer = self.dialog.layers[idx]
            self._mode = "move"
            self._drag_off = (px - layer["x"], py - layer["y"])
        self.update()

    def mouseMoveEvent(self, event):
        pos = event.position().toPoint()
        if self._mode is None:
            # hover cursor feedback
            if self._hit_handle(pos) in ("tl", "br"):
                self.setCursor(Qt.CursorShape.SizeFDiagCursor)
            elif self._hit_handle(pos) in ("tr", "bl"):
                self.setCursor(Qt.CursorShape.SizeBDiagCursor)
            else:
                px, py = self._to_image(event.position())
                self.setCursor(Qt.CursorShape.SizeAllCursor
                               if self._hit_layer(px, py) is not None
                               else Qt.CursorShape.ArrowCursor)
            return

        if self.selected_index is None:
            return
        layer = self.dialog.layers[self.selected_index]
        px, py = self._to_image(event.position())

        if self._mode == "move":
            layer["x"] = int(round(px - self._drag_off[0]))
            layer["y"] = int(round(py - self._drag_off[1]))
        elif self._mode == "resize":
            rs = self._resize_start
            ax, ay = rs["anchor"]
            new_w = abs(px - ax)
            new_h = abs(py - ay)
            # uniform scale from the larger relative change
            ratio = max(new_w / max(1, rs["w0"]), new_h / max(1, rs["h0"]))
            new_scale = max(1, int(round(rs["scale0"] * ratio)))
            layer["scale"] = min(1000, new_scale)
            # keep the anchor corner fixed
            _, _, ow, oh = self.dialog.layer_bbox(layer)
            h = rs["handle"]
            layer["x"] = int(round(ax if "l" in h else ax - ow))
            layer["y"] = int(round(ay if "t" in h else ay - oh))
        self.layer_changed.emit()
        self.update()

    def mouseReleaseEvent(self, event):
        self._mode = None
        self._resize_start = None

    # --- paint ---
    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)
        disp, ox, oy = self._geometry()
        bw, bh = self.base_size.width(), self.base_size.height()
        dw, dh = int(bw * disp), int(bh * disp)

        # checkerboard behind, then composed image
        painter.drawPixmap(int(ox), int(oy), create_checkerboard(dw, dh))
        try:
            composed = self.dialog._compose(scale=disp)
            painter.drawPixmap(int(ox), int(oy), pil_to_qpixmap(composed))
        except Exception as e:
            print(f"Canvas compose error: {e}")

        # page border
        painter.setPen(QPen(QColor(120, 120, 120), 1))
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.drawRect(int(ox), int(oy), dw, dh)

        # selection + handles
        if self.selected_index is not None and 0 <= self.selected_index < len(self.dialog.layers):
            rect = self._layer_view_rect(self.dialog.layers[self.selected_index])
            painter.setPen(QPen(QColor(ACCENT), 2, Qt.PenStyle.DashLine))
            painter.drawRect(rect)
            painter.setPen(QPen(QColor(255, 255, 255), 1))
            painter.setBrush(QColor(ACCENT))
            for hr in self._handle_rects(rect).values():
                painter.drawRect(hr)
        painter.end()


class OverlayDialog(QDialog):
    """Interactive compositor: add multiple images, select them individually,
    drag to move / overlap, resize with handles, arrange on the current image
    or a blank page, then merge them together."""

    BLEND_MODES = ["Normal", "Multiply", "Screen", "Overlay"]

    def __init__(self, base_pil, parent=None, preload=None):
        super().__init__(parent)
        self.setWindowTitle("Compose / Overlay Images")
        self.setMinimumSize(960, 700)
        self.source_image = base_pil.convert('RGBA')
        self.base_pil = self.source_image
        self.layers = []
        self.result_pil = None
        self._loading = False
        self._source_added = False
        self._base_cache = {}

        main = QHBoxLayout(self)

        # ---- interactive canvas ----
        self.canvas = CompositorCanvas(self)
        self.canvas.base_size = QSize(*self.base_pil.size)
        main.addWidget(self.canvas, 1)

        # ---- controls ----
        side = QVBoxLayout()
        side.setSpacing(8)

        banner = QLabel("① Add an image   ② Drag / resize on canvas   "
                        "③ Adjust Overlap & Blend   ④ Merge All")
        banner.setWordWrap(True)
        banner.setStyleSheet(
            f"background:{ACCENT}; color:white; padding:8px; border-radius:6px; font-weight:600;")
        side.addWidget(banner)

        canvas_group = QGroupBox("Canvas / Page")
        cg = QFormLayout(canvas_group)
        self.combo_canvas = QComboBox()
        self.combo_canvas.addItems(["Current Image"] + list(PAGE_SIZES_MM.keys()))
        cg.addRow("Background:", self.combo_canvas)
        side.addWidget(canvas_group)

        layer_group = QGroupBox("Images (layers)")
        lg = QVBoxLayout(layer_group)
        self.layer_list = QListWidget()
        self.layer_list.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.layer_list.setMaximumHeight(120)
        lg.addWidget(self.layer_list)
        row = QHBoxLayout()
        self.btn_add = QPushButton("Add Image…")
        self.btn_add.setObjectName("primaryButton")
        self.btn_remove = QPushButton("Remove")
        self.btn_up = QPushButton("↑"); self.btn_up.setFixedWidth(34)
        self.btn_down = QPushButton("↓"); self.btn_down.setFixedWidth(34)
        for b in (self.btn_add, self.btn_remove, self.btn_up, self.btn_down):
            row.addWidget(b)
        lg.addLayout(row)
        side.addWidget(layer_group)

        self.props_group = QGroupBox("Selected Image")
        form = QFormLayout(self.props_group)
        form.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow)
        form.setLabelAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        form.setRowWrapPolicy(QFormLayout.RowWrapPolicy.DontWrapRows)
        form.setHorizontalSpacing(10)
        form.setVerticalSpacing(8)
        self.spin_x = QSpinBox(); self.spin_x.setRange(-20000, 20000)
        self.spin_y = QSpinBox(); self.spin_y.setRange(-20000, 20000)
        self.spin_scale = QSpinBox(); self.spin_scale.setRange(1, 1000); self.spin_scale.setSuffix(" %")
        self.spin_rotate = QSpinBox(); self.spin_rotate.setRange(-180, 180); self.spin_rotate.setSuffix(" °")
        self.slider_opacity = QSlider(Qt.Orientation.Horizontal); self.slider_opacity.setRange(0, 100)
        self.lbl_opacity = QLabel("100%")
        op_row = QHBoxLayout(); op_row.addWidget(self.slider_opacity); op_row.addWidget(self.lbl_opacity)
        self.combo_blend = QComboBox(); self.combo_blend.addItems(self.BLEND_MODES)
        form.addRow("X:", self.spin_x)
        form.addRow("Y:", self.spin_y)
        form.addRow("Size:", self.spin_scale)
        form.addRow("Rotation:", self.spin_rotate)
        form.addRow("Overlap:", op_row)
        form.addRow("Blend:", self.combo_blend)
        self.btn_center = QPushButton("Center on Canvas")
        form.addRow(self.btn_center)
        side.addWidget(self.props_group)

        hint = QLabel("Drag an image on the canvas to move it; drag its corner "
                      "handles to resize. Use the list to pick which image to edit.")
        hint.setWordWrap(True)
        hint.setStyleSheet("color:#667; font-size:11px;")
        side.addWidget(hint)
        side.addStretch()

        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        self.btn_ok = buttons.button(QDialogButtonBox.StandardButton.Ok)
        self.btn_ok.setText("Merge All"); self.btn_ok.setEnabled(False)
        side.addWidget(buttons)

        # Put the controls in a scroll area so the panels keep their natural
        # heights and never compress into each other on short windows.
        sw = QWidget(); sw.setLayout(side)
        side.setContentsMargins(0, 0, 6, 0)
        side_scroll = QScrollArea()
        side_scroll.setWidget(sw)
        side_scroll.setWidgetResizable(True)
        side_scroll.setFrameShape(QFrame.Shape.NoFrame)
        side_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        side_scroll.setFixedWidth(322)
        main.addWidget(side_scroll)

        # signals
        self.combo_canvas.currentTextChanged.connect(self._on_canvas_changed)
        self.btn_add.clicked.connect(self.add_layer)
        self.btn_remove.clicked.connect(self.remove_layer)
        self.btn_up.clicked.connect(lambda: self.move_layer(-1))
        self.btn_down.clicked.connect(lambda: self.move_layer(1))
        self.btn_center.clicked.connect(self.center_current)
        self.layer_list.currentRowChanged.connect(self._on_list_row)
        self.canvas.selection_changed.connect(self._on_canvas_select)
        self.canvas.layer_changed.connect(self._on_canvas_geometry)
        for w in (self.spin_x, self.spin_y, self.spin_scale, self.spin_rotate):
            w.valueChanged.connect(self.on_prop_changed)
        self.slider_opacity.valueChanged.connect(self.on_prop_changed)
        self.combo_blend.currentIndexChanged.connect(self.on_prop_changed)
        buttons.accepted.connect(self.accept_merge)
        buttons.rejected.connect(self.reject)

        self.props_group.setEnabled(False)

        # Preload extra images (e.g. from a multi-file drag-and-drop) as layers.
        for item in (preload or []):
            try:
                img = item if isinstance(item, Image.Image) else Image.open(item)
                name = "layer" if isinstance(item, Image.Image) else os.path.basename(item)
                self._add_image_layer(img.convert('RGBA'), name)
            except Exception as e:
                print(f"Could not preload overlay image ({e}).")

    # ---- canvas background ----
    def _on_canvas_changed(self, text):
        if text == "Current Image":
            self.base_pil = self.source_image
        else:
            pw_mm, ph_mm = PAGE_SIZES_MM[text]
            dpi = 150
            pw = max(1, int(round(pw_mm / 25.4 * dpi)))
            ph = max(1, int(round(ph_mm / 25.4 * dpi)))
            self.base_pil = Image.new('RGBA', (pw, ph), (255, 255, 255, 255))
            if not self._source_added:
                self._add_image_layer(self.source_image.copy(), "base image")
                self._source_added = True
        self._base_cache = {}
        self.canvas.base_size = QSize(*self.base_pil.size)
        self.canvas.update()

    # ---- layer management ----
    def _current(self):
        i = self.layer_list.currentRow()
        return self.layers[i] if 0 <= i < len(self.layers) else None

    def _add_image_layer(self, img, name):
        scale = 1.0
        if img.width > self.base_pil.width or img.height > self.base_pil.height:
            scale = min(self.base_pil.width / img.width, self.base_pil.height / img.height)
        layer = {"image": img, "name": name, "x": 0, "y": 0,
                 "scale": max(1, int(scale * 100)), "rotate": 0, "opacity": 100, "blend": "Normal"}
        self.layers.append(layer)
        self.layer_list.addItem(QListWidgetItem(name))
        self.layer_list.setCurrentRow(len(self.layers) - 1)
        self.center_current()
        self.btn_ok.setEnabled(True)

    def add_layer(self):
        path, _ = QFileDialog.getOpenFileName(self, "Add Image", "",
            "Images (*.png *.jpg *.jpeg *.bmp *.webp *.tiff *.gif)")
        if not path:
            return
        try:
            img = Image.open(path).convert('RGBA')
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to load image: {e}")
            return
        self._add_image_layer(img, os.path.basename(path))

    def remove_layer(self):
        i = self.layer_list.currentRow()
        if 0 <= i < len(self.layers):
            self.layers.pop(i)
            self.layer_list.takeItem(i)
            self.btn_ok.setEnabled(bool(self.layers))
            self.canvas.set_selected(self.layer_list.currentRow() if self.layers else None)
            self.canvas.update()

    def move_layer(self, delta):
        i = self.layer_list.currentRow()
        j = i + delta
        if 0 <= i < len(self.layers) and 0 <= j < len(self.layers):
            self.layers[i], self.layers[j] = self.layers[j], self.layers[i]
            self.layer_list.item(i).setText(self.layers[i]["name"])
            self.layer_list.item(j).setText(self.layers[j]["name"])
            self.layer_list.setCurrentRow(j)
            self.canvas.set_selected(j)
            self.canvas.update()

    # ---- selection sync ----
    def _select(self, idx):
        self.layer_list.blockSignals(True)
        self.layer_list.setCurrentRow(idx if idx is not None else -1)
        self.layer_list.blockSignals(False)
        self.canvas.set_selected(idx)
        self._load_props()

    def _on_list_row(self, row):
        idx = row if 0 <= row < len(self.layers) else None
        self.canvas.set_selected(idx)
        self._load_props()

    def _on_canvas_select(self, idx):
        idx = None if idx < 0 else idx
        self.layer_list.blockSignals(True)
        self.layer_list.setCurrentRow(idx if idx is not None else -1)
        self.layer_list.blockSignals(False)
        self._load_props()

    def _on_canvas_geometry(self):
        # layer moved/resized on canvas -> refresh spinboxes
        self._load_props()

    def _load_props(self):
        layer = self._current()
        self.props_group.setEnabled(layer is not None)
        if layer is None:
            return
        self._loading = True
        self.spin_x.setValue(layer["x"]); self.spin_y.setValue(layer["y"])
        self.spin_scale.setValue(layer["scale"]); self.spin_rotate.setValue(layer["rotate"])
        self.slider_opacity.setValue(layer["opacity"]); self.lbl_opacity.setText(f'{layer["opacity"]}%')
        self.combo_blend.setCurrentText(layer["blend"])
        self._loading = False

    def on_prop_changed(self, *args):
        if self._loading:
            return
        layer = self._current()
        if layer is None:
            return
        layer["x"] = self.spin_x.value(); layer["y"] = self.spin_y.value()
        layer["scale"] = self.spin_scale.value(); layer["rotate"] = self.spin_rotate.value()
        layer["opacity"] = self.slider_opacity.value(); layer["blend"] = self.combo_blend.currentText()
        self.lbl_opacity.setText(f'{layer["opacity"]}%')
        self.canvas.update()

    def center_current(self):
        layer = self._current()
        if layer is None:
            return
        _, _, ow, oh = self.layer_bbox(layer)
        layer["x"] = int((self.base_pil.width - ow) // 2)
        layer["y"] = int((self.base_pil.height - oh) // 2)
        self._load_props()
        self.canvas.update()

    # ---- compositing ----
    def layer_bbox(self, layer):
        sw = layer["image"].width * layer["scale"] / 100.0
        sh = layer["image"].height * layer["scale"] / 100.0
        if layer["rotate"]:
            r = math.radians(layer["rotate"])
            c, s = abs(math.cos(r)), abs(math.sin(r))
            ow, oh = sw * c + sh * s, sw * s + sh * c
        else:
            ow, oh = sw, sh
        return (layer["x"], layer["y"], ow, oh)

    def _scaled_base(self, scale):
        key = round(scale, 4)
        if key not in self._base_cache:
            if scale == 1.0:
                self._base_cache[key] = self.base_pil.copy()
            else:
                bw, bh = self.base_pil.size
                self._base_cache[key] = self.base_pil.resize(
                    (max(1, int(round(bw * scale))), max(1, int(round(bh * scale)))), Image.LANCZOS)
        return self._base_cache[key].copy()

    def _apply_layer(self, base, layer, scale):
        ov = layer["image"]
        s = (layer["scale"] / 100.0) * scale
        tw = max(1, int(round(ov.width * s)))
        th = max(1, int(round(ov.height * s)))
        ov = ov.resize((tw, th), Image.LANCZOS) if (tw, th) != ov.size else ov.copy()
        if layer["rotate"]:
            ov = ov.rotate(-layer["rotate"], expand=True, resample=Image.BICUBIC)
        opacity = layer["opacity"] / 100.0
        if opacity < 1.0:
            ov.putalpha(ov.getchannel('A').point(lambda a: int(a * opacity)))
        x = int(round(layer["x"] * scale)); y = int(round(layer["y"] * scale))
        canvas = Image.new('RGBA', base.size, (0, 0, 0, 0))
        canvas.alpha_composite(ov, (x, y))
        if layer["blend"] == "Normal":
            return Image.alpha_composite(base, canvas)
        return self._blend_math(base, canvas, layer["blend"])

    def _compose(self, scale=1.0):
        result = self._scaled_base(scale)
        for layer in self.layers:
            result = self._apply_layer(result, layer, scale)
        return result

    @staticmethod
    def _blend_math(base, layer, mode):
        b = np.array(base, dtype=np.float32) / 255.0
        l = np.array(layer, dtype=np.float32) / 255.0
        la = l[:, :, 3:4]
        bl_rgb, l_rgb = b[:, :, :3], l[:, :, :3]
        if mode == "Multiply":
            blended = bl_rgb * l_rgb
        elif mode == "Screen":
            blended = 1 - (1 - bl_rgb) * (1 - l_rgb)
        elif mode == "Overlay":
            blended = np.where(bl_rgb <= 0.5, 2 * bl_rgb * l_rgb,
                               1 - 2 * (1 - bl_rgb) * (1 - l_rgb))
        else:
            blended = l_rgb
        out_rgb = bl_rgb * (1 - la) + blended * la
        out = np.dstack([out_rgb, np.maximum(b[:, :, 3:4], la)])
        return Image.fromarray((np.clip(out, 0, 1) * 255).astype(np.uint8), 'RGBA')

    def accept_merge(self):
        if not self.layers:
            self.reject()
            return
        try:
            self.result_pil = self._compose(scale=1.0)
            self.accept()
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to merge: {e}")



# ======================================================================
# PDF export dialog
# ======================================================================

class PdfExportDialog(QDialog):
    PAGE_SIZES_MM = {**PAGE_SIZES_MM, "Original size": None}

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Export as PDF")
        self.setMinimumWidth(340)
        layout = QVBoxLayout(self)
        form = QFormLayout()

        self.combo_page = QComboBox(); self.combo_page.addItems(self.PAGE_SIZES_MM.keys())
        self.combo_orient = QComboBox(); self.combo_orient.addItems(["Auto", "Portrait", "Landscape"])
        self.spin_dpi = QSpinBox(); self.spin_dpi.setRange(72, 600); self.spin_dpi.setValue(150); self.spin_dpi.setSuffix(" DPI")
        self.spin_margin = QDoubleSpinBox(); self.spin_margin.setRange(0, 50); self.spin_margin.setValue(10); self.spin_margin.setSuffix(" mm")
        self.combo_fit = QComboBox(); self.combo_fit.addItems(["Fit (contain)", "Fill (cover)"])

        form.addRow("Page size:", self.combo_page)
        form.addRow("Orientation:", self.combo_orient)
        form.addRow("Resolution:", self.spin_dpi)
        form.addRow("Margin:", self.spin_margin)
        form.addRow("Scaling:", self.combo_fit)
        layout.addLayout(form)

        self.combo_page.currentTextChanged.connect(self._toggle_page_opts)
        self._toggle_page_opts(self.combo_page.currentText())

        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Save | QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def _toggle_page_opts(self, text):
        is_page = self.PAGE_SIZES_MM.get(text) is not None
        for w in (self.combo_orient, self.spin_margin, self.combo_fit):
            w.setEnabled(is_page)

    def options(self):
        return {
            "page": self.combo_page.currentText(),
            "orientation": self.combo_orient.currentText().lower(),
            "dpi": self.spin_dpi.value(),
            "margin_mm": self.spin_margin.value(),
            "fit": "fill" if self.combo_fit.currentIndex() == 1 else "fit",
        }


def export_to_pdf(path, image_rgb, opts):
    dpi = opts["dpi"]
    page_dims = PdfExportDialog.PAGE_SIZES_MM.get(opts["page"])

    if page_dims is None:  # Original size
        image_rgb.save(path, "PDF", resolution=float(dpi))
        return

    pw_mm, ph_mm = page_dims
    orient = opts["orientation"]
    if orient == "landscape" or (orient == "auto" and image_rgb.width > image_rgb.height):
        pw_mm, ph_mm = ph_mm, pw_mm

    pw = max(1, int(round(pw_mm / 25.4 * dpi)))
    ph = max(1, int(round(ph_mm / 25.4 * dpi)))
    margin = int(round(opts["margin_mm"] / 25.4 * dpi))
    avail_w = max(1, pw - 2 * margin)
    avail_h = max(1, ph - 2 * margin)

    iw, ih = image_rgb.size
    if opts["fit"] == "fill":
        scale = max(avail_w / iw, avail_h / ih)
    else:
        scale = min(avail_w / iw, avail_h / ih)
    new_w, new_h = max(1, int(iw * scale)), max(1, int(ih * scale))
    resized = image_rgb.resize((new_w, new_h), Image.LANCZOS)

    page = Image.new("RGB", (pw, ph), (255, 255, 255))
    ox, oy = (pw - new_w) // 2, (ph - new_h) // 2
    # For "fill", crop the overflow
    if new_w > pw or new_h > ph:
        crop_l = max(0, (new_w - pw) // 2)
        crop_t = max(0, (new_h - ph) // 2)
        resized = resized.crop((crop_l, crop_t, crop_l + min(new_w, pw), crop_t + min(new_h, ph)))
        new_w, new_h = resized.size
        ox, oy = (pw - new_w) // 2, (ph - new_h) // 2
    page.paste(resized, (ox, oy))
    page.save(path, "PDF", resolution=float(dpi))


# ======================================================================
# Page-size background canvas
# ======================================================================

class PageBackgroundDialog(QDialog):
    """Place the current image onto a standard page-sized background."""

    FIT_MODES = ["Fit (contain)", "Fill (cover)", "Center (no scale)", "Stretch"]

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Set Page Background")
        self.setMinimumWidth(360)
        self.bg_color = QColor(255, 255, 255)
        layout = QVBoxLayout(self)
        form = QFormLayout()

        self.combo_page = QComboBox(); self.combo_page.addItems(list(PAGE_SIZES_MM.keys()) + ["Custom…"])
        self.combo_page.setCurrentText("A4")
        self.combo_orient = QComboBox(); self.combo_orient.addItems(["Portrait", "Landscape"])
        self.spin_dpi = QSpinBox(); self.spin_dpi.setRange(72, 600); self.spin_dpi.setValue(150); self.spin_dpi.setSuffix(" DPI")
        self.spin_cw = QDoubleSpinBox(); self.spin_cw.setRange(1, 2000); self.spin_cw.setValue(210); self.spin_cw.setSuffix(" mm")
        self.spin_ch = QDoubleSpinBox(); self.spin_ch.setRange(1, 2000); self.spin_ch.setValue(297); self.spin_ch.setSuffix(" mm")
        self.combo_fit = QComboBox(); self.combo_fit.addItems(self.FIT_MODES)
        self.spin_margin = QDoubleSpinBox(); self.spin_margin.setRange(0, 100); self.spin_margin.setValue(10); self.spin_margin.setSuffix(" mm")

        self.combo_bg = QComboBox(); self.combo_bg.addItems(["White", "Black", "Transparent", "Custom…"])
        self.btn_bg_color = QPushButton("Pick…"); self.btn_bg_color.setEnabled(False)
        bg_row = QHBoxLayout(); bg_row.addWidget(self.combo_bg, 1); bg_row.addWidget(self.btn_bg_color)

        form.addRow("Page size:", self.combo_page)
        form.addRow("Orientation:", self.combo_orient)
        form.addRow("Custom width:", self.spin_cw)
        form.addRow("Custom height:", self.spin_ch)
        form.addRow("Resolution:", self.spin_dpi)
        form.addRow("Image scaling:", self.combo_fit)
        form.addRow("Margin:", self.spin_margin)
        form.addRow("Background:", bg_row)
        layout.addLayout(form)

        self.combo_page.currentTextChanged.connect(self._toggle_custom)
        self.combo_bg.currentTextChanged.connect(self._toggle_bg)
        self.btn_bg_color.clicked.connect(self._pick_color)
        self._toggle_custom(self.combo_page.currentText())

        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        buttons.button(QDialogButtonBox.StandardButton.Ok).setText("Apply")
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def _toggle_custom(self, text):
        custom = (text == "Custom…")
        self.spin_cw.setEnabled(custom); self.spin_ch.setEnabled(custom)
        self.combo_orient.setEnabled(not custom)

    def _toggle_bg(self, text):
        self.btn_bg_color.setEnabled(text == "Custom…")

    def _pick_color(self):
        c = QColorDialog.getColor(self.bg_color, self, "Background Color")
        if c.isValid():
            self.bg_color = c

    def options(self):
        page = self.combo_page.currentText()
        if page == "Custom…":
            pw_mm, ph_mm = self.spin_cw.value(), self.spin_ch.value()
        else:
            pw_mm, ph_mm = PAGE_SIZES_MM[page]
            if self.combo_orient.currentText() == "Landscape":
                pw_mm, ph_mm = ph_mm, pw_mm

        bg_choice = self.combo_bg.currentText()
        if bg_choice == "White":
            bg = (255, 255, 255, 255)
        elif bg_choice == "Black":
            bg = (0, 0, 0, 255)
        elif bg_choice == "Transparent":
            bg = (0, 0, 0, 0)
        else:
            bg = self.bg_color.getRgb()

        return {
            "page_mm": (pw_mm, ph_mm),
            "dpi": self.spin_dpi.value(),
            "fit": self.combo_fit.currentText(),
            "margin_mm": self.spin_margin.value(),
            "bg": bg,
        }


def create_page_background(image, opts):
    """Return an RGBA image of the current image placed on a page canvas."""
    pw_mm, ph_mm = opts["page_mm"]
    dpi = opts["dpi"]
    pw = max(1, int(round(pw_mm / 25.4 * dpi)))
    ph = max(1, int(round(ph_mm / 25.4 * dpi)))
    margin = int(round(opts["margin_mm"] / 25.4 * dpi))

    canvas = Image.new("RGBA", (pw, ph), tuple(opts["bg"]))
    img = image.convert("RGBA")
    iw, ih = img.size
    avail_w, avail_h = max(1, pw - 2 * margin), max(1, ph - 2 * margin)
    fit = opts["fit"]

    if fit.startswith("Stretch"):
        placed = img.resize((avail_w, avail_h), Image.LANCZOS)
    elif fit.startswith("Center"):
        placed = img
    else:
        if fit.startswith("Fill"):
            scale = max(avail_w / iw, avail_h / ih)
        else:  # Fit
            scale = min(avail_w / iw, avail_h / ih)
        placed = img.resize((max(1, int(iw * scale)), max(1, int(ih * scale))), Image.LANCZOS)

    # Crop overflow (Fill / oversized Center) to the available area
    pw_i, ph_i = placed.size
    if pw_i > avail_w or ph_i > avail_h:
        crop_l = max(0, (pw_i - avail_w) // 2)
        crop_t = max(0, (ph_i - avail_h) // 2)
        placed = placed.crop((crop_l, crop_t,
                              crop_l + min(pw_i, avail_w), crop_t + min(ph_i, avail_h)))
        pw_i, ph_i = placed.size

    ox = (pw - pw_i) // 2
    oy = (ph - ph_i) // 2
    canvas.alpha_composite(placed, (ox, oy))
    return canvas


# ======================================================================
# AI model download manager
# ======================================================================

class OperationWorker(QThread):
    """Runs a heavy image operation off the UI thread so it can report stage
    messages and be cancelled. `func(image, report)` returns the result image;
    `report(str)` posts a status message back to the UI."""
    finished_ok = Signal(object)
    failed = Signal(str)
    progress = Signal(str)

    def __init__(self, func, image):
        super().__init__()
        self.func = func
        self.image = image
        self._cancelled = False

    def cancel(self):
        self._cancelled = True

    @property
    def cancelled(self):
        return self._cancelled

    def run(self):
        try:
            result = self.func(self.image, self.progress.emit)
            if not self._cancelled:
                self.finished_ok.emit(result)
        except Exception as e:
            if not self._cancelled:
                self.failed.emit(str(e))


class SvgExportDialog(QDialog):
    """Choose how to export the current image as a vector (SVG) file."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Export as SVG (vector)")
        self.setMinimumWidth(440)
        self._fill_color = QColor(0, 0, 0)

        layout = QVBoxLayout(self)

        mode_group = QGroupBox("What to export")
        mg = QVBoxLayout(mode_group)
        self.radio_embed = QRadioButton("Embedded image  (full colour, scalable container)")
        self.radio_outline = QRadioButton("Vector outline  (trace the cut-out silhouette)")
        self.radio_embed.setChecked(True)
        mg.addWidget(self.radio_embed)
        mg.addWidget(self.radio_outline)
        layout.addWidget(mode_group)

        self.outline_group = QGroupBox("Outline options")
        og = QFormLayout(self.outline_group)
        self.combo_fill = QComboBox()
        self.combo_fill.addItems(["Clip embedded image (keep colours)", "Solid colour fill"])
        self.btn_color = QPushButton("Choose colour…")
        self.btn_color.setEnabled(False)
        self.spin_alpha = QSpinBox(); self.spin_alpha.setRange(1, 255); self.spin_alpha.setValue(128)
        self.spin_simplify = QDoubleSpinBox()
        self.spin_simplify.setRange(0.0, 20.0); self.spin_simplify.setSingleStep(0.5)
        self.spin_simplify.setValue(1.5); self.spin_simplify.setSuffix(" px")
        og.addRow("Fill:", self.combo_fill)
        og.addRow("", self.btn_color)
        og.addRow("Edge opacity ≥:", self.spin_alpha)
        og.addRow("Simplify:", self.spin_simplify)
        layout.addWidget(self.outline_group)

        if not CV2_AVAILABLE:
            self.radio_outline.setEnabled(False)
            self.radio_outline.setText(self.radio_outline.text() + "  (needs OpenCV)")

        self.note = QLabel(
            "Embedded keeps every pixel; the SVG scales without a resolution "
            "cap but the picture inside stays raster. Vector outline traces the "
            "transparent edge into true, infinitely-scalable paths — ideal for "
            "logos, stickers and cut files.")
        self.note.setWordWrap(True)
        self.note.setStyleSheet("color:#667; font-size:11px;")
        layout.addWidget(self.note)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        buttons.button(QDialogButtonBox.StandardButton.Ok).setText("Export…")
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

        self.radio_outline.toggled.connect(self._sync)
        self.combo_fill.currentIndexChanged.connect(self._sync)
        self.btn_color.clicked.connect(self._pick_color)
        self._sync()

    def _sync(self, *_):
        outline = self.radio_outline.isChecked()
        self.outline_group.setEnabled(outline)
        self.btn_color.setEnabled(outline and self.combo_fill.currentIndex() == 1)

    def _pick_color(self):
        col = QColorDialog.getColor(self._fill_color, self, "Outline fill colour")
        if col.isValid():
            self._fill_color = col
            self.btn_color.setStyleSheet(f"background:{col.name()}; color:white;")

    def options(self):
        if self.radio_embed.isChecked():
            return {"mode": "embed"}
        fill = None
        if self.combo_fill.currentIndex() == 1:
            fill = self._fill_color.getRgb()[:3]
        return {"mode": "outline", "fill_color": fill,
                "alpha_threshold": self.spin_alpha.value(),
                "simplify": self.spin_simplify.value()}


class ModelDownloadWorker(QThread):
    done = Signal(bool, str)

    def __init__(self, model_name):
        super().__init__()
        self.model_name = model_name

    def run(self):
        try:
            from rembg import new_session
            new_session(self.model_name)   # downloads into U2NET_HOME if missing
            self.done.emit(True, self.model_name)
        except Exception as e:
            self.done.emit(False, f"{self.model_name}: {e}")


class ModelManagerDialog(QDialog):
    """Download rembg AI models on demand and open their storage folder."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("AI Model Manager")
        self.setMinimumWidth(480)
        self._queue = []
        self._worker = None

        layout = QVBoxLayout(self)
        info = QLabel(f"Models are stored in:\n{MODELS_DIR}")
        info.setWordWrap(True)
        info.setStyleSheet("color:#556; font-size:11px;")
        layout.addWidget(info)

        self.list = QListWidget()
        layout.addWidget(self.list)

        self.progress = QLabel("")
        self.progress.setWordWrap(True)
        self.progress.setStyleSheet(f"color:{ACCENT};")
        layout.addWidget(self.progress)

        row = QHBoxLayout()
        self.btn_download = QPushButton("Download Selected")
        self.btn_download_all = QPushButton("Download All Missing")
        self.btn_open = QPushButton("Open Folder")
        row.addWidget(self.btn_download)
        row.addWidget(self.btn_download_all)
        row.addWidget(self.btn_open)
        layout.addLayout(row)

        close_box = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
        close_box.rejected.connect(self.reject)
        layout.addWidget(close_box)

        self.btn_download.clicked.connect(self.download_selected)
        self.btn_download_all.clicked.connect(self.download_all)
        self.btn_open.clicked.connect(
            lambda: QDesktopServices.openUrl(QUrl.fromLocalFile(MODELS_DIR)))
        self.refresh()

    def refresh(self):
        self.list.clear()
        for name in REMBG_MODELS:
            ok = model_is_downloaded(name)
            label = ("✓  " + name) if ok else ("•  " + name + "   (not downloaded)")
            self.list.addItem(QListWidgetItem(label))
        if self.list.count():
            self.list.setCurrentRow(0)

    def _set_busy(self, busy):
        self.btn_download.setEnabled(not busy)
        self.btn_download_all.setEnabled(not busy)

    def download_selected(self):
        i = self.list.currentRow()
        if 0 <= i < len(REMBG_MODELS):
            self._start([REMBG_MODELS[i]])

    def download_all(self):
        missing = [n for n in REMBG_MODELS if not model_is_downloaded(n)]
        if not missing:
            self.progress.setText("All models are already downloaded.")
            return
        self._start(missing)

    def _start(self, queue):
        if not ensure_rembg():
            QMessageBox.warning(self, "Unavailable",
                                "rembg / onnxruntime is not available, so models "
                                "cannot be downloaded.\n(pip install rembg onnxruntime)")
            return
        self._queue = list(queue)
        self._set_busy(True)
        self._download_next()

    def _download_next(self):
        if not self._queue:
            self._set_busy(False)
            self.progress.setText("Done.")
            self.refresh()
            return
        name = self._queue.pop(0)
        self.progress.setText(f"Downloading “{name}”…  (first download can take a while)")
        self._worker = ModelDownloadWorker(name)
        self._worker.done.connect(self._on_done)
        self._worker.start()

    def _on_done(self, ok, msg):
        if not ok:
            self.progress.setText(f"Error downloading {msg}")
        self.refresh()
        self._download_next()

    def closeEvent(self, event):
        if self._worker and self._worker.isRunning():
            self._worker.wait(100)
        event.accept()


# ======================================================================
# Update checker
# ======================================================================

class UpdateCheckWorker(QThread):
    """Fetch the latest GitHub release tag off the UI thread."""
    done = Signal(bool, str, str)   # ok, latest_tag_or_error, html_url

    def run(self):
        try:
            import requests
            resp = requests.get(UPDATE_API_URL, timeout=15,
                                headers={"Accept": "application/vnd.github+json"})
            resp.raise_for_status()
            data = resp.json()
            tag = data.get("tag_name") or data.get("name") or ""
            url = data.get("html_url") or GITHUB_URL
            if not tag:
                self.done.emit(False, "No release information found.", url)
            else:
                self.done.emit(True, tag, url)
        except Exception as e:
            self.done.emit(False, str(e), GITHUB_URL)


# ======================================================================
# Batch background removal
# ======================================================================

class BatchWorker(QThread):
    """Remove backgrounds (and optionally upscale) for a list of files,
    saving each result as a PNG into an output folder."""
    progress = Signal(int, int, str)     # index, total, message
    finished_all = Signal(int, int)      # succeeded, failed

    def __init__(self, files, out_dir, model, matting, providers,
                 upscale_model=None, upscale_scale=2, parent=None):
        super().__init__(parent)
        self.files = files
        self.out_dir = out_dir
        self.model = model
        self.matting = matting
        self.providers = providers
        self.upscale_model = upscale_model
        self.upscale_scale = upscale_scale
        self._cancel = False

    def cancel(self):
        self._cancel = True

    def run(self):
        ok = fail = 0
        session = None
        try:
            from rembg import new_session
            session = (new_session(self.model, providers=self.providers)
                       if self.providers else new_session(self.model))
        except Exception as e:
            print(f"Batch: could not create session ({e}); using per-call model.")
        total = len(self.files)
        for i, path in enumerate(self.files, 1):
            if self._cancel:
                break
            name = os.path.basename(path)
            self.progress.emit(i, total, f"Processing {name}…")
            try:
                img = Image.open(path).convert("RGBA")
                if session is not None:
                    result = remove_bg(img, session=session, alpha_matting=self.matting)
                else:
                    result = remove_bg(img, model=self.model, alpha_matting=self.matting)
                if not isinstance(result, Image.Image):
                    result = Image.open(io.BytesIO(result)).convert("RGBA")
                if self.upscale_model is not None:
                    result = upscale_image(result, scale=self.upscale_scale,
                                           model=self.upscale_model, use_ai=True)
                stem = os.path.splitext(name)[0]
                out_path = os.path.join(self.out_dir, f"{stem}_nobg.png")
                result.save(out_path)
                ok += 1
            except Exception as e:
                print(f"Batch failed for {name}: {e}")
                fail += 1
        self.finished_all.emit(ok, fail)


class BatchProcessDialog(QDialog):
    """Pick input images and an output folder, then remove backgrounds for all."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.main = parent
        self.setWindowTitle("Batch Background Removal")
        self.setMinimumWidth(560)
        self.files = []
        self.out_dir = ""
        self.worker = None

        lay = QVBoxLayout(self)

        self.list = QListWidget()
        self.list.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        lay.addWidget(QLabel("Input images:"))
        lay.addWidget(self.list, 1)

        file_row = QHBoxLayout()
        btn_add = QPushButton("Add Images…")
        btn_folder = QPushButton("Add Folder…")
        btn_clear = QPushButton("Clear")
        btn_add.clicked.connect(self._add_files)
        btn_folder.clicked.connect(self._add_folder)
        btn_clear.clicked.connect(lambda: (self.files.clear(), self.list.clear()))
        for b in (btn_add, btn_folder, btn_clear):
            file_row.addWidget(b)
        lay.addLayout(file_row)

        form = QFormLayout()
        self.combo_model = QComboBox(); self.combo_model.addItems(REMBG_MODELS)
        if self.main is not None:
            mi = self.combo_model.findText(self.main.rembg_model)
            if mi >= 0:
                self.combo_model.setCurrentIndex(mi)
        self.cb_matting = QCheckBox("Alpha matting (refined edges, slower)")
        self.cb_upscale = QCheckBox("Also upscale results")
        self.combo_up_model = QComboBox()
        for key, entry in SR_MODELS.items():
            self.combo_up_model.addItem(entry["label"], key)
        self.combo_up_scale = QComboBox(); self.combo_up_scale.addItems(["2×", "3×", "4×"])
        self.combo_up_model.setEnabled(False)
        self.combo_up_scale.setEnabled(False)
        self.cb_upscale.toggled.connect(self.combo_up_model.setEnabled)
        self.cb_upscale.toggled.connect(self.combo_up_scale.setEnabled)
        form.addRow("Model:", self.combo_model)
        form.addRow(self.cb_matting)
        form.addRow(self.cb_upscale)
        form.addRow("Upscaler:", self.combo_up_model)
        form.addRow("Scale:", self.combo_up_scale)
        lay.addLayout(form)

        out_row = QHBoxLayout()
        self.lbl_out = QLineEdit(); self.lbl_out.setPlaceholderText("Output folder…")
        self.lbl_out.setReadOnly(True)
        btn_out = QPushButton("Choose…")
        btn_out.clicked.connect(self._choose_out)
        out_row.addWidget(QLabel("Save to:"))
        out_row.addWidget(self.lbl_out, 1)
        out_row.addWidget(btn_out)
        lay.addLayout(out_row)

        self.bar = QProgressBar(); self.bar.setVisible(False)
        lay.addWidget(self.bar)
        self.status = QLabel("")
        lay.addWidget(self.status)

        self.buttons = QDialogButtonBox()
        self.btn_start = self.buttons.addButton("Start", QDialogButtonBox.ButtonRole.AcceptRole)
        self.btn_close = self.buttons.addButton(QDialogButtonBox.StandardButton.Close)
        self.btn_start.clicked.connect(self._start)
        self.btn_close.clicked.connect(self.reject)
        lay.addWidget(self.buttons)

    def _add_files(self):
        paths, _ = QFileDialog.getOpenFileNames(self, "Add Images", "",
            "Images (*.png *.jpg *.jpeg *.bmp *.webp *.tiff *.gif *.avif)")
        self._append(paths)

    def _add_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "Add Folder")
        if not folder:
            return
        exts = (".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tiff", ".gif", ".avif")
        found = [os.path.join(folder, f) for f in sorted(os.listdir(folder))
                 if f.lower().endswith(exts)]
        self._append(found)

    def _append(self, paths):
        for p in paths:
            if p and p not in self.files:
                self.files.append(p)
                self.list.addItem(os.path.basename(p))

    def _choose_out(self):
        folder = QFileDialog.getExistingDirectory(self, "Output Folder")
        if folder:
            self.out_dir = folder
            self.lbl_out.setText(folder)

    def _start(self):
        if not self.files:
            QMessageBox.information(self, "Batch", "Add some images first.")
            return
        if not self.out_dir:
            QMessageBox.information(self, "Batch", "Choose an output folder.")
            return
        providers = None
        if self.main is not None:
            use_gpu = self.main.settings.value("gpu_acceleration", True, type=bool)
            providers = preferred_ort_providers(use_gpu)
        up_model = self.combo_up_model.currentData() if self.cb_upscale.isChecked() else None
        up_scale = int(self.combo_up_scale.currentText().replace("×", ""))
        self.bar.setVisible(True)
        self.bar.setRange(0, len(self.files))
        self.btn_start.setEnabled(False)
        self.worker = BatchWorker(list(self.files), self.out_dir,
                                  self.combo_model.currentText(),
                                  self.cb_matting.isChecked(), providers,
                                  upscale_model=up_model, upscale_scale=up_scale)
        self.worker.progress.connect(self._on_progress)
        self.worker.finished_all.connect(self._on_done)
        self.worker.start()

    def _on_progress(self, i, total, msg):
        self.bar.setValue(i)
        self.status.setText(msg)

    def _on_done(self, ok, fail):
        self.btn_start.setEnabled(True)
        self.status.setText(f"Done. {ok} succeeded, {fail} failed.")
        QMessageBox.information(self, "Batch Complete",
                                f"Processed {ok + fail} image(s).\n"
                                f"{ok} succeeded, {fail} failed.\n\nSaved to:\n{self.out_dir}")

    def closeEvent(self, event):
        if self.worker and self.worker.isRunning():
            self.worker.cancel()
            self.worker.wait(3000)
        event.accept()


# ======================================================================
# Preferences
# ======================================================================

class PreferencesDialog(QDialog):
    """Edit persisted application preferences."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.settings = parent.settings if parent is not None else QSettings(APP_ORG, APP_NAME)
        self.setWindowTitle("Preferences")
        self.setMinimumWidth(420)

        lay = QVBoxLayout(self)
        form = QFormLayout()

        self.combo_theme = QComboBox(); self.combo_theme.addItems(["Light", "Dark"])
        self.combo_theme.setCurrentText(
            self.settings.value("theme", "light", type=str).capitalize())

        self.combo_format = QComboBox()
        self.combo_format.addItems(["png", "jpg", "tiff", "webp", "avif"])
        self.combo_format.setCurrentText(self.settings.value("default_format", "png", type=str))

        self.spin_quality = QSpinBox(); self.spin_quality.setRange(1, 100)
        self.spin_quality.setValue(self.settings.value("save_quality", 92, type=int))

        self.cb_gpu = QCheckBox("Use hardware acceleration (GPU / CoreML) when available")
        self.cb_gpu.setChecked(self.settings.value("gpu_acceleration", True, type=bool))

        self.cb_update = QCheckBox("Check for updates on startup")
        self.cb_update.setChecked(self.settings.value("check_updates", False, type=bool))

        form.addRow("Theme:", self.combo_theme)
        form.addRow("Default save format:", self.combo_format)
        form.addRow("JPEG/WebP/AVIF quality:", self.spin_quality)
        form.addRow(self.cb_gpu)
        form.addRow(self.cb_update)
        lay.addLayout(form)

        note = QLabel("Hardware acceleration needs the matching onnxruntime build "
                      "(e.g. onnxruntime-gpu on CUDA, or onnxruntime-silicon / CoreML on macOS).")
        note.setWordWrap(True)
        note.setStyleSheet("color:#889; font-size:10px;")
        lay.addWidget(note)

        btns = QDialogButtonBox(QDialogButtonBox.StandardButton.Save |
                                QDialogButtonBox.StandardButton.Cancel)
        btns.accepted.connect(self._save)
        btns.rejected.connect(self.reject)
        lay.addWidget(btns)

    def _save(self):
        self.settings.setValue("theme", self.combo_theme.currentText().lower())
        self.settings.setValue("default_format", self.combo_format.currentText())
        self.settings.setValue("save_quality", self.spin_quality.value())
        self.settings.setValue("gpu_acceleration", self.cb_gpu.isChecked())
        self.settings.setValue("check_updates", self.cb_update.isChecked())
        self.accept()


# ======================================================================
# Main window
# ======================================================================

class MainWindow(QMainWindow):
    MAX_HISTORY = 20

    def __init__(self):
        super().__init__()
        # Persistent user settings (window geometry, theme, recent files, …).
        self.settings = QSettings(APP_ORG, APP_NAME)
        self.recent_files = self.settings.value("recent_files", [], type=list) or []
        self.current_theme = self.settings.value("theme", "light", type=str)
        self._rembg_sessions = {}       # model_name -> cached rembg session

        self.setWindowTitle(f"{APP_NAME} — {APP_TAGLINE}")
        self.setWindowIcon(get_app_icon())
        # Keep the window usable on small / low-resolution displays: never let
        # it shrink below a workable minimum, and clamp the preferred launch
        # size to whatever the current screen can actually show.
        self.setMinimumSize(900, 600)
        self.temp_files_to_clean = []

        self.original_pil_image, self.current_pil_image = None, None
        self.original_qpixmap, self.current_qpixmap = None, None
        self.background_color = None

        self.undo_stack, self.redo_stack = [], []

        self.rembg_model = self.settings.value("rembg_model", "u2net", type=str)
        self.alpha_matting_enabled = False
        self.fg_threshold, self.bg_threshold, self.erode_size = 240, 10, 10
        self.brush_size = 20
        self.selected_color_rgb = None
        self.preview_angle = 0.0        # live (unbaked) rotation angle, degrees CW
        self._active_workers = []       # background OperationWorker threads
        self.setAcceptDrops(True)

        self._create_widgets()
        self._create_actions()
        self._create_menu_bar()
        self._create_toolbars()
        self._create_layout()
        self._create_status_bar()
        self._connect_signals()
        self._restore_ui_settings()

        # Restore the last window geometry, or fit a sensible size to the screen.
        geo = self.settings.value("geometry")
        if geo is not None and not self.restoreGeometry(geo):
            self._apply_initial_geometry(preferred=(1440, 900))
        elif geo is None:
            self._apply_initial_geometry(preferred=(1440, 900))

        QApplication.clipboard().dataChanged.connect(self._update_ui_states)
        self._update_ui_states()

    def _restore_ui_settings(self):
        """Push persisted preferences into the freshly-built widgets."""
        s = self.settings
        idx = self.model_combo.findText(s.value("rembg_model", "u2net", type=str))
        if idx >= 0:
            self.model_combo.setCurrentIndex(idx)
        up_model = s.value("upscale_model", "EDSR", type=str)
        ui = self.combo_upscale_method.findData(None if up_model == "" else up_model)
        if ui >= 0:
            self.combo_upscale_method.setCurrentIndex(ui)
            self._refresh_upscale_scales()
        self.chk_upscale_denoise.setChecked(s.value("upscale_denoise", False, type=bool))
        self.chk_upscale_sharpen.setChecked(s.value("upscale_sharpen", False, type=bool))

    def _apply_initial_geometry(self, preferred=(1440, 900)):
        """Size and centre the window, fitting it to the available screen so it
        is never larger than the desktop (accounts for taskbars/menu bars and
        any display scale). Falls back to the preferred size if no screen info
        is available."""
        pref_w, pref_h = preferred
        screen = QApplication.primaryScreen()
        if screen is not None:
            avail = screen.availableGeometry()
            # Leave a small margin so the frame/shadow isn't flush to the edges.
            w = min(pref_w, avail.width() - 40)
            h = min(pref_h, avail.height() - 60)
            w = max(w, self.minimumWidth())
            h = max(h, self.minimumHeight())
            x = avail.x() + (avail.width() - w) // 2
            y = avail.y() + (avail.height() - h) // 2
            self.setGeometry(x, y, w, h)
        else:
            self.resize(pref_w, pref_h)

    # ---- widget / action creation ----
    def _create_widgets(self):
        self.image_label_preview = InteractiveLabel()
        self.image_label_preview.setToolTip("Preview / Edit area. Use the tools panel on the right.")
        self.scroll_area_preview = QScrollArea()
        self.scroll_area_preview.setWidget(self.image_label_preview)
        self.scroll_area_preview.setWidgetResizable(False)
        self.scroll_area_preview.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.image_label_preview.set_scroll_area(self.scroll_area_preview)

    def _icon(self, sp):
        return self.style().standardIcon(sp)

    def _zoom_icon(self, kind, color=None):
        """Draw a crisp magnifying-glass icon ('in' / 'out' / 'fit') so the
        zoom buttons have real icons in the icon-only toolbar. The tone follows
        the active theme so it reads well on both light and dark toolbars."""
        if color is None:
            color = "#c3c9d4" if getattr(self, "current_theme", "light") == "dark" else "#5c6270"
        size = 40
        pm = QPixmap(size, size)
        pm.fill(Qt.GlobalColor.transparent)
        pt = QPainter(pm)
        pt.setRenderHint(QPainter.RenderHint.Antialiasing)
        col = QColor(color)
        pen = QPen(col, 3.2, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap)
        pt.setPen(pen)
        cx, cy, r = 16, 16, 9
        pt.drawEllipse(QPoint(cx, cy), r, r)          # lens
        pt.drawLine(QPoint(cx + 7, cy + 7), QPoint(31, 31))   # handle
        pt.setPen(QPen(col, 2.6, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap))
        if kind in ("in", "out"):
            pt.drawLine(QPoint(cx - 4, cy), QPoint(cx + 4, cy))       # minus
            if kind == "in":
                pt.drawLine(QPoint(cx, cy - 4), QPoint(cx, cy + 4))   # plus
        elif kind == "fit":
            pt.drawRect(cx - 4, cy - 4, 8, 8)                         # fit box
        pt.end()
        return QIcon(pm)

    def _create_actions(self):
        SP = QStyle.StandardPixmap
        self.action_open = QAction(self._icon(SP.SP_DialogOpenButton), "Open…", self,
                                   shortcut=QKeySequence.StandardKey.Open, toolTip="Open Image (Ctrl+O)")
        self.action_paste = QAction(self._icon(SP.SP_FileDialogDetailedView), "Paste Image", self,
                                    shortcut=QKeySequence.StandardKey.Paste, toolTip="Paste from Clipboard (Ctrl+V)")
        self.action_save = QAction(self._icon(SP.SP_DialogSaveButton), "Save As…", self,
                                   shortcut=QKeySequence.StandardKey.SaveAs, toolTip="Save Image (Ctrl+Shift+S)")
        self.action_export_pdf = QAction(self._icon(SP.SP_FileIcon), "Export as PDF…", self,
                                         shortcut="Ctrl+P", toolTip="Export as PDF (Ctrl+P)")
        self.action_copy = QAction(self._icon(SP.SP_FileDialogContentsView), "Copy Image", self,
                                   shortcut=QKeySequence.StandardKey.Copy, toolTip="Copy to Clipboard (Ctrl+C)")
        self.action_overlay = QAction(self._icon(SP.SP_FileDialogNewFolder), "Overlay Images…", self,
                                      toolTip="Overlay / composite one or more images")
        self.action_page_bg = QAction(self._icon(SP.SP_FileDialogListView), "Page Background…", self,
                                      toolTip="Place image on an A4/A5/Letter… page")
        self.action_quit = QAction("Quit", self, shortcut=QKeySequence.StandardKey.Quit, toolTip="Exit (Ctrl+Q)")
        self.action_undo = QAction(self._icon(SP.SP_ArrowBack), "Undo", self,
                                   shortcut=QKeySequence.StandardKey.Undo, toolTip="Undo (Ctrl+Z)")
        self.action_redo = QAction(self._icon(SP.SP_ArrowForward), "Redo", self,
                                   shortcut="Ctrl+Y", toolTip="Redo (Ctrl+Y)")
        self.action_reset = QAction(self._icon(SP.SP_BrowserReload), "Reset Image", self,
                                    toolTip="Reset to original loaded image")
        self.action_zoom_in = QAction(self._zoom_icon("in"), "Zoom In", self, shortcut="Ctrl+=", toolTip="Zoom In (Ctrl++)")
        self.action_zoom_out = QAction(self._zoom_icon("out"), "Zoom Out", self, shortcut="Ctrl+-", toolTip="Zoom Out (Ctrl+-)")
        self.action_zoom_reset = QAction(self._zoom_icon("fit"), "Fit to View", self, shortcut="Ctrl+0", toolTip="Fit image to view (Ctrl+0)")
        self.action_info = QAction(self._icon(SP.SP_MessageBoxInformation), "About", self, toolTip="About & help")
        self.action_batch = QAction(self._icon(SP.SP_DirIcon), "Batch Background Removal…", self,
                                    toolTip="Remove backgrounds from a whole folder of images")
        self.action_prefs = QAction("Preferences…", self, shortcut="Ctrl+,",
                                    toolTip="Application preferences")
        self.action_theme = QAction("Dark Mode", self, checkable=True,
                                    toolTip="Toggle light / dark theme")
        self.action_theme.setChecked(self.current_theme == "dark")
        self.action_shortcuts = QAction("Keyboard Shortcuts", self, shortcut="F1",
                                        toolTip="Show keyboard shortcuts")
        self.action_update = QAction("Check for Updates…", self,
                                     toolTip="Check GitHub for a newer version")

    def _create_menu_bar(self):
        menubar = self.menuBar()
        m_file = menubar.addMenu("&File")
        m_file.addActions([self.action_open, self.action_paste])
        self.menu_recent = m_file.addMenu("Open &Recent")
        self._rebuild_recent_menu()
        m_file.addSeparator()
        m_file.addActions([self.action_save, self.action_export_pdf, self.action_copy])
        m_file.addSeparator()
        m_file.addAction(self.action_quit)

        m_edit = menubar.addMenu("&Edit")
        m_edit.addActions([self.action_undo, self.action_redo, self.action_reset])
        m_edit.addSeparator()
        m_edit.addAction(self.action_prefs)

        m_tools = menubar.addMenu("&Tools")
        m_tools.addActions([self.action_overlay, self.action_page_bg])
        m_tools.addSeparator()
        m_tools.addAction(self.action_batch)

        m_view = menubar.addMenu("&View")
        m_view.addActions([self.action_zoom_in, self.action_zoom_out, self.action_zoom_reset])
        m_view.addSeparator()
        m_view.addAction(self.action_theme)

        m_help = menubar.addMenu("&Help")
        m_help.addAction(self.action_info)
        m_help.addAction(self.action_shortcuts)
        m_help.addSeparator()
        m_help.addAction(self.action_update)

    def _create_toolbars(self):
        # A single, icon-only toolbar. Hovering any button shows a descriptive
        # tooltip (set on each QAction). When the window is too narrow to show
        # every icon, Qt adds a ">>" overflow button that drops down the rest,
        # so nothing is ever hidden.
        self.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonIconOnly)
        self.main_toolbar = self.addToolBar("Tools")
        self.main_toolbar.setMovable(False)
        self.main_toolbar.setFloatable(False)
        self.main_toolbar.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonIconOnly)
        self.main_toolbar.setIconSize(QSize(22, 22))

        # File group
        self.main_toolbar.addActions([self.action_open, self.action_paste,
                                      self.action_save, self.action_export_pdf,
                                      self.action_copy])
        self.main_toolbar.addSeparator()
        # Compose / tools group
        self.main_toolbar.addActions([self.action_overlay, self.action_page_bg,
                                      self.action_batch])
        self.main_toolbar.addSeparator()
        # Edit group
        self.main_toolbar.addActions([self.action_undo, self.action_redo,
                                      self.action_reset])
        self.main_toolbar.addSeparator()
        # View / zoom group
        self.main_toolbar.addActions([self.action_zoom_out, self.action_zoom_in,
                                      self.action_zoom_reset])
        self.main_toolbar.addSeparator()
        # Help
        self.main_toolbar.addAction(self.action_info)

    # ---- layout ----
    def _create_layout(self):
        control_widget = QWidget()
        control_layout = QVBoxLayout(control_widget)
        control_layout.setContentsMargins(8, 8, 8, 8)
        control_widget.setFixedWidth(370)

        tab_widget = QTabWidget()
        pages = [
            ("AI Tools", self._build_ai_tab()),
            ("Manual Edit", self._build_manual_tab()),
            ("Correct", self._build_correct_tab()),
            ("Compose", self._build_compose_tab()),
        ]
        for name, page in pages:
            tab_widget.addTab(page, name)

        # Fixed, equal-width tabs that always fit the panel (no scroll arrows)
        tab_bar = tab_widget.tabBar()
        tab_bar.setExpanding(False)
        tab_bar.setUsesScrollButtons(False)
        tab_bar.setElideMode(Qt.TextElideMode.ElideNone)
        tab_px = (370 - 20) // len(pages)   # leave a little slack for borders
        tab_bar.setStyleSheet(
            f"QTabBar::tab {{ width: {tab_px}px; min-width: {tab_px}px; "
            f"max-width: {tab_px}px; padding: 8px 0px; margin: 0px; }}")

        # Fix the tab pane to the tallest page's height so no tab ever needs to
        # scroll and switching tabs doesn't resize the panel.
        content_w = 370 - 24
        max_h = 0
        for _, page in pages:
            page.setMinimumWidth(content_w)
            hint = page.sizeHint().height()
            # account for word-wrapped labels whose height grows as width shrinks
            heightForWidth = page.heightForWidth(content_w) if page.hasHeightForWidth() else 0
            max_h = max(max_h, hint, heightForWidth)
        tab_bar_h = tab_widget.tabBar().sizeHint().height()
        tab_widget.setMinimumHeight(max_h + tab_bar_h + 28)
        control_layout.addWidget(tab_widget)

        # Wrap the fixed-width tools panel in a vertical scroll area so every
        # control stays reachable even on short screens / high UI scaling.
        control_scroll = QScrollArea()
        control_scroll.setWidgetResizable(True)
        control_scroll.setWidget(control_widget)
        control_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        control_scroll.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        control_scroll.setFixedWidth(370 + 18)   # panel + room for a scrollbar

        main_splitter = QSplitter(Qt.Orientation.Horizontal)
        main_splitter.addWidget(self.scroll_area_preview)
        main_splitter.addWidget(control_scroll)
        main_splitter.setStretchFactor(0, 1)
        main_splitter.setSizes([1050, 388])
        self.setCentralWidget(main_splitter)

    def _build_ai_tab(self):
        w = QWidget()
        lay = QVBoxLayout(w)
        rembg_group = QGroupBox("AI Background Removal (rembg)")
        rl = QFormLayout(rembg_group)
        self.btn_rembg = QPushButton("  Remove Background")
        self.btn_rembg.setObjectName("primaryButton")
        self.model_combo = QComboBox()
        self.model_combo.addItems(REMBG_MODELS)
        self.cb_alpha_matting = QCheckBox("Enable Alpha Matting (refined edges)")
        self.spin_fg_thresh = QSpinBox(); self.spin_fg_thresh.setRange(1, 254); self.spin_fg_thresh.setValue(self.fg_threshold)
        self.spin_bg_thresh = QSpinBox(); self.spin_bg_thresh.setRange(1, 254); self.spin_bg_thresh.setValue(self.bg_threshold)
        self.spin_erode_size = QSpinBox(); self.spin_erode_size.setRange(0, 50); self.spin_erode_size.setValue(self.erode_size)
        rl.addRow(self.btn_rembg)
        rl.addRow("Model:", self.model_combo)
        rl.addRow(self.cb_alpha_matting)
        rl.addRow("FG Threshold:", self.spin_fg_thresh)
        rl.addRow("BG Threshold:", self.spin_bg_thresh)
        rl.addRow("Erode Size:", self.spin_erode_size)
        lay.addWidget(rembg_group)

        enhance_group = QGroupBox("AI Enhance (Sharpen / Upscale)")
        el = QFormLayout(enhance_group)
        self.slider_sharpen = QSlider(Qt.Orientation.Horizontal)
        self.slider_sharpen.setRange(0, 300); self.slider_sharpen.setValue(150)
        self.lbl_sharpen = QLabel("150%")
        self.slider_sharpen.valueChanged.connect(lambda v: self.lbl_sharpen.setText(f"{v}%"))
        sh_row = QHBoxLayout(); sh_row.addWidget(self.slider_sharpen); sh_row.addWidget(self.lbl_sharpen)
        # Radius controls how wide the sharpening halo is (fine detail vs. edges).
        self.spin_sharpen_radius = QDoubleSpinBox()
        self.spin_sharpen_radius.setRange(0.5, 10.0)
        self.spin_sharpen_radius.setSingleStep(0.5)
        self.spin_sharpen_radius.setValue(2.0)
        self.spin_sharpen_radius.setSuffix(" px")
        self.btn_sharpen = QPushButton("Sharpen")
        el.addRow("Amount:", sh_row)
        el.addRow("Radius:", self.spin_sharpen_radius)
        el.addRow(self.btn_sharpen)
        # Upscaler model: Real-ESRGAN, every dnn_superres network, and Lanczos.
        self.combo_upscale_method = QComboBox()
        for key, entry in REALESRGAN_MODELS.items():
            self.combo_upscale_method.addItem(entry["label"], key)
        for key, entry in SR_MODELS.items():
            self.combo_upscale_method.addItem(entry["label"], key)
        self.combo_upscale_method.addItem(LANCZOS_LABEL, None)
        # Default to EDSR (best classic quality, no huge download) when present.
        edsr_idx = self.combo_upscale_method.findData("EDSR")
        if edsr_idx >= 0:
            self.combo_upscale_method.setCurrentIndex(edsr_idx)

        self.combo_upscale = QComboBox()
        self.combo_upscale_method.currentIndexChanged.connect(self._refresh_upscale_scales)

        self.chk_upscale_denoise = QCheckBox("Denoise first")
        self.chk_upscale_sharpen = QCheckBox("Sharpen result")
        opt_row = QHBoxLayout()
        opt_row.addWidget(self.chk_upscale_denoise)
        opt_row.addWidget(self.chk_upscale_sharpen)
        self.btn_upscale = QPushButton("Upscale")
        self.btn_upscale.setObjectName("primaryButton")
        el.addRow("Model:", self.combo_upscale_method)
        el.addRow("Scale:", self.combo_upscale)
        el.addRow(opt_row)
        el.addRow(self.btn_upscale)
        self._refresh_upscale_scales()
        lay.addWidget(enhance_group)

        models_group = QGroupBox("AI Models")
        mg = QVBoxLayout(models_group)
        self.btn_manage_models = QPushButton("Download / Manage Models…")
        self.btn_open_models = QPushButton("Open Models Folder")
        mg.addWidget(self.btn_manage_models)
        mg.addWidget(self.btn_open_models)
        models_path = QLabel(f"Folder: {MODELS_DIR}")
        models_path.setWordWrap(True)
        models_path.setStyleSheet("color:#889; font-size:10px;")
        mg.addWidget(models_path)
        lay.addWidget(models_group)

        hint = QLabel("Tip: pick a model, then refine edges with Alpha Matting "
                      "or the Manual Edit tools. Models download on first use, or "
                      "grab them ahead of time above.")
        hint.setWordWrap(True)
        hint.setStyleSheet("color:#667; font-size:11px;")
        lay.addWidget(hint)
        lay.addStretch()
        return w

    def _build_manual_tab(self):
        w = QWidget()
        lay = QVBoxLayout(w)

        tools_group = QGroupBox("Drawing & Crop")
        tl = QVBoxLayout(tools_group)
        self.btn_mode_crop = QPushButton("Select Crop Area"); self.btn_mode_crop.setCheckable(True)
        self.btn_apply_crop = QPushButton("Apply Crop")
        crop_layout = QHBoxLayout(); crop_layout.addWidget(self.btn_mode_crop, 1); crop_layout.addWidget(self.btn_apply_crop)
        tl.addLayout(crop_layout)
        self.btn_mode_keep = QPushButton("Mark Keep"); self.btn_mode_keep.setCheckable(True)
        self.btn_mode_remove = QPushButton("Mark Remove"); self.btn_mode_remove.setCheckable(True)
        mask_mode_layout = QHBoxLayout(); mask_mode_layout.addWidget(self.btn_mode_keep); mask_mode_layout.addWidget(self.btn_mode_remove)
        tl.addLayout(mask_mode_layout)
        self.brush_slider = QSlider(Qt.Orientation.Horizontal); self.brush_slider.setRange(1, 200); self.brush_slider.setValue(self.brush_size)
        self.brush_size_label_value = QLabel(f"{self.brush_size}px")
        brush_layout = QHBoxLayout(); brush_layout.addWidget(QLabel("Brush:")); brush_layout.addWidget(self.brush_slider); brush_layout.addWidget(self.brush_size_label_value)
        tl.addLayout(brush_layout)
        self.btn_apply_mask = QPushButton("Apply Keep/Remove Marks")
        tl.addWidget(self.btn_apply_mask)
        lay.addWidget(tools_group)

        magic_wand_group = QGroupBox("Magic Wand")
        ml = QVBoxLayout(magic_wand_group)
        self.btn_magic_wand = QPushButton("Magic Wand Select"); self.btn_magic_wand.setCheckable(True)
        if not SCIPY_AVAILABLE:
            self.btn_magic_wand.setToolTip("Disabled: 'scipy' not found (pip install scipy)")
        self.wand_tolerance_spin = QSpinBox(); self.wand_tolerance_spin.setRange(0, 255); self.wand_tolerance_spin.setValue(20)
        wand_tolerance_layout = QHBoxLayout(); wand_tolerance_layout.addWidget(QLabel("Tolerance:")); wand_tolerance_layout.addWidget(self.wand_tolerance_spin)
        self.btn_apply_wand_remove = QPushButton("Remove Selected Area")
        self.btn_apply_wand_keep = QPushButton("Keep Selected (Remove BG)")
        ml.addWidget(self.btn_magic_wand)
        ml.addLayout(wand_tolerance_layout)
        ml.addWidget(self.btn_apply_wand_remove)
        ml.addWidget(self.btn_apply_wand_keep)
        lay.addWidget(magic_wand_group)

        other_tools_group = QGroupBox("Color & Background")
        ol = QVBoxLayout(other_tools_group)
        self.btn_select_color = QPushButton("Pick Color to Remove")
        self.color_preview = QLabel(" None"); self.color_preview.setMinimumWidth(60)
        self.color_preview.setStyleSheet("border:1px solid #bbb; background:#eee; padding:5px; border-radius:4px;")
        color_select_layout = QHBoxLayout(); color_select_layout.addWidget(self.btn_select_color, 1); color_select_layout.addWidget(self.color_preview)
        ol.addLayout(color_select_layout)
        self.tolerance_spin = QSpinBox(); self.tolerance_spin.setRange(0, 255); self.tolerance_spin.setValue(30)
        tolerance_layout = QHBoxLayout(); tolerance_layout.addWidget(QLabel("Tolerance:")); tolerance_layout.addWidget(self.tolerance_spin)
        ol.addLayout(tolerance_layout)
        self.btn_apply_color_remove = QPushButton("Remove Selected Color")
        ol.addWidget(self.btn_apply_color_remove)
        ol.addSpacing(6)
        self.btn_fill_bg = QPushButton("Fill Background")
        self.btn_remove_fill = QPushButton("Remove Fill")
        fill_layout = QHBoxLayout(); fill_layout.addWidget(self.btn_fill_bg); fill_layout.addWidget(self.btn_remove_fill)
        ol.addLayout(fill_layout)
        ol.addSpacing(6)
        self.btn_show_original = QPushButton("Show Original (Hold)")
        ol.addWidget(self.btn_show_original)
        lay.addWidget(other_tools_group)
        lay.addStretch()

        return w

    def _build_correct_tab(self):
        w = QWidget()
        lay = QVBoxLayout(w)

        # --- Rotation & alignment grid ---
        rot_group = QGroupBox("Rotate & Straighten")
        rl = QVBoxLayout(rot_group)

        angle_row = QHBoxLayout()
        self.angle_slider = QSlider(Qt.Orientation.Horizontal)
        self.angle_slider.setRange(-180, 180); self.angle_slider.setValue(0)
        self.angle_spin = QDoubleSpinBox()
        self.angle_spin.setRange(-180.0, 180.0); self.angle_spin.setSingleStep(0.1)
        self.angle_spin.setDecimals(1); self.angle_spin.setSuffix(" °")
        angle_row.addWidget(QLabel("Angle:"))
        angle_row.addWidget(self.angle_slider, 1)
        angle_row.addWidget(self.angle_spin)
        rl.addLayout(angle_row)

        self.chk_grid = QCheckBox("Show alignment grid")
        grid_row = QHBoxLayout()
        grid_row.addWidget(self.chk_grid)
        grid_row.addWidget(QLabel("Spacing:"))
        self.grid_spin = QSpinBox(); self.grid_spin.setRange(10, 500); self.grid_spin.setValue(50); self.grid_spin.setSuffix(" px")
        grid_row.addWidget(self.grid_spin)
        rl.addLayout(grid_row)

        rot_btn_row = QHBoxLayout()
        self.btn_angle_reset = QPushButton("Reset Angle")
        self.btn_apply_rotation = QPushButton("Apply Rotation")
        self.btn_apply_rotation.setObjectName("primaryButton")
        rot_btn_row.addWidget(self.btn_angle_reset)
        rot_btn_row.addWidget(self.btn_apply_rotation)
        rl.addLayout(rot_btn_row)
        lay.addWidget(rot_group)

        persp_group = QGroupBox("Perspective Correction")
        pl = QVBoxLayout(persp_group)

        mode_layout = QHBoxLayout()
        self.radio_4pt = QRadioButton("4-point"); self.radio_4pt.setChecked(True)
        self.radio_6pt = QRadioButton("6-point (curvature)")
        mode_layout.addWidget(self.radio_4pt); mode_layout.addWidget(self.radio_6pt)
        pl.addLayout(mode_layout)

        self.btn_persp_place = QPushButton("Place Points"); self.btn_persp_place.setCheckable(True)
        pl.addWidget(self.btn_persp_place)

        self.persp_hint = QLabel()
        self.persp_hint.setWordWrap(True)
        self.persp_hint.setStyleSheet("color:#557; font-size:11px; padding:2px;")
        pl.addWidget(self.persp_hint)

        btn_row = QHBoxLayout()
        self.btn_persp_clear = QPushButton("Clear Points")
        self.btn_persp_apply = QPushButton("Apply Correction")
        self.btn_persp_apply.setObjectName("primaryButton")
        btn_row.addWidget(self.btn_persp_clear); btn_row.addWidget(self.btn_persp_apply)
        pl.addLayout(btn_row)

        if not CV2_AVAILABLE:
            persp_group.setEnabled(False)
            persp_group.setToolTip("Disabled: OpenCV (cv2) not found (pip install opencv-python)")
        lay.addWidget(persp_group)

        guide = QLabel(
            "<b>4-point:</b> click the four corners in order — "
            "Top-Left → Top-Right → Bottom-Right → Bottom-Left.<br><br>"
            "<b>6-point:</b> add two more points at the Top-Middle and "
            "Bottom-Middle edges to straighten curved/warped images "
            "(e.g. smiling gel lanes).<br><br>"
            "Drag any placed point to fine-tune before applying.")
        guide.setWordWrap(True)
        guide.setStyleSheet("color:#556; font-size:11px;")
        lay.addWidget(guide)
        lay.addStretch()
        self._update_perspective_hint()

        return w

    def _build_compose_tab(self):
        w = QWidget()
        lay = QVBoxLayout(w)

        overlay_group = QGroupBox("Overlay / Overlap Images")
        og = QVBoxLayout(overlay_group)
        self.btn_open_compositor = QPushButton("  Open Compositor…")
        self.btn_open_compositor.setObjectName("primaryButton")
        og.addWidget(self.btn_open_compositor)
        og.addWidget(QLabel(
            "Add multiple images, drag to move & overlap them, resize with corner "
            "handles, and set each image's Overlap (opacity) and blend mode — "
            "then merge them together."))
        og.itemAt(1).widget().setWordWrap(True)
        og.itemAt(1).widget().setStyleSheet("color:#667; font-size:11px;")
        lay.addWidget(overlay_group)

        page_group = QGroupBox("Page Background")
        pg = QVBoxLayout(page_group)
        self.btn_open_pagebg = QPushButton("Set Page Background (A4/A5/Letter…)")
        pg.addWidget(self.btn_open_pagebg)
        pg.addWidget(QLabel("Place the current image onto a standard page canvas."))
        pg.itemAt(1).widget().setWordWrap(True)
        pg.itemAt(1).widget().setStyleSheet("color:#667; font-size:11px;")
        lay.addWidget(page_group)

        export_group = QGroupBox("Export")
        eg = QVBoxLayout(export_group)
        self.btn_export_pdf2 = QPushButton("Export as PDF…")
        self.btn_export_svg = QPushButton("Export as SVG (vector)…")
        eg.addWidget(self.btn_export_pdf2)
        eg.addWidget(self.btn_export_svg)
        eg.addWidget(QLabel("SVG wraps the image in a scalable container, or "
                            "traces the cut-out edge into true vector paths."))
        eg.itemAt(2).widget().setWordWrap(True)
        eg.itemAt(2).widget().setStyleSheet("color:#667; font-size:11px;")
        lay.addWidget(export_group)

        lay.addStretch()
        self.btn_open_compositor.clicked.connect(self.open_overlay_dialog)
        self.btn_open_pagebg.clicked.connect(self.open_page_background_dialog)
        self.btn_export_pdf2.clicked.connect(self.export_pdf)
        self.btn_export_svg.clicked.connect(self.export_svg)
        return w

    def _create_status_bar(self):
        self.statusBar = QStatusBar()
        self.setStatusBar(self.statusBar)
        self.statusBar.showMessage("Ready. Load an image, paste from clipboard, or drag & drop a file.")

    # ---- signals ----
    def _connect_signals(self):
        self.action_open.triggered.connect(self.open_image)
        self.action_paste.triggered.connect(self.paste_image)
        self.action_save.triggered.connect(self.save_image)
        self.action_export_pdf.triggered.connect(self.export_pdf)
        self.action_copy.triggered.connect(self.copy_to_clipboard)
        self.action_overlay.triggered.connect(self.open_overlay_dialog)
        self.action_page_bg.triggered.connect(self.open_page_background_dialog)
        self.action_quit.triggered.connect(self.close)
        self.action_undo.triggered.connect(self.undo_state)
        self.action_redo.triggered.connect(self.redo_state)
        self.action_reset.triggered.connect(self.reset_image)
        self.action_zoom_in.triggered.connect(lambda: self.image_label_preview.set_zoom(self.image_label_preview.zoom_level * 1.25))
        self.action_zoom_out.triggered.connect(lambda: self.image_label_preview.set_zoom(self.image_label_preview.zoom_level * 0.8))
        self.action_zoom_reset.triggered.connect(self.image_label_preview.fit_to_view)
        self.action_info.triggered.connect(self.show_about)
        self.action_batch.triggered.connect(self.open_batch_dialog)
        self.action_prefs.triggered.connect(self.open_preferences)
        self.action_theme.toggled.connect(self.toggle_theme)
        self.action_shortcuts.triggered.connect(self.show_shortcuts)
        self.action_update.triggered.connect(lambda: self.check_for_updates(silent=False))
        self.model_combo.currentTextChanged.connect(
            lambda m: self.settings.setValue("rembg_model", m))

        self.btn_rembg.clicked.connect(self.run_rembg)
        self.btn_sharpen.clicked.connect(self.apply_sharpen)
        self.btn_upscale.clicked.connect(self.apply_upscale)
        self.btn_manage_models.clicked.connect(self.open_model_manager)
        self.btn_open_models.clicked.connect(
            lambda: QDesktopServices.openUrl(QUrl.fromLocalFile(MODELS_DIR)))
        self.model_combo.currentTextChanged.connect(lambda m: setattr(self, 'rembg_model', m))
        self.cb_alpha_matting.stateChanged.connect(lambda s: (setattr(self, 'alpha_matting_enabled', bool(s)), self._update_ui_states()))
        self.btn_mode_crop.clicked.connect(lambda: self.set_interaction_mode(InteractiveLabel.MODE_CROP))
        self.btn_apply_crop.clicked.connect(self.apply_crop)
        self.btn_mode_keep.clicked.connect(lambda: self.set_interaction_mode(InteractiveLabel.MODE_KEEP))
        self.btn_mode_remove.clicked.connect(lambda: self.set_interaction_mode(InteractiveLabel.MODE_REMOVE))
        self.brush_slider.valueChanged.connect(self._update_brush_size)
        self.btn_apply_mask.clicked.connect(self.apply_mask_refinement)

        self.btn_magic_wand.clicked.connect(lambda: self.set_interaction_mode(InteractiveLabel.MODE_WAND))
        self.image_label_preview.wand_point_selected.connect(self.calculate_wand_selection)
        self.btn_apply_wand_remove.clicked.connect(self.apply_wand_remove)
        self.btn_apply_wand_keep.clicked.connect(self.apply_wand_keep)

        self.btn_select_color.clicked.connect(self.select_color_to_remove)
        self.btn_apply_color_remove.clicked.connect(self.apply_color_removal)
        self.btn_fill_bg.clicked.connect(self.fill_background)
        self.btn_remove_fill.clicked.connect(self.remove_background_color)
        self.btn_show_original.pressed.connect(self._show_original_image_fast)
        self.btn_show_original.released.connect(self._show_current_image_display)

        # Rotation & grid
        self.angle_slider.valueChanged.connect(lambda v: self._set_angle(float(v)))
        self.angle_spin.valueChanged.connect(self._set_angle)
        self.btn_angle_reset.clicked.connect(self._reset_angle_controls)
        self.btn_apply_rotation.clicked.connect(self.apply_rotation)
        self.chk_grid.toggled.connect(lambda on: self.image_label_preview.set_grid(on, self.grid_spin.value()))
        self.grid_spin.valueChanged.connect(
            lambda v: self.chk_grid.isChecked() and self.image_label_preview.set_grid(True, v))

        # Perspective
        self.radio_4pt.toggled.connect(self._on_perspective_mode_changed)
        self.btn_persp_place.clicked.connect(lambda: self.set_interaction_mode(InteractiveLabel.MODE_PERSPECTIVE))
        self.btn_persp_clear.clicked.connect(self.image_label_preview.clear_perspective)
        self.btn_persp_apply.clicked.connect(self.apply_perspective)
        self.image_label_preview.perspective_changed.connect(self._update_perspective_hint)
        self.image_label_preview.perspective_changed.connect(self._update_ui_states)

        self.mode_buttons = {
            InteractiveLabel.MODE_KEEP: self.btn_mode_keep,
            InteractiveLabel.MODE_REMOVE: self.btn_mode_remove,
            InteractiveLabel.MODE_CROP: self.btn_mode_crop,
            InteractiveLabel.MODE_WAND: self.btn_magic_wand,
            InteractiveLabel.MODE_PERSPECTIVE: self.btn_persp_place,
        }

    # ---- display helpers ----
    def _show_original_image_fast(self):
        if self.original_qpixmap:
            self.image_label_preview.set_display_pixmap(self.original_qpixmap)

    def _show_current_image_display(self):
        self._update_display()

    def _update_brush_size(self, value):
        self.brush_size = value
        self.image_label_preview.set_brush_size(self.brush_size)
        self.brush_size_label_value.setText(f"{value}px")

    def _push_state(self, pil_image, description=""):
        if pil_image is None:
            return
        if len(self.undo_stack) >= self.MAX_HISTORY:
            self.undo_stack.pop(0)
        self.undo_stack.append({"image": pil_image.copy(), "desc": description, "bg_color": self.background_color})
        self.redo_stack.clear()
        self._update_ui_states()

    def _load_new_image(self, pil_image, source_desc="Loaded"):
        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        try:
            self.original_pil_image = pil_image.convert('RGBA')
            self.current_pil_image = self.original_pil_image.copy()
            self.original_qpixmap = pil_to_qpixmap(self.original_pil_image)
            self.current_qpixmap = self.original_qpixmap.copy()
            self.background_color = None
            self.undo_stack, self.redo_stack = [], []
            self.image_label_preview.clear_interaction_state()
            self._reset_angle_controls(refresh=False)
            self._push_state(self.current_pil_image, "Initial Load")
            self._update_display()
            self.image_label_preview.fit_to_view()
            self.statusBar.showMessage(f"{source_desc} successfully.", 5000)
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to process loaded image: {e}")
            self._clear_workspace()
        finally:
            QApplication.restoreOverrideCursor()
            self._update_ui_states()
            self.set_interaction_mode(InteractiveLabel.MODE_NONE, force_off=True)

    def _update_display(self):
        if not self.current_qpixmap:
            self.image_label_preview.set_display_pixmap(QPixmap())
            return

        # Apply live (unbaked) rotation preview if any
        content = self.current_qpixmap
        if abs(self.preview_angle) > 1e-6:
            transform = QTransform().rotate(self.preview_angle)
            content = self.current_qpixmap.transformed(
                transform, Qt.TransformationMode.SmoothTransformation)

        base = QPixmap(content.size())
        base.fill(self.background_color if self.background_color else Qt.GlobalColor.transparent)
        painter = QPainter(base)
        if not self.background_color:
            checkerboard = create_checkerboard(base.width(), base.height())
            painter.drawPixmap(0, 0, checkerboard)
        painter.drawPixmap(0, 0, content)
        painter.end()
        self.image_label_preview.set_display_pixmap(base)

    def _clear_workspace(self):
        self.original_pil_image, self.current_pil_image = None, None
        self.original_qpixmap, self.current_qpixmap = None, None
        self.background_color = None
        self.undo_stack, self.redo_stack = [], []
        self._update_display()
        self.statusBar.showMessage("Workspace cleared. Load or paste an image.")
        self._update_ui_states()

    def _set_current_image(self, pil_image, description):
        self._push_state(self.current_pil_image, description)
        self.current_pil_image = pil_image
        self.current_qpixmap = pil_to_qpixmap(self.current_pil_image)
        self._update_display()

    def undo_state(self):
        if len(self.undo_stack) <= 1:
            return
        self.redo_stack.append(self.undo_stack.pop())
        last_state = self.undo_stack[-1]
        self.current_pil_image = last_state["image"]
        self.current_qpixmap = pil_to_qpixmap(self.current_pil_image)
        self.background_color = last_state["bg_color"]
        self.image_label_preview.clear_interaction_state()
        self._reset_angle_controls(refresh=False)
        self._update_display()
        self.statusBar.showMessage(f"Undo: Restored '{last_state['desc']}'", 3000)
        self._update_ui_states()

    def redo_state(self):
        if not self.redo_stack:
            return
        next_state = self.redo_stack.pop()
        self.undo_stack.append(next_state)
        self.current_pil_image = next_state["image"]
        self.current_qpixmap = pil_to_qpixmap(self.current_pil_image)
        self.background_color = next_state["bg_color"]
        self.image_label_preview.clear_interaction_state()
        self._reset_angle_controls(refresh=False)
        self._update_display()
        self.statusBar.showMessage("Redo: Restored state", 3000)
        self._update_ui_states()

    def _perform_operation(self, operation_func, pre_op_desc, progress_title="Processing…"):
        if not self.current_pil_image:
            return
        progress = QProgressDialog(progress_title, None, 0, 0, self)
        progress.setWindowModality(Qt.WindowModality.WindowModal)
        progress.setCancelButton(None)
        progress.show()
        QApplication.processEvents()
        try:
            result_pil = operation_func(self.current_pil_image.copy())
            self._set_current_image(result_pil, pre_op_desc.replace("Before ", ""))
            self.statusBar.showMessage(f"{pre_op_desc.replace('Before ', '')} applied.", 5000)
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Operation failed: {e}")
        finally:
            progress.close()
            self.image_label_preview.clear_interaction_state()
            self._update_ui_states()
            self.set_interaction_mode(InteractiveLabel.MODE_NONE, force_off=True)

    def _update_ui_states(self):
        has_image = self.current_pil_image is not None
        is_rgba = has_image and self.current_pil_image.mode == 'RGBA'

        magic_wand_enabled = has_image and SCIPY_AVAILABLE
        self.btn_magic_wand.setEnabled(magic_wand_enabled)

        main_actions = [self.action_save, self.action_export_pdf, self.action_copy,
                        self.action_reset, self.action_overlay, self.action_page_bg, self.btn_rembg,
                        self.btn_mode_crop, self.btn_apply_crop, self.btn_mode_keep,
                        self.btn_mode_remove, self.btn_select_color, self.btn_apply_color_remove,
                        self.btn_apply_mask, self.btn_show_original,
                        self.angle_slider, self.angle_spin, self.btn_apply_rotation,
                        self.btn_angle_reset, self.chk_grid, self.grid_spin,
                        self.btn_open_compositor, self.btn_open_pagebg, self.btn_export_pdf2,
                        self.btn_export_svg, self.btn_sharpen, self.btn_upscale,
                        self.slider_sharpen, self.combo_upscale, self.combo_upscale_method]
        for action in main_actions:
            action.setEnabled(has_image)

        if not REMBG_AVAILABLE:
            self.btn_rembg.setEnabled(False)
            self.btn_rembg.setText("  rembg not installed")

        self.btn_fill_bg.setEnabled(is_rgba)
        self.btn_remove_fill.setEnabled(is_rgba and self.background_color is not None)

        has_wand_selection = magic_wand_enabled and self.image_label_preview.wand_selection_mask is not None
        self.btn_apply_wand_remove.setEnabled(has_wand_selection)
        self.btn_apply_wand_keep.setEnabled(has_wand_selection)

        self.btn_apply_color_remove.setEnabled(has_image and self.selected_color_rgb is not None)

        # Perspective buttons
        persp_ok = has_image and CV2_AVAILABLE
        self.btn_persp_place.setEnabled(persp_ok)
        self.btn_persp_clear.setEnabled(persp_ok and bool(self.image_label_preview.perspective_points))
        pts = self.image_label_preview.perspective_points
        self.btn_persp_apply.setEnabled(persp_ok and len(pts) == self.image_label_preview.perspective_required)

        self.action_undo.setEnabled(len(self.undo_stack) > 1)
        self.action_redo.setEnabled(bool(self.redo_stack))
        clip = QApplication.clipboard().mimeData()
        self.action_paste.setEnabled(bool(clip) and clip.hasImage())

        is_matting = self.cb_alpha_matting.isChecked()
        for spin in (self.spin_fg_thresh, self.spin_bg_thresh, self.spin_erode_size):
            spin.setEnabled(is_matting)

    def set_interaction_mode(self, mode_to_set, force_off=False):
        active_button = self.mode_buttons.get(mode_to_set)
        if force_off:
            current_mode = InteractiveLabel.MODE_NONE
            for btn in self.mode_buttons.values():
                btn.setChecked(False)
        else:
            if not active_button:
                return
            is_now_checked = active_button.isChecked()
            current_mode = mode_to_set if is_now_checked else InteractiveLabel.MODE_NONE
            for mode, btn in self.mode_buttons.items():
                if mode != mode_to_set:
                    btn.setChecked(False)
        self.image_label_preview.set_mode(current_mode)

    # ---- file operations ----
    def open_image(self, path=None):
        if not path:
            path, _ = QFileDialog.getOpenFileName(self, "Open Image", "",
                "Images (*.png *.jpg *.jpeg *.bmp *.webp *.tiff *.gif *.avif)")
        if path:
            try:
                self._load_new_image(Image.open(path), f"Loaded '{os.path.basename(path)}'")
                self._add_recent_file(path)
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to load image file: {e}")

    # ---- recent files ----
    def _add_recent_file(self, path):
        path = os.path.abspath(path)
        recent = [p for p in self.recent_files if p != path]
        recent.insert(0, path)
        self.recent_files = recent[:10]
        self.settings.setValue("recent_files", self.recent_files)
        self._rebuild_recent_menu()

    def _rebuild_recent_menu(self):
        if not hasattr(self, "menu_recent"):
            return
        self.menu_recent.clear()
        existing = [p for p in self.recent_files if os.path.exists(p)]
        if not existing:
            act = self.menu_recent.addAction("(no recent files)")
            act.setEnabled(False)
            return
        for p in existing:
            act = self.menu_recent.addAction(os.path.basename(p))
            act.setToolTip(p)
            act.triggered.connect(lambda checked=False, path=p: self.open_image(path))
        self.menu_recent.addSeparator()
        clear = self.menu_recent.addAction("Clear Recent")
        clear.triggered.connect(self._clear_recent_files)

    def _clear_recent_files(self):
        self.recent_files = []
        self.settings.setValue("recent_files", [])
        self._rebuild_recent_menu()

    def paste_image(self):
        qimage = QApplication.clipboard().image()
        if not qimage.isNull():
            self._load_new_image(qimage_to_pil(qimage), "Pasted from clipboard")
        else:
            QMessageBox.information(self, "Paste", "No valid image found on the clipboard.")

    def reset_image(self):
        if self.original_pil_image:
            self.background_color = None
            self._reset_angle_controls(refresh=False)
            self._set_current_image(self.original_pil_image.copy(), "Reset")
            self.image_label_preview.fit_to_view()
            self.statusBar.showMessage("Image reset to original.", 3000)
            self._update_ui_states()

    def save_image(self):
        if not self.current_pil_image:
            return
        default_fmt = self.settings.value("default_format", "png", type=str)
        default_name = f"processed_image.{default_fmt}"
        path, _ = QFileDialog.getSaveFileName(self, "Save Image As…", default_name,
            "PNG (*.png);;JPG (*.jpg);;TIFF (*.tiff);;WebP (*.webp);;AVIF (*.avif)")
        if not path:
            return
        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        try:
            image_to_save = self.current_pil_image.copy()
            if self.background_color:
                image_to_save = flatten_image(image_to_save, self.background_color.getRgb()[:3])
            lower = path.lower()
            # Formats without alpha need a flattened (opaque) image.
            if lower.endswith(('.jpg', '.jpeg')) and image_to_save.mode == 'RGBA':
                image_to_save = flatten_image(image_to_save, (255, 255, 255))
            save_kwargs = {}
            quality = self.settings.value("save_quality", 92, type=int)
            if lower.endswith(('.jpg', '.jpeg', '.webp', '.avif')):
                save_kwargs["quality"] = quality
            try:
                image_to_save.save(path, **save_kwargs)
            except (KeyError, OSError) as fmt_err:
                if lower.endswith('.avif'):
                    QMessageBox.warning(self, "AVIF unavailable",
                        "Saving AVIF needs the 'pillow-avif-plugin' package.\n"
                        "Install it with:  pip install pillow-avif-plugin\n\n"
                        f"({fmt_err})")
                    return
                raise
            self.statusBar.showMessage(f"Image saved to {path}", 5000)
        except Exception as e:
            QMessageBox.critical(self, "Save Error", f"Failed to save image: {e}")
        finally:
            QApplication.restoreOverrideCursor()

    def export_pdf(self):
        if not self.current_pil_image:
            return
        dialog = PdfExportDialog(self)
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return
        path, _ = QFileDialog.getSaveFileName(self, "Export as PDF", "document.pdf", "PDF (*.pdf)")
        if not path:
            return
        if not path.lower().endswith(".pdf"):
            path += ".pdf"
        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        try:
            bg = self.background_color.getRgb()[:3] if self.background_color else (255, 255, 255)
            image_rgb = flatten_image(self.current_pil_image, bg)
            export_to_pdf(path, image_rgb, dialog.options())
            self.statusBar.showMessage(f"PDF exported to {path}", 5000)
        except Exception as e:
            QMessageBox.critical(self, "PDF Export Error", f"Failed to export PDF: {e}")
        finally:
            QApplication.restoreOverrideCursor()

    def export_svg(self):
        if not self.current_pil_image:
            return
        dialog = SvgExportDialog(self)
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return
        path, _ = QFileDialog.getSaveFileName(self, "Export as SVG", "image.svg", "SVG (*.svg)")
        if not path:
            return
        if not path.lower().endswith(".svg"):
            path += ".svg"
        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        try:
            image = self.current_pil_image.copy()
            if self.background_color:
                image = flatten_image(image, self.background_color.getRgb()[:3])
            export_to_svg(path, image, dialog.options())
            self.statusBar.showMessage(f"SVG exported to {path}", 5000)
        except Exception as e:
            QMessageBox.critical(self, "SVG Export Error", f"Failed to export SVG: {e}")
        finally:
            QApplication.restoreOverrideCursor()

    def copy_to_clipboard(self):
        if not self.current_pil_image:
            return
        temp_path = None
        try:
            image_to_copy = self.current_pil_image.copy()
            if self.background_color:
                image_to_copy = flatten_image(image_to_copy, self.background_color.getRgb()[:3])
            with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as temp_f:
                temp_path = temp_f.name
                image_to_copy.save(temp_path, "PNG")
            self.temp_files_to_clean.append(temp_path)
            mime_data = QMimeData()
            mime_data.setUrls([QUrl.fromLocalFile(temp_path)])
            QApplication.clipboard().setMimeData(mime_data)
            self.statusBar.showMessage("Image copied to clipboard (preserves transparency).", 4000)
        except Exception as e:
            QMessageBox.critical(self, "Copy Error", f"Failed to copy image: {e}")
            if temp_path and os.path.exists(temp_path):
                try:
                    os.remove(temp_path)
                    if temp_path in self.temp_files_to_clean:
                        self.temp_files_to_clean.remove(temp_path)
                except OSError as cleanup_error:
                    print(f"Failed to clean up temporary file: {cleanup_error}")

    def open_overlay_dialog(self, preload=None):
        if not self.current_pil_image:
            return
        dialog = OverlayDialog(self.current_pil_image, self, preload=preload)
        if dialog.exec() == QDialog.DialogCode.Accepted and dialog.result_pil is not None:
            self._set_current_image(dialog.result_pil, "Overlay")
            self.image_label_preview.fit_to_view()
            self.statusBar.showMessage("Images composed & merged.", 5000)
            self._update_ui_states()

    def open_page_background_dialog(self):
        if not self.current_pil_image:
            return
        dialog = PageBackgroundDialog(self)
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return
        opts = dialog.options()
        # Flatten any non-destructive fill into the image first
        source = self.current_pil_image
        if self.background_color:
            source = flatten_image(source, self.background_color.getRgb()[:3]).convert('RGBA')
        def operation(_img):
            result = create_page_background(source, opts)
            QTimer.singleShot(0, self.image_label_preview.fit_to_view)
            return result
        self.background_color = None
        self._perform_operation(operation, "Before Page Background", "Building Page…")

    # ---- rotation ----
    def _set_angle(self, angle):
        self.preview_angle = float(angle)
        self.angle_slider.blockSignals(True)
        self.angle_slider.setValue(int(round(angle)))
        self.angle_slider.blockSignals(False)
        self.angle_spin.blockSignals(True)
        self.angle_spin.setValue(angle)
        self.angle_spin.blockSignals(False)
        if self.current_qpixmap:
            self._update_display()

    def _reset_angle_controls(self, refresh=True):
        self.preview_angle = 0.0
        for w in (self.angle_slider, self.angle_spin):
            w.blockSignals(True)
        self.angle_slider.setValue(0)
        self.angle_spin.setValue(0.0)
        for w in (self.angle_slider, self.angle_spin):
            w.blockSignals(False)
        if refresh and self.current_qpixmap:
            self._update_display()

    def apply_rotation(self):
        angle = self.preview_angle
        if abs(angle) < 0.05:
            self.statusBar.showMessage("Set a rotation angle first.", 3000)
            return
        self._reset_angle_controls()   # clears preview so baked result isn't double-rotated
        def operation(img):
            rotated = img.rotate(-angle, expand=True, resample=Image.BICUBIC)
            QTimer.singleShot(0, self.image_label_preview.fit_to_view)
            return rotated
        self._perform_operation(operation, "Before Rotation", "Rotating…")

    # ---- drag & drop ----
    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls() and any(url.isLocalFile() for url in event.mimeData().urls()):
            event.acceptProposedAction()

    def dropEvent(self, event):
        paths = [u.toLocalFile() for u in event.mimeData().urls() if u.isLocalFile()]
        paths = [p for p in paths if p]
        if not paths:
            return
        # First image loads as the working image; any extras are composited as
        # layers via the Overlay tool so a multi-file drop builds up a scene.
        self.open_image(paths[0])
        extras = paths[1:]
        if extras and self.current_pil_image is not None:
            try:
                self.open_overlay_dialog(preload=extras)
            except Exception:
                self.statusBar.showMessage(
                    f"Loaded first image; {len(extras)} more ready to add via Compose.", 5000)

    # ---- AI removal ----
    def _run_async_operation(self, operation_func, description, title):
        """Run a heavy operation on a worker thread with a live progress dialog
        and a working Cancel button that always returns control to the user."""
        if not self.current_pil_image:
            return
        progress = QProgressDialog(title, "Cancel", 0, 0, self)
        progress.setWindowTitle("Working…")
        progress.setWindowModality(Qt.WindowModality.WindowModal)
        progress.setMinimumDuration(0)
        progress.setAutoClose(False)
        progress.setAutoReset(False)
        progress.setMinimumWidth(380)

        worker = OperationWorker(operation_func, self.current_pil_image.copy())
        self._active_workers.append(worker)
        # Keep the thread object alive until it truly finishes, even if cancelled.
        worker.finished.connect(
            lambda w=worker: w in self._active_workers and self._active_workers.remove(w))

        state = {"finished": False, "stage": title, "start": time.monotonic()}

        timer = QTimer(self)

        def _tick():
            elapsed = int(time.monotonic() - state["start"])
            progress.setLabelText(
                f"{state['stage']}\n\nElapsed: {elapsed}s\n"
                "Click Cancel to stop and regain control.")
        timer.timeout.connect(_tick)
        timer.start(400)
        _tick()

        def _finish():
            if state["finished"]:
                return
            state["finished"] = True
            timer.stop()
            progress.close()
            self.image_label_preview.clear_interaction_state()
            self._update_ui_states()
            self.set_interaction_mode(InteractiveLabel.MODE_NONE, force_off=True)

        def _on_progress(msg):
            state["stage"] = msg

        def _on_ok(result):
            if state["finished"]:
                return
            self._set_current_image(result, description.replace("Before ", ""))
            self.statusBar.showMessage(
                f"{description.replace('Before ', '')} applied.", 5000)
            _finish()

        def _on_failed(msg):
            if state["finished"]:
                return
            _finish()
            QMessageBox.critical(self, "Error", f"Operation failed: {msg}")

        def _on_cancel():
            if state["finished"]:
                return
            worker.cancel()
            _finish()
            self.statusBar.showMessage(
                "Cancelled. A running AI task may finish in the background and be discarded.",
                6000)

        worker.progress.connect(_on_progress)
        worker.finished_ok.connect(_on_ok)
        worker.failed.connect(_on_failed)
        progress.canceled.connect(_on_cancel)

        worker.start()
        progress.show()

    def get_rembg_session(self, model):
        """Return a cached rembg session for `model`, built with the preferred
        execution providers (GPU/CoreML when enabled and available). Falls back
        to letting remove_bg build its own session if this fails."""
        if model in self._rembg_sessions:
            return self._rembg_sessions[model]
        try:
            from rembg import new_session
            use_gpu = self.settings.value("gpu_acceleration", True, type=bool)
            providers = preferred_ort_providers(use_gpu)
            session = (new_session(model, providers=providers)
                       if providers else new_session(model))
            self._rembg_sessions[model] = session
            return session
        except Exception as e:
            print(f"Could not create accelerated rembg session ({e}); using default.")
            return None

    def run_rembg(self):
        if not REMBG_AVAILABLE:
            return
        model = self.rembg_model
        matting = self.alpha_matting_enabled
        fg, bg, er = (self.spin_fg_thresh.value(), self.spin_bg_thresh.value(),
                      self.spin_erode_size.value())

        def operation(img, report):
            report("Preparing AI model…")
            if not model_is_downloaded(model):
                report(f"Downloading “{model}” model (first use)…")
            report("Running AI background removal…")
            session = self.get_rembg_session(model)
            kwargs = dict(alpha_matting=matting,
                          alpha_matting_foreground_threshold=fg,
                          alpha_matting_background_threshold=bg,
                          alpha_matting_erode_size=er)
            if session is not None:
                return remove_bg(img, session=session, **kwargs)
            return remove_bg(img, model=model, **kwargs)
        self._run_async_operation(operation, "Before Background Removal", "Removing Background…")

    def apply_sharpen(self):
        if not self.current_pil_image:
            return
        amount = self.slider_sharpen.value()
        radius = self.spin_sharpen_radius.value()

        def operation(img, report):
            return sharpen_image(img, amount=amount, radius=radius, report=report)
        self._run_async_operation(operation, "Before Sharpen", "Sharpening…")

    def _refresh_upscale_scales(self):
        """Repopulate the Scale combo with the factors the chosen model supports.
        Real-ESRGAN is fixed-factor; dnn_superres models list their own scales;
        Lanczos (data None) offers a broad range since it needs no model file."""
        if not hasattr(self, "combo_upscale"):
            return
        model = self.combo_upscale_method.currentData()
        prev = self.combo_upscale.currentText()
        if model in REALESRGAN_MODELS:
            scales = [REALESRGAN_MODELS[model]["scale"]]
        elif model in SR_MODELS:
            scales = sr_model_scales(model)
        else:
            scales = [2, 3, 4, 6, 8]
        self.combo_upscale.blockSignals(True)
        self.combo_upscale.clear()
        self.combo_upscale.addItems([f"{s}×" for s in scales])
        idx = self.combo_upscale.findText(prev)
        self.combo_upscale.setCurrentIndex(idx if idx >= 0 else 0)
        self.combo_upscale.blockSignals(False)

    def apply_upscale(self):
        if not self.current_pil_image:
            return
        scale = int(self.combo_upscale.currentText().replace("×", ""))
        model = self.combo_upscale_method.currentData()   # None -> Lanczos
        use_ai = model is not None
        sharpen = self.chk_upscale_sharpen.isChecked()
        denoise = self.chk_upscale_denoise.isChecked()
        use_gpu = self.settings.value("gpu_acceleration", True, type=bool)
        w, h = self.current_pil_image.size
        if w * h * scale * scale > 60_000_000:
            if QMessageBox.question(
                    self, "Large image",
                    f"The result will be about {w*scale}×{h*scale} pixels, which may "
                    "use a lot of memory and time. Continue?") != QMessageBox.StandardButton.Yes:
                return

        def operation(img, report):
            return upscale_image(img, scale=scale, model=model, use_ai=use_ai,
                                 sharpen=sharpen, denoise=denoise,
                                 use_gpu=use_gpu, report=report)
        self._run_async_operation(operation, "Before Upscale", f"Upscaling ×{scale}…")

    def open_model_manager(self):
        dialog = ModelManagerDialog(self)
        dialog.exec()
        # A just-downloaded model may enable AI removal
        ensure_rembg()
        self._update_ui_states()

    def apply_crop(self):
        crop_qrect = self.image_label_preview.get_crop_rect()
        if not crop_qrect or crop_qrect.isEmpty():
            QMessageBox.warning(self, "Warning", "No crop area selected.")
            return
        box = (crop_qrect.left(), crop_qrect.top(), crop_qrect.right(), crop_qrect.bottom())
        def operation(img):
            cropped_img = img.crop(box)
            QTimer.singleShot(0, self.image_label_preview.fit_to_view)
            return cropped_img
        self._perform_operation(operation, "Before Crop", "Cropping…")

    # ---- magic wand ----
    def calculate_wand_selection(self, start_point):
        if self.current_pil_image is None:
            return
        if not SCIPY_AVAILABLE:
            QMessageBox.critical(self, "Error", "Magic Wand requires 'scipy' (pip install scipy).")
            return
        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        try:
            img = self.current_pil_image.convert('RGB')
            width, height = img.size
            x, y = start_point.x(), start_point.y()
            if not (0 <= x < width and 0 <= y < height):
                self.image_label_preview.clear_wand_selection()
                return
            img_array = np.array(img, dtype=np.int32)
            target_color = img_array[y, x]
            tolerance_sq = self.wand_tolerance_spin.value() ** 2
            color_diff_sq = np.sum((img_array - target_color) ** 2, axis=2)
            seed_mask = color_diff_sq <= tolerance_sq
            labeled_array, _ = scipy_label(seed_mask)
            clicked_label = labeled_array[y, x]
            if clicked_label == 0:
                final_mask = np.zeros_like(seed_mask, dtype=bool)
            else:
                final_mask = labeled_array == clicked_label
            mask_pil = Image.fromarray((final_mask * 255).astype(np.uint8), 'L')
            self.image_label_preview.set_wand_preview(mask_pil)
        finally:
            QApplication.restoreOverrideCursor()
            self._update_ui_states()

    def apply_wand_remove(self):
        if self.image_label_preview.wand_selection_mask is None:
            QMessageBox.warning(self, "Warning", "No area selected with the Magic Wand.")
            return
        selection_mask = self.image_label_preview.wand_selection_mask
        def operation(img):
            alpha_np = np.array(img.getchannel('A'))
            alpha_np[np.array(selection_mask, dtype=bool)] = 0
            img.putalpha(Image.fromarray(alpha_np))
            return img
        self._perform_operation(operation, "Before Wand Remove", "Removing Selected Area…")

    def apply_wand_keep(self):
        if self.image_label_preview.wand_selection_mask is None:
            QMessageBox.warning(self, "Warning", "No area selected with the Magic Wand.")
            return
        selection_mask = self.image_label_preview.wand_selection_mask
        def operation(img):
            alpha_np = np.array(img.getchannel('A'))
            alpha_np[~np.array(selection_mask, dtype=bool)] = 0
            img.putalpha(Image.fromarray(alpha_np))
            return img
        self._perform_operation(operation, "Before Wand Keep", "Keeping Selected Area…")

    # ---- color / mask ----
    def select_color_to_remove(self):
        color = QColorDialog.getColor(parent=self)
        if color.isValid():
            self.selected_color_rgb = color.getRgb()[:3]
            self.color_preview.setText(f" R:{color.red()} G:{color.green()} B:{color.blue()}")
            self.color_preview.setStyleSheet(f"background:{color.name()}; border:1px solid #bbb; padding:5px; border-radius:4px;")
        self._update_ui_states()

    def apply_color_removal(self):
        if self.selected_color_rgb is None:
            return
        def operation(img):
            data = np.array(img.convert("RGBA"))
            rgb, alpha = data[:, :, :3], data[:, :, 3]
            diff_sq = np.sum((rgb.astype(np.int32) - np.array(self.selected_color_rgb)) ** 2, axis=2)
            alpha[diff_sq <= self.tolerance_spin.value() ** 2] = 0
            data[:, :, 3] = alpha
            return Image.fromarray(data, 'RGBA')
        self._perform_operation(operation, "Before Color Removal", "Removing Color…")

    def apply_mask_refinement(self):
        overlay_pixmap = self.image_label_preview.get_overlay_pixmap()
        if overlay_pixmap.toImage().isNull() or overlay_pixmap.toImage().allGray():
            QMessageBox.information(self, "Info", "No marks to apply. Use the drawing tools first.")
            return
        def operation(img):
            mask_pil = qimage_to_pil(overlay_pixmap.toImage())
            keep_mask = mask_pil.getchannel('G')
            if np.any(np.array(keep_mask)):
                img.paste(self.original_pil_image, (0, 0), keep_mask)
            remove_mask = mask_pil.getchannel('R')
            if np.any(np.array(remove_mask)):
                alpha = img.getchannel('A')
                alpha = Image.fromarray(np.minimum(np.array(alpha), 255 - np.array(remove_mask)))
                img.putalpha(alpha)
            return img
        self._perform_operation(operation, "Before Mask Refinement", "Applying Marks…")
        self.image_label_preview.clear_overlay()

    # ---- background fill ----
    def fill_background(self):
        if not self.current_pil_image:
            return
        color = QColorDialog.getColor(self.background_color or Qt.GlobalColor.white, self, "Choose Background Color")
        if not color.isValid():
            return
        self._push_state(self.current_pil_image.copy(), "Fill Background")
        self.background_color = color
        self._update_display()
        self._update_ui_states()

    def remove_background_color(self):
        if self.background_color is None:
            return
        self._push_state(self.current_pil_image.copy(), "Remove Fill")
        self.background_color = None
        self._update_display()
        self._update_ui_states()

    # ---- perspective ----
    def _on_perspective_mode_changed(self):
        n = 4 if self.radio_4pt.isChecked() else 6
        self.image_label_preview.set_perspective_required(n)
        self._update_perspective_hint()
        self._update_ui_states()

    def _update_perspective_hint(self):
        n = self.image_label_preview.perspective_required
        placed = len(self.image_label_preview.perspective_points)
        order4 = ["Top-Left", "Top-Right", "Bottom-Right", "Bottom-Left"]
        order6 = order4 + ["Top-Middle", "Bottom-Middle"]
        order = order4 if n == 4 else order6
        if placed >= n:
            self.persp_hint.setText(f"✓ All {n} points placed. Drag to adjust, then Apply.")
        else:
            self.persp_hint.setText(f"Point {placed + 1} of {n}: click the <b>{order[placed]}</b>.")

    def apply_perspective(self):
        pts = self.image_label_preview.get_perspective_points()
        required = self.image_label_preview.perspective_required
        if len(pts) != required:
            QMessageBox.warning(self, "Warning", f"Place all {required} points first.")
            return
        def operation(img):
            result = perspective_correct(img, pts)
            QTimer.singleShot(0, self.image_label_preview.fit_to_view)
            return result
        self._perform_operation(operation, "Before Perspective Correction", "Correcting Perspective…")

    # ---- about ----
    def show_about(self):
        QMessageBox.about(self, f"About {APP_NAME}",
            f"<h3>{APP_NAME}</h3>"
            f"<p><b>{APP_TAGLINE}</b> — v{APP_VERSION}</p>"
            "<p>A professional background removal & image editing tool.</p>"
            "<ul>"
            "<li>AI background removal (rembg) — cancellable with progress</li>"
            "<li>AI enhance: sharpen &amp; upscale (Real-ESRGAN / EDSR / …)</li>"
            "<li>Batch background removal for whole folders</li>"
            "<li>Manual brushes, Magic Wand, color removal</li>"
            "<li>4-point &amp; 6-point perspective correction</li>"
            "<li>Image overlay / compositing</li>"
            "<li>PDF &amp; SVG (vector) export</li>"
            "</ul>"
            f"<p>Developed by <b>{APP_AUTHOR}</b></p>"
            f'<p><a href="{GITHUB_URL}">Project on GitHub</a></p>')

    # ---- batch / preferences / theme / help / updates ----
    def open_batch_dialog(self):
        if not REMBG_AVAILABLE:
            QMessageBox.warning(self, "Unavailable",
                                "rembg / onnxruntime is not available, so batch "
                                "background removal is disabled.")
            return
        dlg = BatchProcessDialog(self)
        dlg.exec()

    def open_preferences(self):
        dlg = PreferencesDialog(self)
        if dlg.exec() == QDialog.DialogCode.Accepted:
            # Theme may have changed; re-apply and sync the menu toggle.
            self.current_theme = self.settings.value("theme", "light", type=str)
            self.action_theme.blockSignals(True)
            self.action_theme.setChecked(self.current_theme == "dark")
            self.action_theme.blockSignals(False)
            apply_theme(QApplication.instance(), self.current_theme)
            self._refresh_zoom_icons()
            # Provider preference may have changed; drop cached sessions.
            self._rembg_sessions.clear()

    def _refresh_zoom_icons(self):
        """Rebuild the drawn zoom icons for the current theme."""
        if not hasattr(self, "action_zoom_in"):
            return
        self.action_zoom_in.setIcon(self._zoom_icon("in"))
        self.action_zoom_out.setIcon(self._zoom_icon("out"))
        self.action_zoom_reset.setIcon(self._zoom_icon("fit"))

    def toggle_theme(self, dark):
        self.current_theme = "dark" if dark else "light"
        self.settings.setValue("theme", self.current_theme)
        apply_theme(QApplication.instance(), self.current_theme)
        self._refresh_zoom_icons()

    def show_shortcuts(self):
        rows = [
            ("Open image", "Ctrl+O"), ("Paste image", "Ctrl+V"),
            ("Save as…", "Ctrl+Shift+S"), ("Export PDF", "Ctrl+P"),
            ("Copy image", "Ctrl+C"), ("Undo", "Ctrl+Z"), ("Redo", "Ctrl+Y"),
            ("Zoom in", "Ctrl+ +"), ("Zoom out", "Ctrl+ -"),
            ("Fit to view", "Ctrl+0"), ("Preferences", "Ctrl+,"),
            ("Keyboard shortcuts", "F1"), ("Quit", "Ctrl+Q"),
        ]
        body = "".join(
            f"<tr><td style='padding:3px 18px 3px 0'>{name}</td>"
            f"<td><b>{keys}</b></td></tr>" for name, keys in rows)
        dlg = QDialog(self)
        dlg.setWindowTitle("Keyboard Shortcuts")
        lay = QVBoxLayout(dlg)
        view = QTextBrowser()
        view.setHtml(f"<h3>Keyboard Shortcuts</h3><table>{body}</table>")
        lay.addWidget(view)
        btns = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
        btns.rejected.connect(dlg.reject)
        btns.accepted.connect(dlg.accept)
        lay.addWidget(btns)
        dlg.resize(360, 420)
        dlg.exec()

    def check_for_updates(self, silent=True):
        """Query GitHub's latest-release API on a worker thread and report if a
        newer version exists. `silent` suppresses the 'up to date' popup (used
        for the optional automatic check on startup)."""
        self._update_worker = UpdateCheckWorker()
        self._update_worker.done.connect(
            lambda ok, latest, url: self._on_update_checked(ok, latest, url, silent))
        self._update_worker.start()

    def _on_update_checked(self, ok, latest, url, silent):
        if not ok:
            if not silent:
                QMessageBox.information(self, "Check for Updates",
                                        f"Could not check for updates.\n\n{latest}")
            return
        if _version_tuple(latest) > _version_tuple(APP_VERSION):
            box = QMessageBox(self)
            box.setWindowTitle("Update Available")
            box.setTextFormat(Qt.TextFormat.RichText)
            box.setText(f"A newer version <b>{latest}</b> is available "
                        f"(you have {APP_VERSION}).<br>"
                        f'<a href="{url}">Open the download page</a>')
            box.exec()
        elif not silent:
            QMessageBox.information(self, "Check for Updates",
                                    f"You're up to date (v{APP_VERSION}).")

    def closeEvent(self, event):
        # Persist window geometry and last-used preferences for next launch.
        try:
            self.settings.setValue("geometry", self.saveGeometry())
            self.settings.setValue("theme", self.current_theme)
            up_model = self.combo_upscale_method.currentData()
            self.settings.setValue("upscale_model", up_model if up_model else "")
            self.settings.setValue("upscale_denoise", self.chk_upscale_denoise.isChecked())
            self.settings.setValue("upscale_sharpen", self.chk_upscale_sharpen.isChecked())
            self.settings.setValue("rembg_model", self.rembg_model)
        except Exception as e:
            print(f"Could not save settings: {e}")
        # Ask any background AI threads to stop, then give them a moment to end.
        for worker in list(self._active_workers):
            worker.cancel()
        for worker in list(self._active_workers):
            worker.wait(2000)
        for path in self.temp_files_to_clean:
            if os.path.exists(path):
                try:
                    os.remove(path)
                except Exception as e:
                    print(f"Error cleaning temp file {path}: {e}")
        event.accept()


# ======================================================================
# Theme
# ======================================================================

THEME_PALETTES = {
    "light": {
        "window": "#f4f6fa", "text": "#23272e", "surface": "#ffffff",
        "border": "#e2e6ee", "input_border": "#ced5e2", "menu_border": "#d6dbe6",
        "tab_bg": "#eef1f7", "tab_text": "#4a5162", "tab_hover": "#e3e8f2",
        "hover": "#f0f5ff", "pressed": "#dfeafd", "tool_hover": "#e9f0fe",
        "tool_pressed": "#d4e2fc", "tool_text": "#2b3038", "groove": "#dfe4ee",
        "disabled_bg": "#f2f3f6", "disabled_text": "#a9afba",
        "status_text": "#4a5162", "primary_disabled": "#b9c6dd",
    },
    "dark": {
        "window": "#1e2229", "text": "#e6e9ef", "surface": "#262b33",
        "border": "#3a414c", "input_border": "#454d5a", "menu_border": "#3a414c",
        "tab_bg": "#2b313a", "tab_text": "#aab3c0", "tab_hover": "#333b46",
        "hover": "#2f3a4d", "pressed": "#26344a", "tool_hover": "#2f3a4d",
        "tool_pressed": "#26344a", "tool_text": "#dce1ea", "groove": "#3a414c",
        "disabled_bg": "#2a2f37", "disabled_text": "#6b7280",
        "status_text": "#aab3c0", "primary_disabled": "#3a4a63",
    },
}


def apply_theme(app, theme="light"):
    app.setStyle("Fusion")
    p = THEME_PALETTES.get(theme, THEME_PALETTES["light"])
    qss = f"""
    QMainWindow, QDialog {{ background: {p['window']}; }}
    QWidget {{ font-family: '{UI_FONT}', 'Helvetica Neue', 'Segoe UI', Arial; font-size: 13px; color: {p['text']}; }}
    QLabel {{ background: transparent; }}
    QToolTip {{ background: {p['surface']}; color: {p['text']}; border: 1px solid {p['border']}; }}

    QMenuBar {{ background: {p['surface']}; border-bottom: 1px solid {p['border']}; }}
    QMenuBar::item {{ padding: 6px 12px; background: transparent; }}
    QMenuBar::item:selected {{ background: {ACCENT}; color: white; border-radius: 4px; }}
    QMenu {{ background: {p['surface']}; border: 1px solid {p['menu_border']}; padding: 4px; }}
    QMenu::item {{ padding: 6px 24px; border-radius: 4px; }}
    QMenu::item:selected {{ background: {ACCENT}; color: white; }}

    QToolBar {{ background: {p['surface']}; border-bottom: 1px solid {p['border']}; spacing: 4px; padding: 4px; }}
    QToolBar::separator {{ background: {p['groove']}; width: 1px; margin: 4px 6px; }}
    QToolButton {{ padding: 6px 10px; border-radius: 6px; color: {p['tool_text']}; }}
    QToolButton:hover {{ background: {p['tool_hover']}; }}
    QToolButton:pressed {{ background: {p['tool_pressed']}; }}

    QTabWidget::pane {{ border: 1px solid {p['border']}; border-radius: 8px; background: {p['surface']}; top: -1px; }}
    QTabBar::tab {{ background: {p['tab_bg']}; color: {p['tab_text']}; padding: 8px 18px; border-top-left-radius: 7px;
                    border-top-right-radius: 7px; margin-right: 2px; font-weight: 500; }}
    QTabBar::tab:selected {{ background: {p['surface']}; color: {ACCENT}; border: 1px solid {p['border']}; border-bottom: none; }}
    QTabBar::tab:hover:!selected {{ background: {p['tab_hover']}; }}

    QGroupBox {{ background: {p['surface']}; border: 1px solid {p['border']}; border-radius: 8px;
                 margin-top: 14px; padding: 10px 8px 8px 8px; font-weight: 600; }}
    QGroupBox::title {{ subcontrol-origin: margin; subcontrol-position: top left; left: 12px;
                        padding: 0 6px; color: {ACCENT}; }}

    QPushButton {{ background: {p['surface']}; border: 1px solid {p['input_border']}; border-radius: 6px;
                   padding: 7px 12px; color: {p['tool_text']}; }}
    QPushButton:hover {{ background: {p['hover']}; border-color: {ACCENT}; }}
    QPushButton:pressed {{ background: {p['pressed']}; }}
    QPushButton:disabled {{ background: {p['disabled_bg']}; color: {p['disabled_text']}; border-color: {p['border']}; }}
    QPushButton:checked {{ background: {ACCENT}; color: white; border-color: {ACCENT_DARK}; }}

    QPushButton#primaryButton {{ background: {ACCENT}; color: white; border: none; font-weight: 600; padding: 9px 12px; }}
    QPushButton#primaryButton:hover {{ background: {ACCENT_DARK}; }}
    QPushButton#primaryButton:disabled {{ background: {p['primary_disabled']}; color: #eef2f8; }}

    QComboBox, QSpinBox, QDoubleSpinBox, QLineEdit {{ background: {p['surface']}; border: 1px solid {p['input_border']};
                   border-radius: 6px; padding: 5px 8px; min-height: 20px; color: {p['text']}; }}
    QComboBox:focus, QSpinBox:focus, QDoubleSpinBox:focus, QLineEdit:focus {{ border-color: {ACCENT}; }}
    QComboBox::drop-down {{ border: none; width: 20px; }}
    QComboBox QAbstractItemView {{ background: {p['surface']}; border: 1px solid {p['menu_border']}; selection-background-color: {ACCENT};
                   selection-color: white; outline: none; }}

    QCheckBox, QRadioButton {{ spacing: 6px; }}
    QCheckBox::indicator, QRadioButton::indicator {{ width: 16px; height: 16px; }}

    QListWidget, QTextBrowser {{ background: {p['surface']}; border: 1px solid {p['border']}; border-radius: 6px; color: {p['text']}; }}
    QProgressBar {{ background: {p['tab_bg']}; border: 1px solid {p['border']}; border-radius: 6px; text-align: center; }}
    QProgressBar::chunk {{ background: {ACCENT}; border-radius: 6px; }}

    QSlider::groove:horizontal {{ height: 5px; background: {p['groove']}; border-radius: 3px; }}
    QSlider::sub-page:horizontal {{ background: {ACCENT}; border-radius: 3px; }}
    QSlider::handle:horizontal {{ background: {p['surface']}; border: 2px solid {ACCENT}; width: 14px;
                   margin: -6px 0; border-radius: 8px; }}

    QScrollArea {{ border: none; }}
    QStatusBar {{ background: {p['surface']}; border-top: 1px solid {p['border']}; color: {p['status_text']}; }}
    QStatusBar::item {{ border: none; }}

    QProgressDialog {{ background: {p['surface']}; }}
    """
    app.setStyleSheet(qss)


# ======================================================================
# Entry point
# ======================================================================

def main():
    # Crisp rendering on fractional-scaling displays (125%, 150%, …). Must be
    # set before the QApplication is constructed.
    try:
        QApplication.setHighDpiScaleFactorRoundingPolicy(
            Qt.HighDpiScaleFactorRoundingPolicy.PassThrough)
    except Exception:
        pass
    if hasattr(Qt, 'AA_EnableHighDpiScaling'):
        QApplication.setAttribute(Qt.ApplicationAttribute.AA_EnableHighDpiScaling, True)
    if hasattr(Qt, 'AA_UseHighDpiPixmaps'):
        QApplication.setAttribute(Qt.ApplicationAttribute.AA_UseHighDpiPixmaps, True)

    app = QApplication.instance() or QApplication(sys.argv)
    app.setApplicationName(APP_NAME)
    app.setApplicationDisplayName(APP_NAME)
    app.setWindowIcon(get_app_icon())
    app.setOrganizationName(APP_ORG)
    startup_theme = QSettings(APP_ORG, APP_NAME).value("theme", "light", type=str)
    apply_theme(app, startup_theme)

    # --- Splash screen ---
    splash_pix = create_splash_pixmap()
    splash = QSplashScreen(splash_pix, Qt.WindowType.WindowStaysOnTopHint)
    splash.setFont(QFont(UI_FONT, 10))
    splash.show()

    def splash_msg(text):
        splash.showMessage(f"  {text}",
                           Qt.AlignmentFlag.AlignBottom | Qt.AlignmentFlag.AlignLeft,
                           QColor("#dbe6ff"))
        app.processEvents()

    splash_msg("Initializing interface…")
    app.processEvents()

    splash_msg("Loading AI engine (rembg)…")
    ensure_rembg()

    if not SCIPY_AVAILABLE:
        print("Warning: 'scipy' not found. Magic Wand disabled. (pip install scipy)")
    if not CV2_AVAILABLE:
        print("Warning: 'opencv-python' not found. Perspective correction disabled.")
    if not REMBG_AVAILABLE:
        print("Warning: 'rembg'/'onnxruntime' not available. AI removal disabled. "
              "(pip install rembg onnxruntime)")

    splash_msg("Preparing workspace…")
    main_win = MainWindow()

    splash_msg("Ready.")
    main_win.show()
    splash.finish(main_win)

    # Optional, non-blocking update check on startup (opt-in via Preferences).
    if main_win.settings.value("check_updates", False, type=bool):
        QTimer.singleShot(1500, lambda: main_win.check_for_updates(silent=True))

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
