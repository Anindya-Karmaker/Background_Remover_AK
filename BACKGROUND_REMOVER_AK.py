# --- START OF FILE BACKGROUND_REMOVER_AK.py ---

# --- BACKGROUND REMOVER AK (High-Performance Merged Version) ---
# This version combines the best features of both previous scripts.
# - High-performance overlay drawing for zero-lag feedback.
# - Deferred processing until explicit user action (e.g., 'Apply' button).
# - Optimized state management and pre-caching for fast UI response.
# - A single, unified preview window for a simpler, more intuitive workflow.
# - Tabbed interface for cleaner tool organization.
# - Upscaling functionality has been removed as requested.
# - NEW: Background fill is now a non-destructive layer that can be changed or removed.
# - NEW: Magic Wand tool with high-performance selection via SciPy.

import sys
import io
import numpy as np
import os
import tempfile
import requests
import onnxruntime
from collections import deque 
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QLabel, QVBoxLayout, QHBoxLayout,
    QPushButton, QFileDialog, QSplitter, QMenu,
    QMessageBox, QColorDialog, QSpinBox, QSizePolicy,
    QCheckBox, QComboBox, QFormLayout, QDialog, QStatusBar,
    QGroupBox, QProgressDialog, QScrollArea, QTabWidget, QSlider
)
from PySide6.QtGui import (
    QPixmap, QImage, QPainter, QPen, QBrush, QColor, QCursor, QIcon,
    QKeySequence, QAction, QDesktopServices
)
from PySide6.QtCore import Qt, QPoint, QRect, QBuffer, QByteArray, QSize, Signal, QUrl, QMimeData, QTimer

# --- Optional Imports with Fallbacks ---
try:
    from rembg import remove as remove_bg
    REMBG_AVAILABLE = True
except ImportError:
    REMBG_AVAILABLE = False
    def remove_bg(*args, **kwargs): raise ImportError("rembg library is not installed.")

# <<<--- MODIFICATION: Add SciPy for fast Magic Wand
try:
    from scipy.ndimage import label
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
# <<<--- END MODIFICATION

from PIL import Image, ImageDraw, ImageQt

# ... (The rest of the helper functions remain the same) ...
def pil_to_qpixmap(pil_image):
    if pil_image is None: return QPixmap()
    try:
        return QPixmap.fromImage(ImageQt.ImageQt(pil_image.convert("RGBA")))
    except Exception as e:
        print(f"Error converting PIL to QPixmap: {e}")
        return QPixmap()

def qimage_to_pil(qimage):
    if qimage.isNull(): return None
    buffer = QBuffer()
    buffer.open(QBuffer.OpenModeFlag.ReadWrite)
    qimage.save(buffer, 'PNG')
    return Image.open(io.BytesIO(buffer.data())).convert('RGBA')

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
    painter.setPen(Qt.PenStyle.NoPen); painter.setBrush(fill_color)
    painter.drawEllipse(QPoint(int(center), int(center)), int(radius), int(radius))
    painter.end()
    return QCursor(pixmap, int(center), int(center))

# ... (InteractiveLabel class remains the same until the end) ...
class InteractiveLabel(QLabel):
    MODE_NONE, MODE_KEEP, MODE_REMOVE, MODE_CROP, MODE_WAND = 0, 1, 2, 3, 4 # Added MODE_WAND
    interaction_started = Signal()
    stroke_committed = Signal(QPixmap)
    wand_point_selected = Signal(QPoint) # Signal for Magic Wand clicks

    def __init__(self, parent=None):
        super().__init__(parent)
        self.current_mode = self.MODE_NONE
        self.drawing, self.cropping = False, False
        self.last_point, self.crop_start_point, self.crop_end_point = QPoint(), QPoint(), QPoint()
        self.crop_rect_visual = None
        self.base_pixmap, self.overlay_pixmap = QPixmap(), QPixmap()
        self.wand_preview_pixmap = QPixmap() # For Magic Wand selection preview
        self.wand_selection_mask = None # Holds the raw PIL mask for the wand
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
        self.zoom_level = max(0.1, level)
        if not self.base_pixmap.isNull():
            self.resize(self.base_pixmap.size() * self.zoom_level)
        self.update_cursor()
        self.update()

    def fit_to_view(self):
        if self.base_pixmap.isNull() or not self.scroll_area: return
        vp_size = self.scroll_area.viewport().size() - QSize(2, 2)
        img_size = self.base_pixmap.size()
        if img_size.width() == 0 or img_size.height() == 0: return
        w_ratio = vp_size.width() / img_size.width()
        h_ratio = vp_size.height() / img_size.height()
        self.set_zoom(min(w_ratio, h_ratio))

    def set_mode(self, mode):
        if self.current_mode == self.MODE_WAND and mode != self.MODE_WAND:
            self.clear_wand_selection()
            
        self.current_mode = mode if self.current_mode != mode else self.MODE_NONE
        if self.current_mode != self.MODE_CROP:
            self.crop_rect_visual = None
        self.update_cursor()
        self.update()

    def clear_interaction_state(self):
        self.crop_rect_visual, self.drawing, self.cropping = None, False, False
        self.clear_overlay()
        self.clear_wand_selection()

    def clear_wand_selection(self):
        if not self.wand_preview_pixmap.isNull():
            self.wand_preview_pixmap.fill(Qt.GlobalColor.transparent)
        self.wand_selection_mask = None
        self.update()

    def set_wand_preview(self, mask_pil):
        self.clear_wand_selection()
        if mask_pil is None: return

        self.wand_selection_mask = mask_pil
        preview_color = QColor(0, 150, 255, 100) # Semi-transparent blue
        mask_np = np.array(mask_pil)
        color_img_np = np.zeros((mask_np.shape[0], mask_np.shape[1], 4), dtype=np.uint8)
        color_img_np[:,:,0] = preview_color.red()
        color_img_np[:,:,1] = preview_color.green()
        color_img_np[:,:,2] = preview_color.blue()
        color_img_np[:,:,3] = (mask_np / 255.0 * preview_color.alpha()).astype(np.uint8)
        
        preview_pil = Image.fromarray(color_img_np, 'RGBA')
        self.wand_preview_pixmap = pil_to_qpixmap(preview_pil)
        self.update()

    def get_crop_rect(self):
        if not self.crop_rect_visual or self.base_pixmap.isNull(): return None
        img_x = self.crop_rect_visual.x() / self.zoom_level
        img_y = self.crop_rect_visual.y() / self.zoom_level
        img_w = self.crop_rect_visual.width() / self.zoom_level
        img_h = self.crop_rect_visual.height() / self.zoom_level
        return QRect(int(img_x), int(img_y), int(img_w), int(img_h)).normalized()

    def map_to_image(self, view_point):
        if self.base_pixmap.isNull() or self.zoom_level == 0: return QPoint(0,0)
        return view_point / self.zoom_level

    def set_brush_size(self, size):
        self.brush_size = max(1, size)
        self.update_cursor()

    def update_cursor(self):
        if not self.isEnabled(): return self.setCursor(Qt.CursorShape.ArrowCursor)
        cursor_size = self.brush_size * self.zoom_level
        if self.current_mode == self.MODE_KEEP:
            self.setCursor(create_brush_cursor(cursor_size, QColor(0, 255, 0)))
        elif self.current_mode == self.MODE_REMOVE:
            self.setCursor(create_brush_cursor(cursor_size, QColor(255, 0, 0)))
        elif self.current_mode == self.MODE_CROP:
            self.setCursor(Qt.CursorShape.CrossCursor)
        elif self.current_mode == self.MODE_WAND:
            self.setCursor(Qt.CursorShape.CrossCursor) 
        else: self.setCursor(Qt.CursorShape.ArrowCursor)

    def mousePressEvent(self, event):
        if not self.isEnabled() or self.base_pixmap.isNull() or event.button() != Qt.MouseButton.LeftButton:
            return super().mousePressEvent(event)
        
        self.interaction_started.emit()
        if self.current_mode in [self.MODE_KEEP, self.MODE_REMOVE]:
            self.drawing = True
            self.last_point = event.pos()
            self._draw_on_overlay(self.last_point, self.last_point)
        elif self.current_mode == self.MODE_CROP:
            self.cropping = True
            self.crop_start_point = self.crop_end_point = event.pos()
            self.update()
        elif self.current_mode == self.MODE_WAND:
            self.wand_point_selected.emit(self.map_to_image(event.pos()))


    def mouseMoveEvent(self, event):
        if not self.isEnabled() or not (event.buttons() & Qt.MouseButton.LeftButton): return
        if self.drawing:
            self._draw_on_overlay(self.last_point, event.pos())
            self.last_point = event.pos()
        elif self.cropping:
            self.crop_end_point = event.pos()
            self.update()

    def mouseReleaseEvent(self, event):
        if not self.isEnabled() or event.button() != Qt.MouseButton.LeftButton: return
        if self.drawing:
            self.drawing = False
            self.stroke_committed.emit(self.overlay_pixmap)
        elif self.cropping:
            self.cropping = False
            self.crop_rect_visual = QRect(self.crop_start_point, self.crop_end_point).normalized()
            self.update()

    def _draw_on_overlay(self, start_point, end_point):
        painter = QPainter(self.overlay_pixmap)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        
        start_img_point = self.map_to_image(start_point)
        end_img_point = self.map_to_image(end_point)
        
        color = QColor(0, 255, 0, 180) if self.current_mode == self.MODE_KEEP else QColor(255, 0, 0, 180)
        pen = QPen(color, self.brush_size, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap, Qt.PenJoinStyle.RoundJoin)
        painter.setPen(pen)
        
        if start_point == end_point:
             painter.drawPoint(start_img_point)
        else:
             painter.drawLine(start_img_point, end_img_point)
        
        painter.end()
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        if self.base_pixmap.isNull():
            painter.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, "Load, Paste, or Drag & Drop an Image")
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
            painter.setPen(pen); painter.setBrush(Qt.BrushStyle.NoBrush)
            rect = self.crop_rect_visual if not self.cropping else QRect(self.crop_start_point, self.crop_end_point).normalized()
            painter.drawRect(rect)

# ... (MainWindow __init__, _create_actions, _create_toolbars are unchanged) ...
class MainWindow(QMainWindow):
    MAX_HISTORY = 20

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Background Remover AK")
        self.setGeometry(100, 100, 1000, 500)
        self.temp_files_to_clean = []
        
        self.original_pil_image, self.current_pil_image = None, None
        self.original_qpixmap, self.current_qpixmap = None, None
        
        self.background_color = None # QColor or None

        self.undo_stack, self.redo_stack = [], []

        self.rembg_model = "u2net"
        self.alpha_matting_enabled = False
        self.fg_threshold, self.bg_threshold, self.erode_size = 240, 10, 10
        self.brush_size = 20
        self.setAcceptDrops(True)

        self._create_widgets()
        self._create_actions()
        self._create_toolbars()
        self._create_layout()
        self._create_status_bar()
        self._connect_signals()

        QApplication.clipboard().dataChanged.connect(self._update_ui_states)
        self._update_ui_states()

    def _create_widgets(self):
        self.image_label_preview = InteractiveLabel()
        self.image_label_preview.setToolTip("Preview/Edit area.\nRight-click for context menu. Use tools to edit.")
        self.scroll_area_preview = QScrollArea()
        self.scroll_area_preview.setWidget(self.image_label_preview)
        self.scroll_area_preview.setWidgetResizable(False)
        self.scroll_area_preview.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.image_label_preview.set_scroll_area(self.scroll_area_preview)

    def _create_actions(self):
        self.action_open = QAction("Open...", self, shortcut=QKeySequence.StandardKey.Open, toolTip="Open Image (Ctrl+O)")
        self.action_paste = QAction("Paste Image", self, shortcut=QKeySequence.StandardKey.Paste, toolTip="Paste Image from Clipboard (Ctrl+V)")
        self.action_save = QAction("Save As...", self, shortcut=QKeySequence.StandardKey.SaveAs, toolTip="Save Processed Image (Ctrl+Shift+S)")
        self.action_copy = QAction("Copy Image", self, shortcut=QKeySequence.StandardKey.Copy, toolTip="Copy Processed Image to Clipboard (Ctrl+C)")
        self.action_quit = QAction("Quit", self, shortcut=QKeySequence.StandardKey.Quit, toolTip="Exit Application (Ctrl+Q)")
        self.action_undo = QAction("Undo", self, shortcut=QKeySequence.StandardKey.Undo, toolTip="Undo Last Action (Ctrl+Z)")
        self.action_redo = QAction("Redo", self, shortcut="Ctrl+Y", toolTip="Redo Last Action (Ctrl+Y)")
        self.action_reset = QAction("Reset Image", self, toolTip="Reset Image to Original Loaded State")
        self.action_zoom_in = QAction("Zoom In", self, shortcut="Ctrl+=", toolTip="Zoom In (Ctrl++)")
        self.action_zoom_out = QAction("Zoom Out", self, shortcut="Ctrl+-", toolTip="Zoom Out (Ctrl+-)")
        self.action_zoom_reset = QAction("Reset Zoom", self, shortcut="Ctrl+0", toolTip="Fit image to view (Ctrl+0)")
        self.action_info = QAction("About", self, toolTip="Visit GitHub page for help and info")

    def _create_toolbars(self):
        file_toolbar = self.addToolBar("File")
        file_toolbar.addActions([self.action_open, self.action_paste, self.action_save, self.action_copy])
        edit_toolbar = self.addToolBar("Edit")
        edit_toolbar.addActions([self.action_undo, self.action_redo, self.action_reset])
        edit_toolbar.addSeparator()
        edit_toolbar.addActions([self.action_zoom_out, self.action_zoom_in, self.action_zoom_reset])
        info_toolbar = self.addToolBar("Info")
        info_toolbar.addAction(self.action_info)

    def _create_layout(self):
        control_widget = QWidget()
        control_layout = QVBoxLayout(control_widget)
        control_widget.setMaximumWidth(350)
        
        tab_widget = QTabWidget()

        ai_tools_widget = QWidget()
        ai_tools_layout = QVBoxLayout(ai_tools_widget)
        rembg_group = QGroupBox("Background Removal (rembg)")
        rembg_layout = QFormLayout(rembg_group)
        self.btn_rembg = QPushButton("Remove Background")
        self.model_combo = QComboBox(); self.model_combo.addItems(["u2net", "u2netp", "u2net_human_seg", "silueta", "isnet-general-use", "isnet-anime"])
        self.cb_alpha_matting = QCheckBox("Enable Alpha Matting")
        self.spin_fg_thresh = QSpinBox(); self.spin_fg_thresh.setRange(1, 254); self.spin_fg_thresh.setValue(self.fg_threshold)
        self.spin_bg_thresh = QSpinBox(); self.spin_bg_thresh.setRange(1, 254); self.spin_bg_thresh.setValue(self.bg_threshold)
        self.spin_erode_size = QSpinBox(); self.spin_erode_size.setRange(0, 50); self.spin_erode_size.setValue(self.erode_size)
        rembg_layout.addRow(self.btn_rembg)
        rembg_layout.addRow("Model:", self.model_combo)
        rembg_layout.addRow(self.cb_alpha_matting)
        rembg_layout.addRow("FG Threshold:", self.spin_fg_thresh)
        rembg_layout.addRow("BG Threshold:", self.spin_bg_thresh)
        rembg_layout.addRow("Erode Size:", self.spin_erode_size)
        ai_tools_layout.addWidget(rembg_group)
        ai_tools_layout.addStretch()
        tab_widget.addTab(ai_tools_widget, "AI Tools")

        manual_edit_widget = QWidget()
        manual_edit_layout = QVBoxLayout(manual_edit_widget)
        
        tools_group = QGroupBox("Drawing & Crop Tools")
        tools_layout = QVBoxLayout(tools_group)
        self.btn_mode_crop = QPushButton("Select Crop Area"); self.btn_mode_crop.setCheckable(True)
        self.btn_apply_crop = QPushButton("Apply Crop")
        crop_layout = QHBoxLayout(); crop_layout.addWidget(self.btn_mode_crop, 1); crop_layout.addWidget(self.btn_apply_crop)
        tools_layout.addLayout(crop_layout)
        self.btn_mode_keep = QPushButton("Mark Keep"); self.btn_mode_keep.setCheckable(True)
        self.btn_mode_remove = QPushButton("Mark Remove"); self.btn_mode_remove.setCheckable(True)
        mask_mode_layout = QHBoxLayout(); mask_mode_layout.addWidget(self.btn_mode_keep); mask_mode_layout.addWidget(self.btn_mode_remove)
        tools_layout.addLayout(mask_mode_layout)
        self.brush_slider = QSlider(Qt.Orientation.Horizontal); self.brush_slider.setRange(1, 200); self.brush_slider.setValue(self.brush_size)
        self.brush_size_label_value = QLabel(f"{self.brush_size}px")
        brush_layout = QHBoxLayout(); brush_layout.addWidget(QLabel("Brush Size:")); brush_layout.addWidget(self.brush_slider); brush_layout.addWidget(self.brush_size_label_value)
        tools_layout.addLayout(brush_layout)
        self.btn_apply_mask = QPushButton("Apply Keep/Remove Marks")
        tools_layout.addWidget(self.btn_apply_mask)
        manual_edit_layout.addWidget(tools_group)

        magic_wand_group = QGroupBox("Magic Wand Tool")
        magic_wand_layout = QVBoxLayout(magic_wand_group)
        self.btn_magic_wand = QPushButton("Magic Wand Select"); self.btn_magic_wand.setCheckable(True)
        
        # <<<--- MODIFICATION: Add tooltip if scipy is missing
        if not SCIPY_AVAILABLE:
            self.btn_magic_wand.setToolTip("Functionality disabled: 'scipy' library not found.\n(Run: pip install scipy)")
        # <<<--- END MODIFICATION
            
        self.wand_tolerance_spin = QSpinBox(); self.wand_tolerance_spin.setRange(0, 255); self.wand_tolerance_spin.setValue(20)
        wand_tolerance_layout = QHBoxLayout(); wand_tolerance_layout.addWidget(QLabel("Tolerance:")); wand_tolerance_layout.addWidget(self.wand_tolerance_spin)
        self.btn_apply_wand_remove = QPushButton("Remove Selected Area (Delete)")
        self.btn_apply_wand_keep = QPushButton("Keep Selected Area (Remove BG)")
        magic_wand_layout.addWidget(self.btn_magic_wand)
        magic_wand_layout.addLayout(wand_tolerance_layout)
        magic_wand_layout.addWidget(self.btn_apply_wand_remove)
        magic_wand_layout.addWidget(self.btn_apply_wand_keep)
        manual_edit_layout.addWidget(magic_wand_group)
        
        # ... (rest of _create_layout is unchanged) ...
        other_tools_group = QGroupBox("Other Manual Tools")
        other_tools_layout = QVBoxLayout(other_tools_group)
        self.btn_select_color = QPushButton("Select Color to Remove")
        self.color_preview = QLabel(" None"); self.color_preview.setMinimumWidth(60); self.color_preview.setStyleSheet("border: 1px solid grey; background-color: lightgrey; padding: 5px;")
        self.selected_color_rgb = None
        color_select_layout = QHBoxLayout(); color_select_layout.addWidget(self.btn_select_color, 1); color_select_layout.addWidget(self.color_preview)
        other_tools_layout.addLayout(color_select_layout)
        self.tolerance_spin = QSpinBox(); self.tolerance_spin.setRange(0, 255); self.tolerance_spin.setValue(30)
        tolerance_layout = QHBoxLayout(); tolerance_layout.addWidget(QLabel("Tolerance:")); tolerance_layout.addWidget(self.tolerance_spin)
        other_tools_layout.addLayout(tolerance_layout)
        self.btn_apply_color_remove = QPushButton("Remove Selected Color")
        other_tools_layout.addWidget(self.btn_apply_color_remove)
        other_tools_layout.addSpacing(10)
        self.btn_fill_bg = QPushButton("Fill Background")
        self.btn_remove_fill = QPushButton("Remove Fill")
        fill_layout = QHBoxLayout(); fill_layout.addWidget(self.btn_fill_bg); fill_layout.addWidget(self.btn_remove_fill)
        other_tools_layout.addLayout(fill_layout)
        other_tools_layout.addSpacing(10)
        self.btn_show_original = QPushButton("Show Original (Hold)")
        other_tools_layout.addWidget(self.btn_show_original)
        manual_edit_layout.addWidget(other_tools_group)

        manual_edit_layout.addStretch()
        tab_widget.addTab(manual_edit_widget, "Manual Edit")

        control_layout.addWidget(tab_widget)
        
        main_splitter = QSplitter(Qt.Orientation.Horizontal)
        main_splitter.addWidget(self.scroll_area_preview)
        main_splitter.addWidget(control_widget)
        main_splitter.setSizes([950, 350])
        self.setCentralWidget(main_splitter)

    # ... (_create_status_bar and _connect_signals are unchanged) ...
    def _create_status_bar(self):
        self.statusBar = QStatusBar()
        self.setStatusBar(self.statusBar)
        self.statusBar.showMessage("Ready. Load an image, paste from clipboard, or drag & drop a file.")

    def _connect_signals(self):
        self.action_open.triggered.connect(self.open_image)
        self.action_paste.triggered.connect(self.paste_image)
        self.action_save.triggered.connect(self.save_image)
        self.action_copy.triggered.connect(self.copy_to_clipboard)
        self.action_quit.triggered.connect(self.close)
        self.action_undo.triggered.connect(self.undo_state)
        self.action_redo.triggered.connect(self.redo_state)
        self.action_reset.triggered.connect(self.reset_image)
        self.action_zoom_in.triggered.connect(lambda: self.image_label_preview.set_zoom(self.image_label_preview.zoom_level * 1.25))
        self.action_zoom_out.triggered.connect(lambda: self.image_label_preview.set_zoom(self.image_label_preview.zoom_level * 0.8))
        self.action_zoom_reset.triggered.connect(self.image_label_preview.fit_to_view)
        self.action_info.triggered.connect(lambda: QDesktopServices.openUrl(QUrl("https://github.com/Anindya-Karmaker/Background_Remover_AK")))

        self.btn_rembg.clicked.connect(self.run_rembg)
        self.model_combo.currentTextChanged.connect(lambda m: setattr(self, 'rembg_model', m))
        self.cb_alpha_matting.stateChanged.connect(lambda s: setattr(self, 'alpha_matting_enabled', bool(s)) or self._update_ui_states())
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
        
        self.mode_buttons = {
            InteractiveLabel.MODE_KEEP: self.btn_mode_keep,
            InteractiveLabel.MODE_REMOVE: self.btn_mode_remove,
            InteractiveLabel.MODE_CROP: self.btn_mode_crop,
            InteractiveLabel.MODE_WAND: self.btn_magic_wand, 
        }

    # ... (most of MainWindow methods are unchanged until _update_ui_states) ...
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
        if pil_image is None: return
        if len(self.undo_stack) >= self.MAX_HISTORY: self.undo_stack.pop(0)
        state = {
            "image": pil_image.copy(),
            "desc": description,
            "bg_color": self.background_color
        }
        self.undo_stack.append(state)
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

        base = QPixmap(self.current_qpixmap.size())
        painter = QPainter(base)
        
        if self.background_color:
            base.fill(self.background_color)
        else:
            checkerboard = create_checkerboard(base.width(), base.height())
            painter.drawPixmap(0, 0, checkerboard)

        painter.drawPixmap(0, 0, self.current_qpixmap)
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
        if len(self.undo_stack) <= 1: return
        self.redo_stack.append(self.undo_stack.pop())
        last_state = self.undo_stack[-1]
        
        self.current_pil_image = last_state["image"]
        self.current_qpixmap = pil_to_qpixmap(self.current_pil_image)
        self.background_color = last_state["bg_color"]

        self.image_label_preview.clear_interaction_state()
        self._update_display()
        self.statusBar.showMessage(f"Undo: Restored '{last_state['desc']}'", 3000)
        self._update_ui_states()

    def redo_state(self):
        if not self.redo_stack: return
        next_state = self.redo_stack.pop()
        self.undo_stack.append(next_state)

        self.current_pil_image = next_state["image"]
        self.current_qpixmap = pil_to_qpixmap(self.current_pil_image)
        self.background_color = next_state["bg_color"]
        
        self.image_label_preview.clear_interaction_state()
        self._update_display()
        self.statusBar.showMessage(f"Redo: Restored state", 3000)
        self._update_ui_states()
    
    def _perform_operation(self, operation_func, pre_op_desc, progress_title="Processing..."):
        if not self.current_pil_image: return
        progress = QProgressDialog(progress_title, None, 0, 0, self)
        progress.setWindowModality(Qt.WindowModal); progress.setCancelButton(None); progress.show()
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
        
        # <<<--- MODIFICATION: Disable wand if scipy is missing
        magic_wand_enabled = has_image and SCIPY_AVAILABLE
        self.btn_magic_wand.setEnabled(magic_wand_enabled)
        # <<<--- END MODIFICATION
        
        main_actions = [self.action_save, self.action_copy, self.action_reset, self.btn_rembg,
                        self.btn_mode_crop, self.btn_apply_crop, self.btn_mode_keep,
                        self.btn_mode_remove, self.btn_select_color, self.btn_apply_color_remove,
                        self.btn_apply_mask, self.btn_show_original]
        for action in main_actions:
            action.setEnabled(has_image)

        if not REMBG_AVAILABLE: self.btn_rembg.setEnabled(False); self.btn_rembg.setText("rembg not installed")
        
        self.btn_fill_bg.setEnabled(is_rgba)
        self.btn_remove_fill.setEnabled(is_rgba and self.background_color is not None)
        
        has_wand_selection = magic_wand_enabled and self.image_label_preview.wand_selection_mask is not None
        self.btn_apply_wand_remove.setEnabled(has_wand_selection)
        self.btn_apply_wand_keep.setEnabled(has_wand_selection)

        self.btn_apply_color_remove.setEnabled(has_image and self.selected_color_rgb is not None)
        self.action_undo.setEnabled(len(self.undo_stack) > 1)
        self.action_redo.setEnabled(bool(self.redo_stack))
        self.action_paste.setEnabled(QApplication.clipboard().mimeData().hasImage())

        is_matting = self.cb_alpha_matting.isChecked()
        for spin in [self.spin_fg_thresh, self.spin_bg_thresh, self.spin_erode_size]:
            spin.setEnabled(is_matting)

    # ... (set_interaction_mode and other methods are unchanged until the wand functions) ...
    def set_interaction_mode(self, mode_to_set, force_off=False):
        active_button = self.mode_buttons.get(mode_to_set)
        if force_off:
            current_mode = InteractiveLabel.MODE_NONE
            for btn in self.mode_buttons.values(): btn.setChecked(False)
        else:
            if not active_button: return
            is_now_checked = active_button.isChecked()
            current_mode = mode_to_set if is_now_checked else InteractiveLabel.MODE_NONE
            for mode, btn in self.mode_buttons.items():
                if mode != mode_to_set: btn.setChecked(False)
        self.image_label_preview.set_mode(current_mode)
    
    def open_image(self, path=None):
        if not path:
            path, _ = QFileDialog.getOpenFileName(self, "Open Image", "", "Images (*.png *.jpg *.jpeg *.bmp *.webp *.tiff *.gif)")
        if path:
            try: self._load_new_image(Image.open(path), f"Loaded '{os.path.basename(path)}'")
            except Exception as e: QMessageBox.critical(self, "Error", f"Failed to load image file: {e}")

    def paste_image(self):
        qimage = QApplication.clipboard().image()
        if not qimage.isNull(): self._load_new_image(qimage_to_pil(qimage), "Pasted from clipboard")
        else: QMessageBox.information(self, "Paste", "No valid image found on the clipboard.")
    
    def reset_image(self):
        if self.original_pil_image:
            self.background_color = None 
            self._set_current_image(self.original_pil_image.copy(), "Reset")
            self.image_label_preview.fit_to_view()
            self.statusBar.showMessage("Image reset to original.", 3000)
            self._update_ui_states()

    def save_image(self):
        if not self.current_pil_image: return
        path, sel_filter = QFileDialog.getSaveFileName(self, "Save Image As...", "processed_image.png", "PNG (*.png);;JPG (*.jpg);;TIFF (*.tiff);;WebP (*.webp)")
        if not path: return
        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        try:
            image_to_save = self.current_pil_image.copy()
            if self.background_color:
                background = Image.new('RGB', image_to_save.size, self.background_color.getRgb()[:3])
                background.paste(image_to_save, mask=image_to_save)
                image_to_save = background
            
            if path.lower().endswith(('.jpg', '.jpeg')) and image_to_save.mode == 'RGBA':
                background = Image.new("RGB", image_to_save.size, (255, 255, 255))
                background.paste(image_to_save, mask=image_to_save.getchannel('A'))
                image_to_save = background

            image_to_save.save(path)
            self.statusBar.showMessage(f"Image saved to {path}", 5000)
        except Exception as e: QMessageBox.critical(self, "Save Error", f"Failed to save image: {e}")
        finally: QApplication.restoreOverrideCursor()

    def copy_to_clipboard(self):
        if not self.current_pil_image: return
        
        temp_path = None # Initialize to None for error handling
        try:
            image_to_copy = self.current_pil_image.copy()
            if self.background_color:
                background = Image.new('RGB', image_to_copy.size, self.background_color.getRgb()[:3])
                background.paste(image_to_copy, mask=image_to_copy)
                image_to_copy = background

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
            # If an error occurred after the temp file was created, try to clean it up now.
            if temp_path and os.path.exists(temp_path):
                try:
                    os.remove(temp_path)
                    if temp_path in self.temp_files_to_clean:
                        self.temp_files_to_clean.remove(temp_path)
                except OSError as cleanup_error:
                    print(f"Failed to clean up temporary file during error handling: {cleanup_error}")

    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls() and any(url.isLocalFile() for url in event.mimeData().urls()):
            event.acceptProposedAction()
    
    def dropEvent(self, event):
        urls = event.mimeData().urls()
        if urls:
            path = urls[0].toLocalFile()
            self.open_image(path)

    def run_rembg(self):
        if not REMBG_AVAILABLE: return
        def operation(img):
            return remove_bg(img,
                model=self.rembg_model, alpha_matting=self.alpha_matting_enabled,
                alpha_matting_foreground_threshold=self.spin_fg_thresh.value(),
                alpha_matting_background_threshold=self.spin_bg_thresh.value(),
                alpha_matting_erode_size=self.spin_erode_size.value())
        self._perform_operation(operation, "Before Background Removal", "Removing Background...")
    
    def apply_crop(self):
        crop_qrect = self.image_label_preview.get_crop_rect()
        if not crop_qrect or crop_qrect.isEmpty():
            QMessageBox.warning(self, "Warning", "No crop area selected."); return
        box = (crop_qrect.left(), crop_qrect.top(), crop_qrect.right(), crop_qrect.bottom())
        def operation(img):
            cropped_img = img.crop(box)
            QTimer.singleShot(0, self.image_label_preview.fit_to_view)
            return cropped_img
        self._perform_operation(operation, "Before Crop", "Cropping...")

    # <<<--- MODIFICATION: Complete rewrite of Magic Wand calculation for speed
    def calculate_wand_selection(self, start_point):
        """Performs a flood-fill from the start_point using scipy for high performance."""
        if self.current_pil_image is None: return
        if not SCIPY_AVAILABLE:
            QMessageBox.critical(self, "Error", "Magic Wand tool requires the 'scipy' library. Please install it using 'pip install scipy'.")
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
            tolerance = self.wand_tolerance_spin.value()
            tolerance_sq = tolerance ** 2

            # 1. Fast vectorized color distance calculation over the whole image
            color_diff_sq = np.sum((img_array - target_color) ** 2, axis=2)
            seed_mask = color_diff_sq <= tolerance_sq

            # 2. Use scipy.ndimage.label to find all connected components
            labeled_array, num_features = label(seed_mask)

            # 3. Identify the label of the component at the click position
            clicked_label = labeled_array[y, x]

            # 4. Create the final mask by selecting all pixels with the same label
            if clicked_label == 0:
                # Clicked on a pixel that wasn't within tolerance, so no selection
                final_mask = np.zeros_like(seed_mask, dtype=bool)
            else:
                final_mask = labeled_array == clicked_label
            
            mask_pil = Image.fromarray((final_mask * 255).astype(np.uint8), 'L')
            self.image_label_preview.set_wand_preview(mask_pil)
            
        finally:
            QApplication.restoreOverrideCursor()
            self._update_ui_states()

    # <<<--- END MODIFICATION
            
    def apply_wand_remove(self):
        # ... (This function remains unchanged)
        if self.image_label_preview.wand_selection_mask is None:
            QMessageBox.warning(self, "Warning", "No area selected with the Magic Wand.")
            return

        selection_mask = self.image_label_preview.wand_selection_mask
        
        def operation(img):
            alpha = img.getchannel('A')
            alpha_np = np.array(alpha)
            mask_np = np.array(selection_mask, dtype=bool)
            alpha_np[mask_np] = 0
            img.putalpha(Image.fromarray(alpha_np))
            return img

        self._perform_operation(operation, "Before Wand Remove", "Removing Selected Area...")

    def apply_wand_keep(self):
        # ... (This function remains unchanged)
        if self.image_label_preview.wand_selection_mask is None:
            QMessageBox.warning(self, "Warning", "No area selected with the Magic Wand.")
            return

        selection_mask = self.image_label_preview.wand_selection_mask
        
        def operation(img):
            alpha = img.getchannel('A')
            alpha_np = np.array(alpha)
            mask_np = np.array(selection_mask, dtype=bool)
            alpha_np[~mask_np] = 0
            img.putalpha(Image.fromarray(alpha_np))
            return img

        self._perform_operation(operation, "Before Wand Keep", "Keeping Selected Area...")

    # ... (All remaining functions are unchanged)
    def select_color_to_remove(self):
        color = QColorDialog.getColor(parent=self)
        if color.isValid():
            self.selected_color_rgb = color.getRgb()[:3]
            self.color_preview.setText(f" R:{color.red()} G:{color.green()} B:{color.blue()}")
            self.color_preview.setStyleSheet(f"background-color: {color.name()}; border: 1px solid grey; padding: 5px;")
        self._update_ui_states()

    def apply_color_removal(self):
        if self.selected_color_rgb is None: return
        def operation(img):
            data = np.array(img.convert("RGBA"))
            rgb, alpha = data[:, :, :3], data[:, :, 3]
            diff_sq = np.sum((rgb.astype(np.int32) - np.array(self.selected_color_rgb))**2, axis=2)
            alpha[diff_sq <= self.tolerance_spin.value()**2] = 0
            data[:,:,3] = alpha
            return Image.fromarray(data, 'RGBA')
        self._perform_operation(operation, "Before Color Removal", "Removing Color...")

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

        self._perform_operation(operation, "Before Mask Refinement", "Applying Marks...")
        self.image_label_preview.clear_overlay()

    def fill_background(self):
        if not self.current_pil_image: return
        color = QColorDialog.getColor(self.background_color or Qt.white, self, "Choose Background Color")
        if not color.isValid(): return

        self._push_state(self.current_pil_image.copy(), "Fill Background")
        
        self.background_color = color
        self._update_display()
        self._update_ui_states()

    def remove_background_color(self):
        if self.background_color is None: return

        self._push_state(self.current_pil_image.copy(), "Remove Fill")
        
        self.background_color = None
        self._update_display()
        self._update_ui_states()

    def closeEvent(self, event):
        for path in self.temp_files_to_clean:
            if os.path.exists(path):
                try: os.remove(path)
                except Exception as e: print(f"Error cleaning temp file {path}: {e}")
        event.accept()


if __name__ == "__main__":
    if hasattr(Qt, 'AA_EnableHighDpiScaling'):
        QApplication.setAttribute(Qt.AA_EnableHighDpiScaling, True)
    if hasattr(Qt, 'AA_UseHighDpiPixmaps'):
        QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps, True)
        
    app = QApplication.instance() or QApplication(sys.argv)
    if not REMBG_AVAILABLE: print("Warning: 'rembg' library not found. Background removal feature disabled. (pip install rembg)")
    # <<<--- MODIFICATION: Add warning for SciPy
    if not SCIPY_AVAILABLE: print("Warning: 'scipy' library not found. Magic Wand tool will be disabled. (pip install scipy)")
    # <<<--- END MODIFICATION
    main_win = MainWindow()
    main_win.show()
    sys.exit(app.exec())
