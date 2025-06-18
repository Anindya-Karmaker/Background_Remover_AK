# --- START OF FILE BACKGROUND_REMOVER_PYSIDE6.py ---

# Modified to use PySide6 instead of PyQt5
# Enhanced with Zoom, fixed Copy/Paste, Fill Background, Progress Indicators, and Save options.

import sys
import io
import numpy as np
import os      
import tempfile
import onnxruntime
from PIL import Image, ImageDraw, ImageQt, ImageOps, ImageGrab
from PySide6.QtWidgets import (
    QSlider, QApplication, QMainWindow, QWidget, QLabel, QVBoxLayout, QHBoxLayout,
    QPushButton, QFileDialog, QSplitter, QMenu, QMenuBar,
    QMessageBox, QColorDialog, QInputDialog, QSpinBox, QGraphicsScene,
    QGraphicsView, QGraphicsPixmapItem, QGraphicsRectItem, QSizePolicy,
    QCheckBox, QComboBox, QFormLayout, QDialogButtonBox, QDialog, QStatusBar,
    QGroupBox, QProgressDialog, QScrollArea
)
from PySide6.QtGui import (
    QPixmap, QImage, QPainter, QPen, QBrush, QColor, QCursor, QIcon,
    QKeySequence, QAction, QDesktopServices 
)
from PySide6.QtCore import Qt, QPoint, QRect, QBuffer, QByteArray, QSize, Signal, QUrl, QMimeData
# Make rembg import optional for basic running without it
try:
    from rembg import remove as remove_bg
    REMBG_AVAILABLE = True
except ImportError:
    REMBG_AVAILABLE = False
    print("Warning: 'rembg' library not found. Background removal feature will be disabled.")
    def remove_bg(*args, **kwargs): # Dummy function
        raise ImportError("rembg library is not installed.")

# --- Helper Functions ---
def pil_to_qpixmap(pil_image):
    if pil_image is None: return QPixmap()
    try:
        if pil_image.mode not in ("RGB", "RGBA"):
            pil_image = pil_image.convert("RGBA")
        img_byte_arr = io.BytesIO()
        pil_image.save(img_byte_arr, format='PNG')
        qimage = QImage()
        qimage.loadFromData(img_byte_arr.getvalue())
        return QPixmap.fromImage(qimage)
    except Exception as e:
        print(f"Error converting PIL to QPixmap: {e}")
        return QPixmap()

def qimage_to_pil(qimage):
    if qimage.isNull(): return None
    buffer = QBuffer()
    buffer.open(QBuffer.OpenModeFlag.ReadWrite)
    qimage.save(buffer, 'PNG')
    return Image.open(io.BytesIO(buffer.data())).convert('RGBA')

def create_checkerboard(width, height, grid_size=10):
    pixmap = QPixmap(width, height)
    pixmap.fill(Qt.GlobalColor.transparent)
    painter = QPainter(pixmap)
    color1, color2 = QColor(200, 200, 200), QColor(230, 230, 230)
    painter.setPen(Qt.PenStyle.NoPen)
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
    center = pix_size / 2.0
    radius = diameter / 2.0
    painter.setPen(QColor(0, 0, 0))
    painter.setBrush(Qt.BrushStyle.NoBrush)
    painter.drawEllipse(QPoint(int(center), int(center)), int(radius), int(radius))
    fill_color = QColor(color.red(), color.green(), color.blue(), 100)
    painter.setPen(Qt.PenStyle.NoPen)
    painter.setBrush(fill_color)
    painter.drawEllipse(QPoint(int(center), int(center)), int(radius), int(radius))
    painter.end()
    return QCursor(pixmap, int(center), int(center))

# --- Interactive Label with Zoom ---

class InteractiveLabel(QLabel):
    MODE_NONE, MODE_KEEP, MODE_REMOVE, MODE_CROP = 0, 1, 2, 3
    interaction_finished = Signal()
    interaction_started = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.current_mode = self.MODE_NONE
        self.drawing, self.cropping = False, False
        self.last_point, self.crop_start_point, self.crop_end_point = QPoint(), QPoint(), QPoint()
        self.crop_rect_item_visual = None
        self.overlay_pixmap, self.base_pixmap, self.checkerboard_pixmap = QPixmap(), QPixmap(), None
        self.zoom_level = 1.0
        self.brush_size = 10
        self.keep_points, self.remove_points, self.current_stroke = [], [], []
        self.scroll_area = None
        

        self.setMouseTracking(True)
        #self.setAlignment(Qt.AlignmentFlag.TopLeft)
        self.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Ignored)
        self.setStyleSheet("""
            QToolBar QToolButton {
                padding: 4px;
                margin: 2px;
                border-radius: 4px;
            }
            QToolBar QToolButton:hover {
                background-color: #e0e0e0;
            }
            QToolBar QToolButton:pressed {
                background-color: #c0c0c0;
                border: 1px solid #8c8c8c;
                padding: 4px; /* Maintain padding */
            }
        """)

    def set_scroll_area(self, scroll_area):
        self.scroll_area = scroll_area

    def set_pixmap(self, pixmap):
        if pixmap.isNull():
            self.base_pixmap, self.overlay_pixmap, self.checkerboard_pixmap = QPixmap(), QPixmap(), None
        else:
            self.base_pixmap = pixmap
            self.overlay_pixmap = QPixmap(self.base_pixmap.size())
            self.overlay_pixmap.fill(Qt.GlobalColor.transparent)
            self.checkerboard_pixmap = create_checkerboard(self.base_pixmap.width(), self.base_pixmap.height())
            self.keep_points, self.remove_points = [], []
        self.update_display()

    def set_zoom(self, level):
        self.zoom_level = max(0.1, level) # Prevent zooming too small
        if not self.base_pixmap.isNull():
            new_size = self.base_pixmap.size() * self.zoom_level
            self.resize(new_size)
        self.update_cursor()
        self.update()

    def fit_to_view(self):
        if self.base_pixmap.isNull() or not self.scroll_area: return
        vp_size = self.scroll_area.viewport().size()
        img_size = self.base_pixmap.size()
        if img_size.width() == 0 or img_size.height() == 0: return

        w_ratio = vp_size.width() / img_size.width()
        h_ratio = vp_size.height() / img_size.height()
        self.set_zoom(min(w_ratio, h_ratio))

    def set_mode(self, mode):
        self.current_mode = mode if self.current_mode != mode else self.MODE_NONE
        self.crop_rect_item_visual = None if self.current_mode != self.MODE_CROP else self.crop_rect_item_visual
        self.update_cursor()
        self.update()

    def clear_overlays(self):
        self.keep_points, self.remove_points = [], []
        if not self.overlay_pixmap.isNull(): self.overlay_pixmap.fill(Qt.GlobalColor.transparent)
        self.update_display()

    def clear_interaction_state(self):
        self.crop_rect_item_visual, self.drawing, self.cropping = None, False, False
        self.current_stroke = []
        self.update()

    def get_crop_rect(self):
        if not self.crop_rect_item_visual or self.base_pixmap.isNull(): return None
        # Crop rect is in widget coords, map directly to image coords
        img_x = self.crop_rect_item_visual.x() / self.zoom_level
        img_y = self.crop_rect_item_visual.y() / self.zoom_level
        img_w = self.crop_rect_item_visual.width() / self.zoom_level
        img_h = self.crop_rect_item_visual.height() / self.zoom_level
        return QRect(int(img_x), int(img_y), int(img_w), int(img_h)).normalized()

    def get_mask_strokes(self):
        if self.base_pixmap.isNull(): return [], []
        map_stroke = lambda stroke: [self.map_to_image(p) for p in stroke]
        return [map_stroke(s) for s in self.keep_points], [map_stroke(s) for s in self.remove_points]

    def map_to_image(self, view_point):
        if self.base_pixmap.isNull() or self.zoom_level == 0: return QPoint(0,0)
        img_x = max(0, min(view_point.x() / self.zoom_level, self.base_pixmap.width() - 1))
        img_y = max(0, min(view_point.y() / self.zoom_level, self.base_pixmap.height() - 1))
        return QPoint(int(img_x), int(img_y))

    def set_brush_size(self, size):
        self.brush_size = max(1, size)
        self.update_cursor()

    def update_cursor(self):
        if not self.isEnabled():
            self.setCursor(Qt.CursorShape.ArrowCursor)
            return
        
        cursor_size = self.brush_size * self.zoom_level
        if self.current_mode == self.MODE_KEEP:
            self.setCursor(create_brush_cursor(cursor_size, QColor(0, 255, 0)))
        elif self.current_mode == self.MODE_REMOVE:
            self.setCursor(create_brush_cursor(cursor_size, QColor(255, 0, 0)))
        elif self.current_mode == self.MODE_CROP:
            self.setCursor(Qt.CursorShape.CrossCursor)
        else:
            self.setCursor(Qt.CursorShape.ArrowCursor)

    def mousePressEvent(self, event):
        if not self.isEnabled() or self.base_pixmap.isNull() or event.button() != Qt.MouseButton.LeftButton:
            return super().mousePressEvent(event)

        self.interaction_started.emit()
        if self.current_mode in [self.MODE_KEEP, self.MODE_REMOVE]:
            self.drawing = True
            self.last_point = event.pos()
            self.current_stroke = [self.last_point]
            if not self.overlay_pixmap.isNull():
                img_point = self.map_to_image(event.pos())
                painter = QPainter(self.overlay_pixmap)
                painter.setRenderHint(QPainter.RenderHint.Antialiasing)
                color = QColor(0, 255, 0, 180) if self.current_mode == self.MODE_KEEP else QColor(255, 0, 0, 180)
                painter.setPen(Qt.PenStyle.NoPen)
                painter.setBrush(color)
                r = self.brush_size / 2.0
                painter.drawEllipse(img_point, r, r)
                painter.end()
                self.update_display()
        elif self.current_mode == self.MODE_CROP:
            self.cropping = True
            self.crop_start_point = self.crop_end_point = event.pos()
            self.update()

    def mouseMoveEvent(self, event):
        if not self.isEnabled() or not (event.buttons() & Qt.MouseButton.LeftButton):
            return

        if self.drawing:
            start_img_point = self.map_to_image(self.last_point)
            end_img_point = self.map_to_image(event.pos())
            painter = QPainter(self.overlay_pixmap)
            painter.setRenderHint(QPainter.RenderHint.Antialiasing)
            color = QColor(0, 255, 0, 180) if self.current_mode == self.MODE_KEEP else QColor(255, 0, 0, 180)
            pen = QPen(color, self.brush_size, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap, Qt.PenJoinStyle.RoundJoin)
            painter.setPen(pen)
            painter.drawLine(start_img_point, end_img_point)
            painter.end()
            self.last_point = event.pos()
            self.current_stroke.append(self.last_point)
            self.update_display()
        elif self.cropping:
            self.crop_end_point = event.pos()
            self.update()

    def mouseReleaseEvent(self, event):
        if not self.isEnabled() or event.button() != Qt.MouseButton.LeftButton:
            return

        interaction_done = False
        if self.drawing:
            self.drawing = False
            if self.current_stroke:
                if self.current_mode == self.MODE_KEEP: self.keep_points.append(list(self.current_stroke))
                elif self.current_mode == self.MODE_REMOVE: self.remove_points.append(list(self.current_stroke))
            self.current_stroke = []
            interaction_done = True
        elif self.cropping:
            self.cropping = False
            self.crop_rect_item_visual = QRect(self.crop_start_point, self.crop_end_point).normalized()
            interaction_done = True
        
        if interaction_done:
            self.interaction_finished.emit()

    def contextMenuEvent(self, event):
        if not self.isEnabled() or self.base_pixmap.isNull(): return
        menu = QMenu(self)
        zoom_in_action = menu.addAction("Zoom In (Ctrl+)")
        zoom_out_action = menu.addAction("Zoom Out (Ctrl-)")
        reset_zoom_action = menu.addAction("Reset Zoom (Fit to View)")
        
        zoom_in_action.triggered.connect(lambda: self.set_zoom(self.zoom_level * 1.25))
        zoom_out_action.triggered.connect(lambda: self.set_zoom(self.zoom_level * 0.8))
        reset_zoom_action.triggered.connect(self.fit_to_view)
        
        menu.exec(event.globalPos())

    def paintEvent(self, event):
        painter = QPainter(self)
        if self.base_pixmap.isNull():
            painter.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, "Load or Paste Image")
            return

        painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)
        target_rect = self.rect()

        if self.checkerboard_pixmap:
             painter.drawPixmap(target_rect, self.checkerboard_pixmap, self.checkerboard_pixmap.rect())
        
        painter.drawPixmap(target_rect, self.base_pixmap, self.base_pixmap.rect())
        
        if not self.overlay_pixmap.isNull():
             painter.drawPixmap(target_rect, self.overlay_pixmap, self.overlay_pixmap.rect())

        if self.cropping:
             pen = QPen(QColor(0, 100, 255, 220), 2 / self.zoom_level, Qt.PenStyle.DashLine)
             painter.setPen(pen)
             painter.setBrush(Qt.BrushStyle.NoBrush)
             painter.drawRect(QRect(self.crop_start_point, self.crop_end_point).normalized())
        elif self.crop_rect_item_visual:
             pen = QPen(QColor(0, 100, 255, 220), 2 / self.zoom_level, Qt.PenStyle.DashLine)
             painter.setPen(pen)
             painter.setBrush(Qt.BrushStyle.NoBrush)
             painter.drawRect(self.crop_rect_item_visual)

    def update_display(self):
        self.update()

# --- Main Application Window ---

class MainWindow(QMainWindow):
    MAX_HISTORY = 20

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Background Remover AK")
        self.setGeometry(100, 100, 1400, 800)
        self.temp_files_to_clean = []
        self.original_pil_image, self.current_pil_image = None, None
        self.undo_stack, self.redo_stack = [], []
        self.rembg_model = "u2net"
        self.alpha_matting_enabled = False
        self.fg_threshold, self.bg_threshold, self.erode_size = 240, 10, 10
        self.brush_size = 10

        self._create_widgets()
        self._create_actions()
        #self._create_menus()
        self._create_toolbars()
        self._create_layout()
        self._create_status_bar()
        self._connect_signals()
        
        QApplication.clipboard().dataChanged.connect(self._update_ui_states)
        self._update_ui_states()

    def _create_widgets(self):
        #self.image_label_original = InteractiveLabel()
        #self.image_label_original.setToolTip("Shows the original loaded image. Not editable.")
        #self.image_label_original.setEnabled(False) # Not interactive

        self.image_label_preview = InteractiveLabel()
        self.image_label_preview.setToolTip("Preview/Edit area.\nRight-click to zoom. Use tools to edit.")

        #self.scroll_area_original = QScrollArea()
        #self.scroll_area_original.setWidget(self.image_label_original)
        #self.scroll_area_original.setWidgetResizable(True)
        #self.scroll_area_original.setAlignment(Qt.AlignmentFlag.AlignCenter)

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
        self.action_undo = QAction("Undo", self, toolTip="Undo Last Action (Ctrl+Z)")
        self.action_undo.setShortcut(QKeySequence.StandardKey.Undo)
        self.action_redo = QAction("Redo", self, toolTip="Redo Last Action (Ctrl+Y)")
        self.action_redo.setShortcuts([QKeySequence.StandardKey.Redo, "Ctrl+Y"])
        self.action_reset = QAction("Reset Image", self, toolTip="Reset Image to Original Loaded State")
        self.action_zoom_in = QAction("Zoom In", self, shortcut="Ctrl+=", toolTip="Zoom In (Ctrl++)")
        self.action_zoom_out = QAction("Zoom Out", self, shortcut="Ctrl+-", toolTip="Zoom Out (Ctrl+-)")
        self.action_zoom_reset = QAction("Reset Zoom", self, shortcut="Ctrl+0", toolTip="Fit image to view (Ctrl+0)")
        self.action_info = QAction(QIcon.fromTheme("help-about"), "About", self, toolTip="Visit GitHub page for help and info")
    def _create_menus(self):
        menu_bar = self.menuBar()
        file_menu = menu_bar.addMenu("&File")
        file_menu.addAction(self.action_open)
        file_menu.addAction(self.action_paste)
        file_menu.addSeparator()
        file_menu.addAction(self.action_copy)
        file_menu.addAction(self.action_save)
        file_menu.addSeparator()
        file_menu.addAction(self.action_quit)
        edit_menu = menu_bar.addMenu("&Edit")
        edit_menu.addAction(self.action_undo)
        edit_menu.addAction(self.action_redo)
        edit_menu.addSeparator()
        edit_menu.addAction(self.action_reset)
        view_menu = menu_bar.addMenu("&View")
        view_menu.addAction(self.action_zoom_in)
        view_menu.addAction(self.action_zoom_out)
        view_menu.addAction(self.action_zoom_reset)
        self.action_zoom_in.triggered.connect(lambda: self.image_label_preview.set_zoom(self.image_label_preview.zoom_level * 1.25))
        self.action_zoom_out.triggered.connect(lambda: self.image_label_preview.set_zoom(self.image_label_preview.zoom_level * 0.8))
        self.action_zoom_reset.triggered.connect(self.image_label_preview.fit_to_view)

    def _create_toolbars(self):
        # File Toolbar
        file_toolbar = self.addToolBar("File")
        file_toolbar.addActions([self.action_open, self.action_paste, self.action_save, self.action_copy])

        # Edit Toolbar
        edit_toolbar = self.addToolBar("Edit")
        # Add a separator to visually group undo/redo from zoom
        edit_toolbar.addActions([self.action_undo, self.action_redo, self.action_reset])
        edit_toolbar.addSeparator()
        # Add the zoom actions directly to the toolbar
        edit_toolbar.addAction(self.action_zoom_out)
        edit_toolbar.addAction(self.action_zoom_in)
        edit_toolbar.addAction(self.action_zoom_reset)
        
        # Create a new toolbar for help/info to place it at the end
        info_toolbar = self.addToolBar("Info")
        info_toolbar.addAction(self.action_info)

    def _create_layout(self):
        # --- Control Panel ---
        control_widget = QWidget()
        control_layout = QVBoxLayout(control_widget)
        control_widget.setMaximumWidth(320)

        # Rembg Controls
        rembg_group = QGroupBox("Background Removal (rembg)")
        rembg_layout = QFormLayout(rembg_group)
        self.btn_rembg = QPushButton(QIcon.fromTheme("view-refresh"), " Remove Background")
        self.btn_rembg.setToolTip("Run AI background removal on the current image.")
        self.model_combo = QComboBox()
        self.model_combo.addItems(["u2net", "u2netp", "u2net_human_seg", "silueta", "isnet-general-use", "isnet-anime"])
        self.model_combo.setToolTip("Select the AI model for background removal.")
        self.cb_alpha_matting = QCheckBox("Enable Alpha Matting")
        self.cb_alpha_matting.setToolTip("Refine edges using alpha matting (slower).")
        self.spin_fg_thresh = QSpinBox()
        self.spin_fg_thresh.setRange(1, 254); self.spin_fg_thresh.setValue(self.fg_threshold)
        self.spin_bg_thresh = QSpinBox()
        self.spin_bg_thresh.setRange(1, 254); self.spin_bg_thresh.setValue(self.bg_threshold)
        self.spin_erode_size = QSpinBox()
        self.spin_erode_size.setRange(0, 50); self.spin_erode_size.setValue(self.erode_size)
        rembg_layout.addRow(self.btn_rembg)
        rembg_layout.addRow("Model:", self.model_combo)
        rembg_layout.addRow(self.cb_alpha_matting)
        rembg_layout.addRow("FG Threshold:", self.spin_fg_thresh)
        rembg_layout.addRow("BG Threshold:", self.spin_bg_thresh)
        rembg_layout.addRow("Erode Size:", self.spin_erode_size)
        
        # Edit Tools
        tools_group = QGroupBox("Editing Tools")
        tools_layout = QVBoxLayout(tools_group)
        self.btn_mode_crop = QPushButton(QIcon.fromTheme("transform-crop"), " Select Crop Area"); self.btn_mode_crop.setCheckable(True)
        self.btn_apply_crop = QPushButton("Apply Crop")
        crop_layout = QHBoxLayout(); crop_layout.addWidget(self.btn_mode_crop, 1); crop_layout.addWidget(self.btn_apply_crop)
        tools_layout.addLayout(crop_layout)

        self.btn_mode_keep = QPushButton(QIcon.fromTheme("list-add"), " Mark Keep"); self.btn_mode_keep.setCheckable(True)
        self.btn_mode_remove = QPushButton(QIcon.fromTheme("list-remove"), " Mark Remove"); self.btn_mode_remove.setCheckable(True)
        mask_mode_layout = QHBoxLayout(); mask_mode_layout.addWidget(self.btn_mode_keep); mask_mode_layout.addWidget(self.btn_mode_remove)
        tools_layout.addLayout(mask_mode_layout)

        self.brush_slider = QSlider(Qt.Orientation.Horizontal); self.brush_slider.setRange(1, 100); self.brush_slider.setValue(self.brush_size)
        self.brush_size_label_value = QLabel(f"{self.brush_size}px")
        brush_layout = QHBoxLayout(); brush_layout.addWidget(QLabel("Brush Size:")); brush_layout.addWidget(self.brush_slider); brush_layout.addWidget(self.brush_size_label_value)
        tools_layout.addLayout(brush_layout)

        self.btn_apply_mask = QPushButton("Apply Keep/Remove Marks")
        tools_layout.addWidget(self.btn_apply_mask)
        tools_layout.addSpacing(10)

        self.btn_select_color = QPushButton("Select Color to Remove")
        self.color_preview = QLabel(" None"); self.color_preview.setMinimumWidth(60); self.color_preview.setStyleSheet("border: 1px solid grey; background-color: lightgrey; padding: 5px;")
        self.selected_color_rgb = None
        color_select_layout = QHBoxLayout(); color_select_layout.addWidget(self.btn_select_color, 1); color_select_layout.addWidget(self.color_preview)
        tools_layout.addLayout(color_select_layout)
        self.tolerance_spin = QSpinBox(); self.tolerance_spin.setRange(0, 255); self.tolerance_spin.setValue(30)
        tolerance_layout = QHBoxLayout(); tolerance_layout.addWidget(QLabel("Tolerance:")); tolerance_layout.addWidget(self.tolerance_spin)
        tools_layout.addLayout(tolerance_layout)
        self.btn_apply_color_remove = QPushButton("Remove Selected Color")
        tools_layout.addWidget(self.btn_apply_color_remove)
        tools_layout.addSpacing(10)

        self.btn_fill_bg = QPushButton(QIcon.fromTheme("format-fill-color"), " Fill Background")
        self.btn_fill_bg.setToolTip("Fill transparent areas with a chosen color.\nTo change the color later, Undo this action first.")
        tools_layout.addWidget(self.btn_fill_bg)
        
        control_layout.addWidget(rembg_group)
        control_layout.addWidget(tools_group)
        control_layout.addStretch()

        # Main Layout
        main_splitter = QSplitter(Qt.Orientation.Horizontal)
        
        # The view_splitter is no longer needed, add the preview area directly
        main_splitter.addWidget(self.scroll_area_preview) 
        main_splitter.addWidget(control_widget)

        main_splitter.setSizes([900, 300])
        self.setCentralWidget(main_splitter)

    def _create_status_bar(self):
        self.statusBar = QStatusBar()
        self.setStatusBar(self.statusBar)
        self.statusBar.showMessage("Ready. Load an image or paste from clipboard.")

    def _connect_signals(self):
        # Actions
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
        
        # Controls
        self.btn_rembg.clicked.connect(self.run_rembg)
        self.model_combo.currentTextChanged.connect(lambda m: setattr(self, 'rembg_model', m))
        self.cb_alpha_matting.stateChanged.connect(self.toggle_alpha_matting)
        self.btn_mode_crop.clicked.connect(lambda: self.set_interaction_mode(InteractiveLabel.MODE_CROP))
        self.btn_apply_crop.clicked.connect(self.apply_crop)
        self.btn_mode_keep.clicked.connect(lambda: self.set_interaction_mode(InteractiveLabel.MODE_KEEP))
        self.btn_mode_remove.clicked.connect(lambda: self.set_interaction_mode(InteractiveLabel.MODE_REMOVE))
        self.brush_slider.valueChanged.connect(self._update_brush_size)
        self.btn_apply_mask.clicked.connect(self.apply_mask_refinement)
        self.btn_select_color.clicked.connect(self.select_color_to_remove)
        self.btn_apply_color_remove.clicked.connect(self.apply_color_removal)
        self.btn_fill_bg.clicked.connect(self.fill_background)
        self.action_info.triggered.connect(
            lambda: QDesktopServices.openUrl(QUrl("https://github.com/ahmed-hosny-dev/background-remover-pyside6")) # Replace with your repo URL
        )

        self.mode_buttons = {
            InteractiveLabel.MODE_KEEP: self.btn_mode_keep,
            InteractiveLabel.MODE_REMOVE: self.btn_mode_remove,
            InteractiveLabel.MODE_CROP: self.btn_mode_crop,
        }

    # --- State Management & UI Updates ---
    def _update_brush_size(self, value):
        self.brush_size = value
        self.image_label_preview.set_brush_size(self.brush_size)
        self.brush_size_label_value.setText(f"{value}px")

    def _push_state(self, image_state, description=""):
        if image_state is None: return
        if len(self.undo_stack) >= self.MAX_HISTORY: self.undo_stack.pop(0)
        self.undo_stack.append({"image": image_state.copy(), "desc": description})
        self.redo_stack.clear()
        self._update_ui_states()

    def _load_new_image(self, pil_image, source_desc="Loaded"):
        QApplication.setOverrideCursor(Qt.WaitCursor)
        try:
            pil_image = pil_image.convert('RGBA') if pil_image.mode != 'RGBA' else pil_image
            self.original_pil_image = pil_image.copy() # We still keep the data!
            self.current_pil_image = pil_image.copy()
            self.undo_stack, self.redo_stack = [], []
            self._push_state(self.current_pil_image, "Initial Load")

            # The loop is no longer needed, just clear the preview label
            self.image_label_preview.clear_overlays()
            self.image_label_preview.clear_interaction_state()

            # Remove the lines that updated the original view
            # self.image_label_original.set_pixmap(pil_to_qpixmap(self.original_pil_image))
            
            self.image_label_preview.set_pixmap(pil_to_qpixmap(self.current_pil_image))
            
            # Remove the fit_to_view for the original view
            # self.image_label_original.fit_to_view()
            self.image_label_preview.fit_to_view()
            
            self.selected_color_rgb = None
            self.color_preview.setText(" None"); self.color_preview.setStyleSheet("border: 1px solid grey; background-color: lightgrey; padding: 5px;")
            self.statusBar.showMessage(f"{source_desc} successfully.", 5000)
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to process loaded image: {e}")
            self._clear_workspace()
        finally:
            QApplication.restoreOverrideCursor()
            self._update_ui_states()
            self.set_interaction_mode(InteractiveLabel.MODE_NONE, force_off=True)

    def _clear_workspace(self):
        self.original_pil_image, self.current_pil_image = None, None
        self.undo_stack, self.redo_stack = [], []
        self.image_label_original.set_pixmap(QPixmap())
        self.image_label_preview.set_pixmap(QPixmap())
        self.statusBar.showMessage("Workspace cleared. Load or paste an image.")
        self._update_ui_states()

    def undo_state(self):
        if not self.undo_stack: return
        self.redo_stack.append({"image": self.current_pil_image.copy(), "desc": "State Before Undo"})
        last_state = self.undo_stack.pop()
        self.current_pil_image = last_state["image"]
        self.image_label_preview.set_pixmap(pil_to_qpixmap(self.current_pil_image))
        self.image_label_preview.clear_interaction_state()
        self.statusBar.showMessage(f"Undo: Restored '{last_state['desc']}'", 3000)
        self._update_ui_states()
        self.set_interaction_mode(InteractiveLabel.MODE_NONE, force_off=True)

    def redo_state(self):
        if not self.redo_stack: return
        self.undo_stack.append({"image": self.current_pil_image.copy(), "desc": "State Before Redo"})
        next_state = self.redo_stack.pop()
        self.current_pil_image = next_state["image"]
        self.image_label_preview.set_pixmap(pil_to_qpixmap(self.current_pil_image))
        self.image_label_preview.clear_interaction_state()
        self.statusBar.showMessage(f"Redo: Restored state", 3000)
        self._update_ui_states()
        self.set_interaction_mode(InteractiveLabel.MODE_NONE, force_off=True)

    def _update_ui_states(self):
        has_current = self.current_pil_image is not None
        has_alpha = has_current and self.current_pil_image.mode == 'RGBA'
        
        for action in [self.action_save, self.action_copy, self.action_reset,
                       self.btn_rembg, self.btn_mode_crop, self.btn_apply_crop,
                       self.btn_mode_keep, self.btn_mode_remove, self.btn_apply_mask,
                       self.btn_select_color, self.btn_apply_color_remove]:
            action.setEnabled(has_current)

        self.btn_fill_bg.setEnabled(has_alpha)
        self.btn_apply_color_remove.setEnabled(has_current and self.selected_color_rgb is not None)
        self.action_undo.setEnabled(bool(self.undo_stack))
        self.action_redo.setEnabled(bool(self.redo_stack))
        self.action_paste.setEnabled(QApplication.clipboard().mimeData().hasImage())

        is_matting_enabled = self.cb_alpha_matting.isChecked()
        for spin in [self.spin_fg_thresh, self.spin_bg_thresh, self.spin_erode_size]:
            spin.setEnabled(is_matting_enabled)
    
    def toggle_alpha_matting(self, state):
        self.alpha_matting_enabled = bool(state)
        self._update_ui_states()
    
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
    
    # --- Image I/O ---
    def open_image(self):
        path, _ = QFileDialog.getOpenFileName(self, "Open Image", "", "Images (*.png *.jpg *.jpeg *.bmp *.webp *.tiff *.gif)")
        if path:
            try:
                self._load_new_image(Image.open(path), f"Loaded '{path.split('/')[-1]}'")
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to load image file: {e}")

    def paste_image(self):
        qimage = QApplication.clipboard().image()
        if not qimage.isNull():
            self._load_new_image(qimage_to_pil(qimage), "Pasted from clipboard")
        else:
            QMessageBox.information(self, "Paste", "No valid image found on the clipboard.")
    
    def reset_image(self):
        if self.original_pil_image:
            self._push_state(self.current_pil_image, "State Before Reset")
            self.current_pil_image = self.original_pil_image.copy()
            self._push_state(self.current_pil_image, "Reset to Original")
            self.image_label_preview.set_pixmap(pil_to_qpixmap(self.current_pil_image))
            self.image_label_preview.fit_to_view()
            self.statusBar.showMessage("Image reset to original.", 3000)
        self._update_ui_states()
    
    def save_image(self):
        if not self.current_pil_image: return
        path, selected_filter = QFileDialog.getSaveFileName(self, "Save Image As...", "processed_image.png",
            "PNG Image (*.png);;TIFF Image (*.tif *.tiff);;JPEG Image (*.jpg *.jpeg);;WebP Image (*.webp);;BMP Image (*.bmp)")
        if not path: return

        QApplication.setOverrideCursor(Qt.WaitCursor)
        try:
            image_to_save = self.current_pil_image.copy()
            file_ext = path.split('.')[-1].lower()
            
            # Handle formats that don't support alpha
            if file_ext in ['jpg', 'jpeg', 'bmp']:
                if image_to_save.mode == 'RGBA':
                    reply = QMessageBox.question(self, "Transparency Warning",
                        f".{file_ext.upper()} does not support transparency. Save with a white background?",
                        QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No | QMessageBox.StandardButton.Cancel)
                    if reply == QMessageBox.StandardButton.Yes:
                        background = Image.new("RGB", image_to_save.size, (255, 255, 255))
                        background.paste(image_to_save, mask=image_to_save.getchannel('A'))
                        image_to_save = background
                    elif reply == QMessageBox.StandardButton.Cancel:
                        QApplication.restoreOverrideCursor(); return
                    else:
                        image_to_save = image_to_save.convert('RGB')
            
            image_to_save.save(path)
            self.statusBar.showMessage(f"Image saved to {path}", 5000)
        except Exception as e:
            QMessageBox.critical(self, "Save Error", f"Failed to save image: {e}")
        finally:
            QApplication.restoreOverrideCursor()

    # In MainWindow, replace the entire copy_to_clipboard method
    def copy_to_clipboard(self):
        if not self.current_pil_image:
            QMessageBox.warning(self, "Warning", "No image to copy.")
            return

        try:
            # Create a QMimeData object to hold multiple data formats
            mime_data = QMimeData()

            # --- Format 1: Standard Image Data (for most apps) ---
            # This part is fine and should be kept for compatibility.
            qimage = ImageQt.ImageQt(self.current_pil_image.copy())
            mime_data.setImageData(qimage)

            # --- Format 2: Raw PNG Data (NEW ROBUST IMPLEMENTATION) ---
            # This new logic mimics the reliable save_image function.
            # It uses PIL to save to an in-memory bytes buffer.
            img_byte_buffer = io.BytesIO()
            self.current_pil_image.save(img_byte_buffer, format='PNG')
            
            # Create a QByteArray from the raw bytes data.
            png_data = QByteArray(img_byte_buffer.getvalue())
            mime_data.setData("image/png", png_data)
            # END OF NEW IMPLEMENTATION

            # --- Format 3: Temporary File Path (for file-based apps) ---
            # This part is also fine and uses the reliable PIL save method.
            fd, temp_path = tempfile.mkstemp(suffix='.png', prefix='bgr-app-')
            os.close(fd) # Close the file descriptor, we just need the path

            # Save the image to the temporary path
            self.current_pil_image.save(temp_path, "PNG")

            # Add the file to our cleanup list
            self.temp_files_to_clean.append(temp_path)

            # Set the URL list for file-based clipboard operations
            mime_data.setUrls([QUrl.fromLocalFile(temp_path)])

            # Set the rich mime data on the clipboard
            QApplication.clipboard().setMimeData(mime_data)

            self.statusBar.showMessage("Image copied to clipboard in multiple formats.", 3000)

        except Exception as e:
            QMessageBox.critical(self, "Copy Error", f"Failed to copy image to clipboard: {e}")
            self.statusBar.showMessage("Copy failed.", 5000)

    # --- Image Processing Operations ---
    def run_rembg(self):
        if not self.current_pil_image or not REMBG_AVAILABLE: return
        progress = QProgressDialog("Processing...", None, 0, 0, self)
        progress.setWindowModality(Qt.WindowModal); progress.setCancelButton(None); progress.show()
        
        try:
            self._push_state(self.current_pil_image, "Before Background Removal")
            progress.setLabelText("Preparing image..."); QApplication.processEvents()
            input_image = self.current_pil_image.convert("RGB")
            
            progress.setLabelText("Applying AI model..."); QApplication.processEvents()
            result_image = remove_bg(input_image, model=self.rembg_model,
                alpha_matting=self.alpha_matting_enabled, alpha_matting_foreground_threshold=self.fg_threshold,
                alpha_matting_background_threshold=self.bg_threshold, alpha_matting_erode_size=self.erode_size)
            
            progress.setLabelText("Finalizing..."); QApplication.processEvents()
            self.current_pil_image = result_image
            self.image_label_preview.set_pixmap(pil_to_qpixmap(self.current_pil_image))
            self.statusBar.showMessage("Background removal complete.", 5000)
        except Exception as e:
            QMessageBox.critical(self, "Error", f"An error occurred during background removal: {e}")
            if self.undo_stack and self.undo_stack[-1]["desc"] == "Before Background Removal": self.undo_stack.pop()
        finally:
            progress.close()
            self._update_ui_states()
            self.set_interaction_mode(InteractiveLabel.MODE_NONE, force_off=True)

    def apply_crop(self):
        if not self.current_pil_image: return
        crop_qrect = self.image_label_preview.get_crop_rect()
        if not crop_qrect or crop_qrect.width() <= 0 or crop_qrect.height() <= 0:
            QMessageBox.warning(self, "Warning", "No crop area selected. Use 'Select Crop Area' tool first."); return
        
        QApplication.setOverrideCursor(Qt.WaitCursor)
        try:
            self._push_state(self.current_pil_image, "Before Crop")
            box = (crop_qrect.left(), crop_qrect.top(), crop_qrect.right(), crop_qrect.bottom())
            self.current_pil_image = self.current_pil_image.crop(box)
            self.image_label_preview.set_pixmap(pil_to_qpixmap(self.current_pil_image))
            self.image_label_preview.fit_to_view() # Refit after crop
            self.statusBar.showMessage("Crop applied.", 5000)
        finally:
            QApplication.restoreOverrideCursor()
            self.image_label_preview.clear_interaction_state()
            self._update_ui_states()
            self.set_interaction_mode(InteractiveLabel.MODE_NONE, force_off=True)

    def _perform_timed_operation(self, operation_func, pre_op_desc):
        if not self.current_pil_image: return
        QApplication.setOverrideCursor(Qt.WaitCursor)
        try:
            self._push_state(self.current_pil_image, pre_op_desc)
            self.current_pil_image = operation_func(self.current_pil_image.copy())
            self.image_label_preview.set_pixmap(pil_to_qpixmap(self.current_pil_image))
            self.statusBar.showMessage(f"{pre_op_desc.replace('Before ', '')} applied.", 5000)
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Operation failed: {e}")
            if self.undo_stack and self.undo_stack[-1]["desc"] == pre_op_desc: self.undo_stack.pop()
        finally:
            QApplication.restoreOverrideCursor()
            self.image_label_preview.clear_overlays()
            self.image_label_preview.clear_interaction_state()
            self._update_ui_states()
            self.set_interaction_mode(InteractiveLabel.MODE_NONE, force_off=True)
            
    def apply_mask_refinement(self):
        if not self.original_pil_image:
            QMessageBox.critical(self, "Error", "Cannot apply refinement without an original image reference.")
            return

        keep_strokes, remove_strokes = self.image_label_preview.get_mask_strokes()
        if not keep_strokes and not remove_strokes:
            QMessageBox.information(self, "Info", "No areas marked. Use 'Mark Keep/Remove' tools first.")
            return

        def operation(img):
            # Ensure both current and original images are RGBA for compositing
            img = img.convert("RGBA")
            original_img_rgba = self.original_pil_image.convert("RGBA")

            # --- Helper function to draw a clean, round stroke ---
            def draw_stroke(draw_context, stroke_points, fill_color, brush_width):
                if not stroke_points:
                    return
                radius = brush_width / 2.0
                
                # Draw the connecting lines for a continuous stroke
                if len(stroke_points) > 1:
                    draw_context.line(stroke_points, fill=fill_color, width=int(brush_width), joint="curve")
                
                # Draw circles at each point to ensure round caps and cover single clicks
                for p in stroke_points:
                    draw_context.ellipse(
                        (p[0] - radius, p[1] - radius, p[0] + radius, p[1] + radius),
                        fill=fill_color
                    )

            # 1. Handle "Keep" strokes: Restore from original
            if keep_strokes:
                # Create a temporary mask where 'keep' areas are white
                keep_mask = Image.new("L", img.size, 0) # Black background
                keep_draw = ImageDraw.Draw(keep_mask)
                
                for stroke in keep_strokes:
                    points_as_tuples = [(p.x(), p.y()) for p in stroke]
                    draw_stroke(keep_draw, points_as_tuples, fill_color=255, brush_width=self.brush_size)

                # Paste from the original image onto the current one, using the mask
                # This copies both RGB and Alpha data from the original where the mask is white.
                img.paste(original_img_rgba, (0, 0), keep_mask)

            # 2. Handle "Remove" strokes: Set alpha to 0
            if remove_strokes:
                # Get the alpha channel of the (potentially just modified) image
                alpha = img.getchannel('A')
                alpha_draw = ImageDraw.Draw(alpha)

                for stroke in remove_strokes:
                    points_as_tuples = [(p.x(), p.y()) for p in stroke]
                    draw_stroke(alpha_draw, points_as_tuples, fill_color=0, brush_width=self.brush_size)

                # Put the modified alpha channel back into the image
                img.putalpha(alpha)
                
            return img
            
        self._perform_timed_operation(operation, "Before Mask Refinement")

    def select_color_to_remove(self):
        if not self.current_pil_image: return
        color = QColorDialog.getColor(parent=self)
        if color.isValid():
            self.selected_color_rgb = color.getRgb()[:3]
            self.color_preview.setText(f" R:{color.red()} G:{color.green()} B:{color.blue()}")
            self.color_preview.setStyleSheet(f"background-color: {color.name()}; border: 1px solid grey; padding: 5px;")
        self._update_ui_states()

    def apply_color_removal(self):
        if self.selected_color_rgb is None:
             QMessageBox.warning(self, "Warning", "No color selected. Use 'Select Color' first."); return
        
        def operation(img):
            img = img.convert("RGBA")
            data = np.array(img)
            rgb, alpha = data[:, :, :3], data[:, :, 3]
            target_color = np.array(self.selected_color_rgb)
            tolerance_sq = self.tolerance_spin.value()**2
            diff_sq = np.sum((rgb.astype(np.int32) - target_color)**2, axis=2)
            mask = diff_sq <= tolerance_sq
            alpha[mask] = 0
            data[:, :, 3] = alpha
            return Image.fromarray(data, 'RGBA')
        self._perform_timed_operation(operation, "Before Color Removal")

    def fill_background(self):
        color = QColorDialog.getColor(parent=self, title="Choose Background Color")
        if not color.isValid(): return

        def operation(img):
            img = img.convert("RGBA")
            background = Image.new('RGB', img.size, color.getRgb()[:3])
            background.paste(img, mask=img.getchannel('A'))
            return background
        self._perform_timed_operation(operation, "Before Fill Background")

    def closeEvent(self, event):
        """
        Clean up any temporary files before the application closes.
        """
        for path in self.temp_files_to_clean:
            try:
                if os.path.exists(path):
                    os.remove(path)
                    print(f"Cleaned up temporary file: {path}")
            except Exception as e:
                print(f"Error cleaning up temporary file {path}: {e}")
        
        event.accept() # Allow the window to close

if __name__ == "__main__":
    app = QApplication.instance() or QApplication(sys.argv)
    main_win = MainWindow()
    main_win.show()
    sys.exit(app.exec())