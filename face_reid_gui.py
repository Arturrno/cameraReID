"""
PyQt6 Multi-Camera Face Re-identification GUI Application
==========================================================
A modern desktop application with:
- 2x2 camera grid layout
- Side panel with detected persons
- Add client functionality
- Multi-threaded AI processing with proper synchronization

Architecture:
- FrameGrabber threads: One per camera, grabs latest frame only (no buffer)
- AIProcessor thread: Single thread batches all cameras for GPU efficiency
- Synchronized display: All cameras show frames from same processing cycle

Author: Senior Computer Vision Engineer
Date: 2024
"""

import sys
import cv2
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import time
import logging
import threading
from queue import Queue, Empty

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QGridLayout, QLabel, QLineEdit, QPushButton, QScrollArea,
    QFrame, QSplitter, QFileDialog, QMessageBox, QSizePolicy,
    QDialog, QTextEdit, QListWidget, QListWidgetItem, QAbstractItemView
)
from PyQt6.QtCore import (
    Qt, QThread, pyqtSignal, pyqtSlot, QTimer, QSize, QMutex, QMutexLocker
)
from PyQt6.QtGui import QImage, QPixmap, QFont, QIcon, QPalette, QColor, QLinearGradient, QBrush, QPainter

# Import existing classes from multi_camera_face_reid_optimized
from multi_camera_face_reid_optimized import (
    OptimizedConfig as Config, 
    RobustFaceDetector, 
    OptimizedFaceEmbeddingEngine as RobustFaceEmbeddingEngine,
    OptimizedFaceGallery as RobustFaceGallery, 
    FaceDetection, 
    TrackedFace, 
    CandidateFace,
    EnhancedPreprocessor
)

# Configure logging (before using logger)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Import known faces manager for reference database
try:
    from known_faces_manager import KnownFacesManager, create_known_faces_manager
    KNOWN_FACES_AVAILABLE = True
except ImportError:
    KNOWN_FACES_AVAILABLE = False
    KnownFacesManager = None
    logger.warning("KnownFacesManager not available - reference photos disabled")


# ============================================================================
# Simple Dark Theme
# ============================================================================

class Theme:
    """Simple dark theme."""
    
    # Core colors
    BG_DARKEST = "#1a1a1a"
    BG_DARKER = "#1e1e1e"
    BG_DARK = "#252526"
    BG_MEDIUM = "#2d2d30"
    BG_LIGHT = "#3c3c3c"
    BG_LIGHTER = "#4a4a4a"
    
    # Text colors
    TEXT_PRIMARY = "#ffffff"
    TEXT_SECONDARY = "#cccccc"
    TEXT_MUTED = "#888888"
    
    # Accent colors
    ACCENT_BLUE = "#0078d4"
    ACCENT_GREEN = "#28a745"
    ACCENT_YELLOW = "#ffc107"
    ACCENT_RED = "#dc3545"
    ACCENT_PURPLE = "#6f42c1"
    ACCENT_CYAN = "#17a2b8"
    
    # Status colors
    STATUS_ONLINE = "#28a745"
    STATUS_OFFLINE = "#dc3545"
    STATUS_PENDING = "#ffc107"
    
    # Border colors
    BORDER_DEFAULT = "#3c3c3c"
    BORDER_FOCUS = "#0078d4"
    BORDER_SUCCESS = "#28a745"
    
    @classmethod
    def get_main_stylesheet(cls) -> str:
        """Get the main application stylesheet."""
        return f"""
            QMainWindow {{
                background-color: {cls.BG_DARKEST};
            }}
            QWidget {{
                font-family: 'Segoe UI', sans-serif;
                color: {cls.TEXT_PRIMARY};
            }}
            QLabel {{
                color: {cls.TEXT_PRIMARY};
            }}
            QLineEdit {{
                background-color: {cls.BG_MEDIUM};
                color: {cls.TEXT_PRIMARY};
                border: 1px solid {cls.BORDER_DEFAULT};
                padding: 8px;
            }}
            QLineEdit:focus {{
                border: 1px solid {cls.BORDER_FOCUS};
            }}
            QTextEdit {{
                background-color: {cls.BG_MEDIUM};
                color: {cls.TEXT_PRIMARY};
                border: 1px solid {cls.BORDER_DEFAULT};
                padding: 8px;
            }}
            QScrollBar:vertical {{
                background-color: {cls.BG_DARK};
                width: 10px;
            }}
            QScrollBar::handle:vertical {{
                background-color: {cls.BG_LIGHTER};
                min-height: 20px;
            }}
            QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{
                height: 0;
            }}
            QScrollArea {{
                border: none;
            }}
            QListWidget {{
                background-color: {cls.BG_DARK};
                color: {cls.TEXT_PRIMARY};
                border: 1px solid {cls.BORDER_DEFAULT};
            }}
            QListWidget::item {{
                padding: 8px;
            }}
            QListWidget::item:selected {{
                background-color: {cls.ACCENT_BLUE};
            }}
            QMessageBox {{
                background-color: {cls.BG_DARK};
            }}
        """
    
    @classmethod
    def button_primary(cls) -> str:
        return f"""
            QPushButton {{
                background-color: {cls.ACCENT_BLUE};
                color: white;
                border: none;
                padding: 10px 16px;
                font-weight: bold;
            }}
            QPushButton:hover {{
                background-color: #1a86d9;
            }}
            QPushButton:pressed {{
                background-color: #005a9e;
            }}
            QPushButton:disabled {{
                background-color: {cls.BG_LIGHT};
                color: {cls.TEXT_MUTED};
            }}
        """
    
    @classmethod
    def button_secondary(cls) -> str:
        return f"""
            QPushButton {{
                background-color: {cls.BG_MEDIUM};
                color: {cls.TEXT_PRIMARY};
                border: 1px solid {cls.BORDER_DEFAULT};
                padding: 10px 16px;
            }}
            QPushButton:hover {{
                background-color: {cls.BG_LIGHT};
            }}
            QPushButton:pressed {{
                background-color: {cls.BG_DARK};
            }}
        """
    
    @classmethod
    def button_success(cls) -> str:
        return f"""
            QPushButton {{
                background-color: {cls.ACCENT_GREEN};
                color: white;
                border: none;
                padding: 10px 16px;
                font-weight: bold;
            }}
            QPushButton:hover {{
                background-color: #2dbe4e;
            }}
            QPushButton:pressed {{
                background-color: #1e7e34;
            }}
        """
    
    @classmethod
    def button_danger(cls) -> str:
        return f"""
            QPushButton {{
                background-color: {cls.ACCENT_RED};
                color: white;
                border: none;
                padding: 10px 16px;
                font-weight: bold;
            }}
            QPushButton:hover {{
                background-color: #e04555;
            }}
            QPushButton:pressed {{
                background-color: #bd2130;
            }}
        """
    
    @classmethod
    def card_panel(cls) -> str:
        return f"""
            background-color: {cls.BG_DARK};
            border: 1px solid {cls.BORDER_DEFAULT};
        """


# ============================================================================
# Data Classes for GUI Communication
# ============================================================================

@dataclass
class PersonInfo:
    """Information about a detected person for GUI display."""
    person_id: int
    is_confirmed: bool
    face_thumbnail: np.ndarray
    first_name: str = ""
    last_name: str = ""
    last_seen_camera: int = -1
    last_seen_time: float = 0.0
    is_known: bool = False  # True if matched from reference database


# ============================================================================
# Frame Grabber Thread - One per camera, only grabs latest frame
# ============================================================================

class FrameGrabber(threading.Thread):
    """
    Lightweight thread that continuously grabs frames from a camera.
    Always keeps only the LATEST frame - no buffer buildup.
    """
    
    def __init__(self, camera_id: int, url: str, config: Config):
        super().__init__(daemon=True)
        self.camera_id = camera_id
        self.url = url
        self.config = config
        
        self._running = False
        self._connected = False
        self._cap: Optional[cv2.VideoCapture] = None
        
        # Latest frame storage (thread-safe)
        self._frame_lock = threading.Lock()
        self._latest_frame: Optional[np.ndarray] = None
        self._frame_time: float = 0.0
        
        self.logger = logging.getLogger(f"FrameGrabber-{camera_id}")
    
    def connect(self) -> bool:
        """Connect to the camera stream."""
        try:
            self.logger.info(f"Connecting to camera {self.camera_id}...")
            
            if self._cap is not None:
                self._cap.release()
            
            # Set TCP transport for reliability
            import os
            os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp"
            
            self._cap = cv2.VideoCapture(self.url, cv2.CAP_FFMPEG)
            
            # Minimal buffer - we only want latest frame
            self._cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            self._cap.set(cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, 15000)
            self._cap.set(cv2.CAP_PROP_READ_TIMEOUT_MSEC, 5000)
            self._cap.set(cv2.CAP_PROP_HW_ACCELERATION, cv2.VIDEO_ACCELERATION_ANY)
            
            if self._cap.isOpened():
                width = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                fps = self._cap.get(cv2.CAP_PROP_FPS)
                self.logger.info(f"Camera {self.camera_id} connected: {width}x{height} @ {fps:.1f}fps")
                self._connected = True
                return True
            else:
                self.logger.error(f"Failed to open camera {self.camera_id}")
                self._connected = False
                return False
                
        except Exception as e:
            self.logger.error(f"Connection error: {e}")
            self._connected = False
            return False
    
    def get_latest_frame(self) -> Tuple[Optional[np.ndarray], float]:
        """Get the latest frame (thread-safe). Returns (frame, timestamp)."""
        with self._frame_lock:
            if self._latest_frame is not None:
                return self._latest_frame.copy(), self._frame_time
            return None, 0.0
    
    def is_connected(self) -> bool:
        return self._connected
    
    def run(self):
        """Continuously grab frames, keeping only the latest."""
        self._running = True
        reconnect_attempts = 0
        
        while self._running:
            if not self._connected:
                if reconnect_attempts >= self.config.MAX_RECONNECT_ATTEMPTS:
                    self.logger.error("Max reconnection attempts reached")
                    break
                reconnect_attempts += 1
                if self.connect():
                    reconnect_attempts = 0
                else:
                    time.sleep(self.config.RECONNECT_DELAY)
                continue
            
            try:
                ret, frame = self._cap.read()
                
                if not ret or frame is None:
                    self._connected = False
                    continue
                
                # Skip corrupted frames
                if frame.size == 0:
                    continue
                
                # Store latest frame (overwrites previous - no buffer buildup!)
                with self._frame_lock:
                    self._latest_frame = frame
                    self._frame_time = time.time()
                    
            except Exception as e:
                self.logger.error(f"Frame grab error: {e}")
                self._connected = False
        
        if self._cap is not None:
            self._cap.release()
    
    def stop(self):
        self._running = False


# ============================================================================
# AI Processor Thread - Single thread for all cameras (GPU efficiency)
# ============================================================================

class AIProcessor(QThread):
    """
    Single AI processing thread that handles ALL cameras.
    Batches frames together for maximum GPU utilization.
    Emits synchronized results for all cameras.
    """
    
    # Signals
    frames_ready = pyqtSignal(dict)  # {camera_id: processed_frame}
    detections_ready = pyqtSignal(int, list)  # camera_id, detections
    connection_status = pyqtSignal(int, bool)  # camera_id, connected
    
    def __init__(self, grabbers: List[FrameGrabber], config: Config,
                 detector: RobustFaceDetector,
                 embedding_engine: RobustFaceEmbeddingEngine,
                 gallery: RobustFaceGallery,
                 parent=None):
        super().__init__(parent)
        
        self.grabbers = grabbers
        self.config = config
        self.detector = detector
        self.embedding_engine = embedding_engine
        self.gallery = gallery
        
        self._running = False
        self._frame_count = 0
        
        self.logger = logging.getLogger("AIProcessor")
    
    def _draw_detections(self, frame: np.ndarray, 
                         reid_detections: List[FaceDetection],
                         small_detections: List[FaceDetection]) -> np.ndarray:
        """Draw bounding boxes and labels on frame with person names."""
        # Professional color palette
        COLORS = [
            (88, 166, 255),   # Blue
            (63, 185, 80),    # Green  
            (210, 153, 34),   # Yellow
            (163, 113, 247),  # Purple
            (57, 197, 207),   # Cyan
            (248, 81, 73),    # Red
            (255, 123, 114),  # Light red
            (121, 192, 255),  # Light blue
        ]
        CANDIDATE_COLOR = (110, 118, 129)
        TOO_SMALL_COLOR = (72, 79, 88)
        KNOWN_COLOR = (63, 185, 80)  # Green for known persons
        
        for detection in reid_detections:
            x1, y1, x2, y2 = detection.bbox
            person_id = detection.person_id or 0
            
            if detection.is_confirmed:
                # Check if this is a known person and get their name
                known_name = self.gallery.get_known_person_name(person_id)
                is_known = self.gallery.is_known_person(person_id)
                
                if is_known and known_name:
                    # Known person - show name in green
                    first, last = known_name
                    name = f"{first} {last}".strip()
                    label = name if name else f"#{person_id}"
                    color = KNOWN_COLOR
                else:
                    # Unknown person - show ID
                    color = COLORS[person_id % len(COLORS)]
                    label = f"Unknown #{person_id}"
            else:
                color = CANDIDATE_COLOR
                label = f"? #{abs(person_id)}"
            
            # Draw rounded rectangle effect (thicker lines with anti-aliasing)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2, cv2.LINE_AA)
            
            # Draw label with background
            (label_w, label_h), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
            cv2.rectangle(frame, (x1, y1 - label_h - 12), (x1 + label_w + 10, y1), color, -1, cv2.LINE_AA)
            cv2.putText(frame, label, (x1 + 5, y1 - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2, cv2.LINE_AA)
        
        for detection in small_detections:
            x1, y1, x2, y2 = detection.bbox
            cv2.rectangle(frame, (x1, y1), (x2, y2), TOO_SMALL_COLOR, 1, cv2.LINE_AA)
        
        return frame
    
    def run(self):
        """Main processing loop - processes all cameras synchronously."""
        self._running = True
        self.logger.info("AI Processor started")
        
        while self._running:
            self._frame_count += 1
            
            # Skip frames for performance
            if self._frame_count % self.config.FRAME_SKIP != 0:
                time.sleep(0.01)  # Small sleep to not spin CPU
                continue
            
            # Collect latest frames from ALL cameras simultaneously
            frames_to_process: Dict[int, np.ndarray] = {}
            
            for grabber in self.grabbers:
                if grabber is None:
                    continue
                    
                frame, frame_time = grabber.get_latest_frame()
                camera_id = grabber.camera_id
                
                # Emit connection status
                self.connection_status.emit(camera_id, grabber.is_connected())
                
                if frame is not None:
                    # Only process recent frames (drop if too old - sync!)
                    if time.time() - frame_time < 0.5:  # Max 500ms old
                        frames_to_process[camera_id] = frame
            
            if not frames_to_process:
                time.sleep(0.01)
                continue
            
            # Process all frames
            processed_frames: Dict[int, np.ndarray] = {}
            
            # Process each camera separately to maintain proper per-camera preprocessing
            for camera_id, frame in frames_to_process.items():
                # Update camera stats
                self.embedding_engine.update_camera_stats(camera_id, frame)
                
                # Detect faces
                all_detections = self.detector.detect(frame)
                reid_detections, small_detections = self.detector.filter_for_reid(all_detections)
                
                # Track list for IOU updates
                current_tracks: List[Tuple[int, Tuple[int, int, int, int]]] = []
                
                if reid_detections:
                    # Extract embeddings with CORRECT camera_id for preprocessing
                    face_crops = [d.face_crop for d in reid_detections]
                    embeddings = self.embedding_engine.extract_embeddings_batch(face_crops, camera_id)
                    
                    # Match and update gallery
                    matched_detections = []
                    for detection, embedding in zip(reid_detections, embeddings):
                        detection.embedding = embedding
                        person_id, is_confirmed = self.gallery.match_and_update(
                            embedding, detection.bbox, camera_id
                        )
                        
                        # Skip if person_id is None (unknown face in KNOWN_FACES_ONLY_MODE)
                        if person_id is None:
                            continue
                        
                        detection.person_id = person_id
                        detection.is_confirmed = is_confirmed
                        matched_detections.append(detection)
                        
                        # Add to track list for IOU memory
                        if is_confirmed:
                            current_tracks.append((person_id, detection.bbox))
                    
                    # Replace with only matched detections
                    reid_detections = matched_detections
                
                # Update IOU track memory
                self.gallery.update_tracks(camera_id, current_tracks)
                
                # Draw and store
                processed_frame = self._draw_detections(frame.copy(), reid_detections, small_detections)
                processed_frames[camera_id] = processed_frame
                
                # Emit detections
                if reid_detections:
                    self.detections_ready.emit(camera_id, reid_detections)
            
            # Emit all processed frames at once (synchronized!)
            self.frames_ready.emit(processed_frames)
            
            # Small sleep to control frame rate
            time.sleep(0.01)
        
        self.logger.info("AI Processor stopped")
    
    def stop(self):
        self._running = False


# ============================================================================
# Legacy CameraWorker wrapper (for compatibility)
# ============================================================================

class CameraWorker:
    """Compatibility wrapper - not used in new architecture."""
    pass


# ============================================================================
# Camera Display Widget
# ============================================================================

class CameraWidget(QFrame):
    """Widget for displaying a single camera stream."""
    
    def __init__(self, camera_id: int, parent=None):
        super().__init__(parent)
        self.camera_id = camera_id
        self._setup_ui()
    
    def _setup_ui(self):
        """Setup the widget UI."""
        self.setFrameStyle(QFrame.Shape.NoFrame)
        self.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Preferred)
        
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        
        # Main container with styling
        container = QFrame()
        container.setStyleSheet(f"""
            background-color: {Theme.BG_DARK};
            border: 1px solid {Theme.BORDER_DEFAULT};
        """)
        container_layout = QVBoxLayout(container)
        container_layout.setContentsMargins(0, 0, 0, 0)
        container_layout.setSpacing(0)
        
        # Camera header bar
        header = QFrame()
        header.setFixedHeight(28)
        header.setStyleSheet(f"background-color: {Theme.BG_MEDIUM};")
        header_layout = QHBoxLayout(header)
        header_layout.setContentsMargins(8, 0, 8, 0)
        
        # Status indicator
        self.status_indicator = QLabel()
        self.status_indicator.setFixedSize(8, 8)
        self.status_indicator.setStyleSheet(f"background-color: {Theme.STATUS_OFFLINE};")
        header_layout.addWidget(self.status_indicator)
        
        self.title_label = QLabel(f"Camera {self.camera_id + 1}")
        self.title_label.setFont(QFont("Segoe UI", 9, QFont.Weight.Bold))
        header_layout.addWidget(self.title_label)
        
        header_layout.addStretch()
        
        self.status_label = QLabel("OFFLINE")
        self.status_label.setFont(QFont("Segoe UI", 8))
        self.status_label.setStyleSheet(f"color: {Theme.STATUS_OFFLINE};")
        header_layout.addWidget(self.status_label)
        
        container_layout.addWidget(header)
        
        # Video display area
        self.video_label = QLabel()
        self.video_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.video_label.setMinimumSize(320, 240)
        self.video_label.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Ignored)
        self.video_label.setStyleSheet(f"background-color: {Theme.BG_DARKEST};")
        self.video_label.setText("Connecting...")
        self.video_label.setScaledContents(False)
        container_layout.addWidget(self.video_label, stretch=1)
        
        layout.addWidget(container)
    
    def update_frame(self, frame: np.ndarray):
        """Update the displayed frame with high quality scaling."""
        # Convert BGR to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Get widget size
        label_size = self.video_label.size()
        
        # Original frame dimensions
        h, w = rgb_frame.shape[:2]
        
        # Calculate scale to fit widget while maintaining aspect ratio
        scale = min(label_size.width() / w, label_size.height() / h)
        new_w, new_h = int(w * scale), int(h * scale)
        
        if new_w > 0 and new_h > 0:
            # Use INTER_AREA for downscaling (better quality) or INTER_CUBIC for upscaling
            if scale < 1.0:
                # Downscaling - INTER_AREA gives best results
                resized = cv2.resize(rgb_frame, (new_w, new_h), interpolation=cv2.INTER_AREA)
            else:
                # Upscaling - INTER_CUBIC for better quality
                resized = cv2.resize(rgb_frame, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
            
            # Ensure contiguous array for QImage
            resized = np.ascontiguousarray(resized)
            
            # Convert to QImage
            bytes_per_line = 3 * new_w
            q_image = QImage(resized.data, new_w, new_h, bytes_per_line, QImage.Format.Format_RGB888)
            
            # Create a copy to avoid memory issues
            pixmap = QPixmap.fromImage(q_image.copy())
            
            # Set pixmap
            self.video_label.setPixmap(pixmap)
    
    def set_connected(self, connected: bool):
        """Update connection status."""
        if connected:
            self.status_label.setText("LIVE")
            self.status_label.setStyleSheet(f"color: {Theme.STATUS_ONLINE};")
            self.status_indicator.setStyleSheet(f"background-color: {Theme.STATUS_ONLINE};")
        else:
            self.status_label.setText("OFFLINE")
            self.status_label.setStyleSheet(f"color: {Theme.STATUS_OFFLINE};")
            self.status_indicator.setStyleSheet(f"background-color: {Theme.STATUS_OFFLINE};")
            self.video_label.setText("Connecting...")
    
    def set_status(self, status: str):
        """Set a custom status for the camera (e.g., DISABLED)."""
        if status == "DISABLED":
            self.status_label.setText("DISABLED")
            self.status_label.setStyleSheet(f"color: {Theme.TEXT_MUTED};")
            self.status_indicator.setStyleSheet(f"background-color: {Theme.TEXT_MUTED};")
            self.video_label.setText("Camera Disabled")
            self.video_label.setStyleSheet(f"background-color: {Theme.BG_DARKEST}; color: {Theme.TEXT_MUTED};")
    
    def set_error(self, error_msg: str):
        """Display error message."""
        self.status_label.setText("ERROR")
        self.status_label.setStyleSheet(f"color: {Theme.STATUS_OFFLINE};")
        self.status_indicator.setStyleSheet(f"background-color: {Theme.STATUS_OFFLINE};")


# ============================================================================
# Person Card Widget
# ============================================================================

class PersonCardWidget(QFrame):
    """Widget for displaying a single person in the sidebar."""
    
    clicked = pyqtSignal(int)  # person_id
    
    def __init__(self, person_info: PersonInfo, parent=None):
        super().__init__(parent)
        self.person_info = person_info
        self._setup_ui()
    
    def _setup_ui(self):
        """Setup the widget UI."""
        self.setFrameStyle(QFrame.Shape.NoFrame)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        
        # Determine accent color based on status
        if self.person_info.is_known:
            accent = Theme.ACCENT_BLUE
        elif self.person_info.is_confirmed:
            accent = Theme.ACCENT_GREEN
        else:
            accent = Theme.ACCENT_YELLOW
        
        self.setStyleSheet(f"""
            PersonCardWidget {{
                background-color: {Theme.BG_DARK};
                border: 1px solid {Theme.BORDER_DEFAULT};
                border-left: 3px solid {accent};
            }}
            PersonCardWidget:hover {{
                background-color: {Theme.BG_MEDIUM};
            }}
        """)
        
        layout = QHBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(10)
        
        # Face thumbnail
        self.thumbnail_label = QLabel()
        self.thumbnail_label.setFixedSize(48, 48)
        self.thumbnail_label.setStyleSheet(f"background-color: {Theme.BG_MEDIUM};")
        self._update_thumbnail()
        layout.addWidget(self.thumbnail_label)
        
        # Info layout
        info_layout = QVBoxLayout()
        info_layout.setSpacing(2)
        
        # Name
        if self.person_info.first_name or self.person_info.last_name:
            name = f"{self.person_info.first_name} {self.person_info.last_name}".strip()
        else:
            if self.person_info.is_confirmed:
                name = f"Person #{self.person_info.person_id}"
            else:
                name = f"Candidate #{abs(self.person_info.person_id)}"
        
        self.name_label = QLabel(name)
        self.name_label.setFont(QFont("Segoe UI", 10, QFont.Weight.Bold))
        self.name_label.setStyleSheet(f"color: {Theme.TEXT_PRIMARY};")
        info_layout.addWidget(self.name_label)
        
        # Status text
        if self.person_info.is_known:
            status = "Database Match"
        elif self.person_info.is_confirmed:
            status = "Confirmed"
        else:
            status = "Candidate"
        
        self.status_label = QLabel(status)
        self.status_label.setFont(QFont("Segoe UI", 9))
        self.status_label.setStyleSheet(f"color: {accent};")
        info_layout.addWidget(self.status_label)
        
        # Camera info
        cam_text = f"Camera {self.person_info.last_seen_camera + 1}"
        self.camera_label = QLabel(cam_text)
        self.camera_label.setFont(QFont("Segoe UI", 9))
        self.camera_label.setStyleSheet(f"color: {Theme.TEXT_MUTED};")
        info_layout.addWidget(self.camera_label)
        
        layout.addLayout(info_layout)
        layout.addStretch()
    
    def _update_thumbnail(self):
        """Update the face thumbnail with circular crop."""
        if self.person_info.face_thumbnail is not None:
            thumb = self.person_info.face_thumbnail
            
            # Resize to fit (make it square first for circular crop)
            size = 48
            thumb_resized = cv2.resize(thumb, (size, size), interpolation=cv2.INTER_LINEAR)
            rgb_thumb = cv2.cvtColor(thumb_resized, cv2.COLOR_BGR2RGB)
            
            h, w = rgb_thumb.shape[:2]
            bytes_per_line = 3 * w
            q_image = QImage(rgb_thumb.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
            
            # Create circular pixmap
            pixmap = QPixmap.fromImage(q_image)
            
            self.thumbnail_label.setPixmap(pixmap)
        else:
            self.thumbnail_label.setText("?")
            self.thumbnail_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
    
    def mousePressEvent(self, event):
        """Handle mouse click."""
        self.clicked.emit(self.person_info.person_id)
        super().mousePressEvent(event)
    
    def update_info(self, person_info: PersonInfo):
        """Update person information."""
        self.person_info = person_info
        self._update_thumbnail()
        
        if person_info.first_name or person_info.last_name:
            name = f"{person_info.first_name} {person_info.last_name}".strip()
        else:
            if person_info.is_confirmed:
                name = f"Person #{person_info.person_id}"
            else:
                name = f"Candidate #{abs(person_info.person_id)}"
        
        self.name_label.setText(name)


# ============================================================================
# Add Person Dialog - Modal for enrolling new persons with multiple photos
# ============================================================================

class AddPersonDialog(QDialog):
    """
    Modal dialog for adding a new person to the known faces database.
    Allows entering name, notes, and selecting multiple photos.
    """
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Add New Person")
        self.setModal(True)
        self.setMinimumSize(650, 550)
        self.resize(750, 650)
        
        # Store selected photos
        self._photo_paths: List[str] = []
        self._photo_thumbnails: Dict[str, QPixmap] = {}
        
        self._setup_ui()
        self._apply_styles()
    
    def _setup_ui(self):
        """Setup the dialog UI."""
        layout = QVBoxLayout(self)
        layout.setSpacing(20)
        layout.setContentsMargins(28, 28, 28, 28)
        
        # Title section
        title_section = QVBoxLayout()
        title_section.setSpacing(8)
        
        title = QLabel("Add New Person")
        title.setFont(QFont("Segoe UI", 20, QFont.Weight.Bold))
        title.setStyleSheet(f"color: {Theme.TEXT_PRIMARY};")
        title_section.addWidget(title)
        
        # Description
        desc = QLabel(
            "Add reference photos for reliable recognition. "
            "For best results, include 3-5 photos from different angles."
        )
        desc.setWordWrap(True)
        desc.setStyleSheet(f"color: {Theme.TEXT_SECONDARY}; font-size: 13px;")
        title_section.addWidget(desc)
        
        layout.addLayout(title_section)
        
        # Form section
        form_frame = QFrame()
        form_frame.setStyleSheet(f"""
            QFrame {{
                background-color: {Theme.BG_DARK};
                border: 1px solid {Theme.BORDER_DEFAULT};
            }}
        """)
        form_layout = QVBoxLayout(form_frame)
        form_layout.setContentsMargins(20, 20, 20, 20)
        form_layout.setSpacing(16)
        
        # First name
        fname_label = QLabel("First Name")
        fname_label.setStyleSheet(f"color: {Theme.TEXT_PRIMARY}; font-weight: 500;")
        form_layout.addWidget(fname_label)
        self.first_name_input = QLineEdit()
        self.first_name_input.setPlaceholderText("Enter first name...")
        self.first_name_input.textChanged.connect(self._validate_form)
        form_layout.addWidget(self.first_name_input)
        
        # Last name
        lname_label = QLabel("Last Name")
        lname_label.setStyleSheet(f"color: {Theme.TEXT_PRIMARY}; font-weight: 500;")
        form_layout.addWidget(lname_label)
        self.last_name_input = QLineEdit()
        self.last_name_input.setPlaceholderText("Enter last name...")
        self.last_name_input.textChanged.connect(self._validate_form)
        form_layout.addWidget(self.last_name_input)
        
        # Notes
        notes_label = QLabel("Notes (Optional)")
        notes_label.setStyleSheet(f"color: {Theme.TEXT_PRIMARY}; font-weight: 500;")
        form_layout.addWidget(notes_label)
        self.notes_input = QTextEdit()
        self.notes_input.setPlaceholderText("Additional information...")
        self.notes_input.setMaximumHeight(70)
        form_layout.addWidget(self.notes_input)
        
        layout.addWidget(form_frame)
        
        # Photos section
        photos_frame = QFrame()
        photos_frame.setStyleSheet(f"""
            QFrame {{
                background-color: {Theme.BG_DARK};
                border: 1px solid {Theme.BORDER_DEFAULT};
            }}
        """)
        photos_layout = QVBoxLayout(photos_frame)
        photos_layout.setContentsMargins(20, 20, 20, 20)
        photos_layout.setSpacing(12)
        
        # Photos header with buttons
        photos_header = QHBoxLayout()
        photos_title = QLabel("Reference Photos")
        photos_title.setFont(QFont("Segoe UI", 13, QFont.Weight.DemiBold))
        photos_title.setStyleSheet(f"color: {Theme.TEXT_PRIMARY};")
        photos_header.addWidget(photos_title)
        photos_header.addStretch()
        
        # Add photos button
        self.add_photos_btn = QPushButton("Add Photos")
        self.add_photos_btn.setStyleSheet(Theme.button_primary())
        self.add_photos_btn.clicked.connect(self._on_add_photos)
        photos_header.addWidget(self.add_photos_btn)
        
        # Remove selected button
        self.remove_photo_btn = QPushButton("Remove Selected")
        self.remove_photo_btn.setStyleSheet(Theme.button_secondary())
        self.remove_photo_btn.clicked.connect(self._on_remove_photos)
        self.remove_photo_btn.setEnabled(False)
        photos_header.addWidget(self.remove_photo_btn)
        
        photos_layout.addLayout(photos_header)
        
        # Photos list with thumbnails
        self.photos_list = QListWidget()
        self.photos_list.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        self.photos_list.setIconSize(QSize(64, 64))
        self.photos_list.setMinimumHeight(150)
        self.photos_list.itemSelectionChanged.connect(self._on_selection_changed)
        self.photos_list.setStyleSheet(f"""
            QListWidget {{
                background-color: {Theme.BG_MEDIUM};
                border: 1px solid #636e72;
                color: {Theme.TEXT_PRIMARY};
                border: 1px solid {Theme.BORDER_DEFAULT};
            }}
            QListWidget::item {{
                padding: 8px;
                margin: 2px;
            }}
            QListWidget::item:selected {{
                background-color: {Theme.ACCENT_BLUE}33;
            }}
        """)
        photos_layout.addWidget(self.photos_list)
        
        # Photo count label
        self.photo_count_label = QLabel("No photos added")
        self.photo_count_label.setStyleSheet(f"color: {Theme.ACCENT_YELLOW}; font-size: 12px;")
        photos_layout.addWidget(self.photo_count_label)
        
        layout.addWidget(photos_frame)
        
        # Buttons
        buttons_layout = QHBoxLayout()
        buttons_layout.setSpacing(12)
        buttons_layout.addStretch()
        
        self.cancel_btn = QPushButton("Cancel")
        self.cancel_btn.setStyleSheet(Theme.button_secondary())
        self.cancel_btn.clicked.connect(self.reject)
        buttons_layout.addWidget(self.cancel_btn)
        
        self.save_btn = QPushButton("Save Person")
        self.save_btn.setStyleSheet(Theme.button_success())
        self.save_btn.clicked.connect(self._on_save)
        self.save_btn.setEnabled(False)
        buttons_layout.addWidget(self.save_btn)
        
        layout.addLayout(buttons_layout)
    
    def _apply_styles(self):
        """Apply consistent styling."""
        self.setStyleSheet(Theme.get_main_stylesheet() + f"""
            QDialog {{
                background-color: {Theme.BG_DARKER};
            }}
        """)
    
    def _on_add_photos(self):
        """Handle adding photos."""
        file_paths, _ = QFileDialog.getOpenFileNames(
            self,
            "Select Photos",
            "",
            "Images (*.png *.jpg *.jpeg *.bmp);;All Files (*.*)"
        )
        
        for path in file_paths:
            if path not in self._photo_paths:
                self._photo_paths.append(path)
                self._add_photo_to_list(path)
        
        self._update_photo_count()
        self._validate_form()
    
    def _add_photo_to_list(self, path: str):
        """Add a photo to the list widget with thumbnail."""
        # Create thumbnail
        pixmap = QPixmap(path)
        if pixmap.isNull():
            return
        
        thumbnail = pixmap.scaled(64, 64, Qt.AspectRatioMode.KeepAspectRatio, 
                                   Qt.TransformationMode.SmoothTransformation)
        self._photo_thumbnails[path] = thumbnail
        
        # Create list item
        item = QListWidgetItem()
        item.setIcon(QIcon(thumbnail))
        item.setText(Path(path).name)
        item.setData(Qt.ItemDataRole.UserRole, path)
        self.photos_list.addItem(item)
    
    def _on_remove_photos(self):
        """Remove selected photos from the list."""
        selected_items = self.photos_list.selectedItems()
        for item in selected_items:
            path = item.data(Qt.ItemDataRole.UserRole)
            if path in self._photo_paths:
                self._photo_paths.remove(path)
            if path in self._photo_thumbnails:
                del self._photo_thumbnails[path]
            self.photos_list.takeItem(self.photos_list.row(item))
        
        self._update_photo_count()
        self._validate_form()
    
    def _on_selection_changed(self):
        """Handle selection change in photos list."""
        has_selection = len(self.photos_list.selectedItems()) > 0
        self.remove_photo_btn.setEnabled(has_selection)
    
    def _update_photo_count(self):
        """Update the photo count label."""
        count = len(self._photo_paths)
        if count == 0:
            self.photo_count_label.setText("Nie dodano żadnych zdjęć")
            self.photo_count_label.setStyleSheet("color: #f39c12; font-size: 11px;")
        elif count == 1:
            self.photo_count_label.setText("Dodano 1 zdjęcie (zalecane: 3-5)")
            self.photo_count_label.setStyleSheet(f"color: {Theme.ACCENT_YELLOW}; font-size: 12px;")
        elif count < 3:
            self.photo_count_label.setText(f"{count} photos added (recommended: 3-5)")
            self.photo_count_label.setStyleSheet(f"color: {Theme.ACCENT_YELLOW}; font-size: 12px;")
        else:
            self.photo_count_label.setText(f"{count} photos added")
            self.photo_count_label.setStyleSheet(f"color: {Theme.ACCENT_GREEN}; font-size: 12px;")
    
    def _validate_form(self):
        """Validate form and enable/disable save button."""
        has_name = bool(self.first_name_input.text().strip() or 
                        self.last_name_input.text().strip())
        has_photos = len(self._photo_paths) > 0
        
        self.save_btn.setEnabled(has_name and has_photos)
    
    def _on_save(self):
        """Handle save button click."""
        if not self._photo_paths:
            QMessageBox.warning(self, "No Photos", "Please add at least one photo.")
            return
        
        first_name = self.first_name_input.text().strip()
        last_name = self.last_name_input.text().strip()
        
        if not first_name and not last_name:
            QMessageBox.warning(self, "Missing Data", "Please enter a name.")
            return
        
        self.accept()
    
    def get_data(self) -> Tuple[str, str, str, List[str]]:
        """Get the entered data."""
        return (
            self.first_name_input.text().strip(),
            self.last_name_input.text().strip(),
            self.notes_input.toPlainText().strip(),
            self._photo_paths.copy()
        )


# ============================================================================
# Manage Database Dialog - View and delete persons from the database
# ============================================================================

class ManageDatabaseDialog(QDialog):
    """
    Modal dialog for managing the known faces database.
    Allows viewing list of persons and deleting them.
    """
    
    person_deleted = pyqtSignal(str)  # person_id
    
    def __init__(self, known_faces_manager, parent=None):
        super().__init__(parent)
        self.known_faces_manager = known_faces_manager
        self.setWindowTitle("Manage Database")
        self.setModal(True)
        self.setMinimumSize(750, 550)
        self.resize(900, 650)
        
        self._setup_ui()
        self._apply_styles()
        self._load_persons()
    
    def _setup_ui(self):
        """Setup the dialog UI."""
        layout = QVBoxLayout(self)
        layout.setSpacing(20)
        layout.setContentsMargins(28, 28, 28, 28)
        
        # Title section
        title_section = QVBoxLayout()
        title_section.setSpacing(8)
        
        title = QLabel("Database Management")
        title.setFont(QFont("Segoe UI", 20, QFont.Weight.Bold))
        title.setStyleSheet(f"color: {Theme.TEXT_PRIMARY};")
        title_section.addWidget(title)
        
        # Stats
        self.stats_label = QLabel("Loading...")
        self.stats_label.setStyleSheet(f"color: {Theme.TEXT_SECONDARY}; font-size: 13px;")
        title_section.addWidget(self.stats_label)
        
        layout.addLayout(title_section)
        
        # Main content area
        content_layout = QHBoxLayout()
        content_layout.setSpacing(20)
        
        # Left side: Persons list
        list_frame = QFrame()
        list_frame.setStyleSheet(f"""
            QFrame {{
                background-color: {Theme.BG_DARK};
                border: 1px solid {Theme.BORDER_DEFAULT};
            }}
        """)
        list_layout = QVBoxLayout(list_frame)
        list_layout.setContentsMargins(16, 16, 16, 16)
        list_layout.setSpacing(12)
        
        list_header = QLabel("Registered Persons")
        list_header.setFont(QFont("Segoe UI", 13, QFont.Weight.DemiBold))
        list_header.setStyleSheet(f"color: {Theme.TEXT_PRIMARY};")
        list_layout.addWidget(list_header)
        
        self.persons_list = QListWidget()
        self.persons_list.itemClicked.connect(self._on_person_selected)
        list_layout.addWidget(self.persons_list)
        
        content_layout.addWidget(list_frame, stretch=2)
        
        # Right side: Details panel
        details_frame = QFrame()
        details_frame.setStyleSheet(f"""
            QFrame {{
                background-color: {Theme.BG_DARK};
                border: 1px solid {Theme.BORDER_DEFAULT};
            }}
        """)
        details_layout = QVBoxLayout(details_frame)
        details_layout.setContentsMargins(20, 20, 20, 20)
        details_layout.setSpacing(16)
        
        details_header = QLabel("Details")
        details_header.setFont(QFont("Segoe UI", 13, QFont.Weight.DemiBold))
        details_header.setStyleSheet(f"color: {Theme.TEXT_PRIMARY};")
        details_layout.addWidget(details_header)
        
        # Thumbnail - square
        self.thumbnail_label = QLabel()
        self.thumbnail_label.setFixedSize(140, 140)
        self.thumbnail_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.thumbnail_label.setStyleSheet(f"""
            background-color: {Theme.BG_MEDIUM};
            border: 2px solid {Theme.BORDER_DEFAULT};
        """)
        details_layout.addWidget(self.thumbnail_label, alignment=Qt.AlignmentFlag.AlignCenter)
        
        # Person info
        self.info_label = QLabel("Select a person from the list")
        self.info_label.setStyleSheet(f"color: {Theme.TEXT_SECONDARY};")
        self.info_label.setWordWrap(True)
        self.info_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        details_layout.addWidget(self.info_label)
        
        # Photos count
        self.photos_label = QLabel("")
        self.photos_label.setStyleSheet(f"color: {Theme.ACCENT_CYAN};")
        self.photos_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        details_layout.addWidget(self.photos_label)
        
        # Notes
        self.notes_label = QLabel("")
        self.notes_label.setStyleSheet(f"color: {Theme.TEXT_MUTED}; font-style: italic;")
        self.notes_label.setWordWrap(True)
        self.notes_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        details_layout.addWidget(self.notes_label)
        
        details_layout.addStretch()
        
        # Delete button
        self.delete_btn = QPushButton("Delete Person")
        self.delete_btn.setStyleSheet(Theme.button_danger())
        self.delete_btn.setEnabled(False)
        self.delete_btn.clicked.connect(self._on_delete)
        details_layout.addWidget(self.delete_btn)
        
        content_layout.addWidget(details_frame, stretch=1)
        layout.addLayout(content_layout)
        
        # Bottom buttons
        buttons_layout = QHBoxLayout()
        buttons_layout.setSpacing(12)
        buttons_layout.addStretch()
        
        self.refresh_btn = QPushButton("Refresh")
        self.refresh_btn.setStyleSheet(Theme.button_primary())
        self.refresh_btn.clicked.connect(self._load_persons)
        buttons_layout.addWidget(self.refresh_btn)
        
        self.close_btn = QPushButton("Close")
        self.close_btn.setStyleSheet(Theme.button_secondary())
        self.close_btn.clicked.connect(self.close)
        buttons_layout.addWidget(self.close_btn)
        
        layout.addLayout(buttons_layout)
    
    def _apply_styles(self):
        """Apply dark theme to dialog."""
        self.setStyleSheet(Theme.get_main_stylesheet() + f"""
            QDialog {{
                background-color: {Theme.BG_DARKER};
            }}
        """)
    
    def _load_persons(self):
        """Load all persons from the database."""
        self.persons_list.clear()
        self._selected_person_id = None
        self.delete_btn.setEnabled(False)
        self._clear_details()
        
        if self.known_faces_manager is None:
            self.stats_label.setText("Database unavailable")
            return
        
        persons = self.known_faces_manager.get_all_persons()
        stats = self.known_faces_manager.get_stats()
        
        self.stats_label.setText(
            f"{stats['num_persons']} persons  ·  "
            f"{stats['total_photos']} photos  ·  "
            f"{stats['known_matches']} recognitions"
        )
        
        for person in sorted(persons, key=lambda p: p.full_name.lower()):
            item = QListWidgetItem()
            item.setData(Qt.ItemDataRole.UserRole, person.person_id)
            
            # Create display text
            if person.first_name or person.last_name:
                name = person.full_name
            else:
                name = person.person_id
            
            display_text = f"{name}\n{person.num_photos} photos"
            item.setText(display_text)
            
            self.persons_list.addItem(item)
    
    def _clear_details(self):
        """Clear the details panel."""
        self.thumbnail_label.clear()
        self.thumbnail_label.setText("No Photo")
        self.thumbnail_label.setStyleSheet(f"""
            background-color: {Theme.BG_MEDIUM};
            border: 2px solid {Theme.BORDER_DEFAULT};
            color: {Theme.TEXT_MUTED};
        """)
        self.info_label.setText("Select a person from the list")
        self.info_label.setStyleSheet(f"color: {Theme.TEXT_SECONDARY};")
        self.photos_label.setText("")
        self.notes_label.setText("")
    
    def _on_person_selected(self, item: QListWidgetItem):
        """Handle person selection."""
        person_id = item.data(Qt.ItemDataRole.UserRole)
        self._selected_person_id = person_id
        self.delete_btn.setEnabled(True)
        
        person = self.known_faces_manager.get_person(person_id)
        if person is None:
            self._clear_details()
            return
        
        # Update info
        name = person.full_name if (person.first_name or person.last_name) else person.person_id
        self.info_label.setText(f"<b style='font-size: 14px;'>{name}</b><br><span style='color: {Theme.TEXT_MUTED};'>ID: {person.person_id}</span>")
        self.info_label.setStyleSheet(f"color: {Theme.TEXT_PRIMARY};")
        
        self.photos_label.setText(f"{person.num_photos} photos  ·  {person.num_embeddings} embeddings")
        
        if person.notes:
            self.notes_label.setText(person.notes)
        else:
            self.notes_label.setText("")
        
        # Load first photo as thumbnail
        if person.photo_paths:
            photo_path = Path(self.known_faces_manager.base_dir) / person.photo_paths[0]
            if photo_path.exists():
                pixmap = QPixmap(str(photo_path))
                scaled = pixmap.scaled(
                    130, 130,
                    Qt.AspectRatioMode.KeepAspectRatio,
                    Qt.TransformationMode.SmoothTransformation
                )
                self.thumbnail_label.setPixmap(scaled)
            else:
                self.thumbnail_label.setText("Photo unavailable")
        else:
            self.thumbnail_label.setText("No Photo")
    
    def _on_delete(self):
        """Handle delete button click."""
        if self._selected_person_id is None:
            return
        
        person = self.known_faces_manager.get_person(self._selected_person_id)
        if person is None:
            return
        
        name = person.full_name if (person.first_name or person.last_name) else person.person_id
        
        reply = QMessageBox.question(
            self,
            "Confirm Deletion",
            f"Are you sure you want to delete:\n\n{name}\n\n"
            f"All {person.num_photos} photos will be removed.\n"
            f"This action cannot be undone.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No
        )
        
        if reply == QMessageBox.StandardButton.Yes:
            success = self.known_faces_manager.remove_person(self._selected_person_id)
            if success:
                self.person_deleted.emit(self._selected_person_id)
                QMessageBox.information(
                    self, "Success",
                    f"Deleted: {name}"
                )
                self._load_persons()
            else:
                QMessageBox.warning(
                    self, "Error",
                    "Failed to delete person."
                )


# ============================================================================
# Sidebar Panel
# ============================================================================

class SidebarPanel(QFrame):
    """Side panel with detected persons list and add client form."""
    
    person_selected = pyqtSignal(int)  # person_id
    client_added = pyqtSignal(int, str, str)  # person_id, first_name, last_name
    photo_uploaded = pyqtSignal(str)  # file_path
    add_new_person_requested = pyqtSignal()  # Request to open add person dialog
    manage_database_requested = pyqtSignal()  # Request to open manage database dialog
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self._person_cards: Dict[int, PersonCardWidget] = {}
        self._selected_person_id: Optional[int] = None
        self._setup_ui()
    
    def _setup_ui(self):
        """Setup the sidebar UI."""
        self.setMinimumWidth(300)
        self.setMaximumWidth(380)
        self.setStyleSheet(f"background-color: {Theme.BG_DARKER};")
        
        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 20, 16, 16)
        layout.setSpacing(16)
        
        # ===== Header Section =====
        header_layout = QHBoxLayout()
        
        persons_header = QLabel("Detected Faces")
        persons_header.setFont(QFont("Segoe UI", 11, QFont.Weight.Bold))
        persons_header.setStyleSheet(f"color: {Theme.TEXT_SECONDARY};")
        header_layout.addWidget(persons_header)
        header_layout.addStretch()
        
        layout.addLayout(header_layout)
        
        # Scroll area for persons
        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.scroll_area.setStyleSheet(f"""
            QScrollArea {{
                border: none;
                background-color: transparent;
            }}
        """)
        
        self.persons_container = QWidget()
        self.persons_container.setStyleSheet("background-color: transparent;")
        self.persons_layout = QVBoxLayout(self.persons_container)
        self.persons_layout.setContentsMargins(0, 0, 8, 0)
        self.persons_layout.setSpacing(8)
        self.persons_layout.addStretch()
        
        self.scroll_area.setWidget(self.persons_container)
        layout.addWidget(self.scroll_area, stretch=1)
        
        # ===== Stats Section =====
        self.stats_label = QLabel("Confirmed: 0  ·  Candidates: 0")
        self.stats_label.setStyleSheet(f"color: {Theme.TEXT_MUTED}; font-size: 11px;")
        self.stats_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.stats_label)
        
        # ===== Actions Section =====
        actions_frame = QFrame()
        actions_frame.setStyleSheet(f"""
            QFrame {{
                background-color: {Theme.BG_DARK};
                border: 1px solid {Theme.BORDER_DEFAULT};
            }}
        """)
        actions_layout = QVBoxLayout(actions_frame)
        actions_layout.setContentsMargins(16, 16, 16, 16)
        actions_layout.setSpacing(10)
        
        # === ADD NEW PERSON BUTTON (opens modal) ===
        self.add_new_person_btn = QPushButton("Add New Person")
        self.add_new_person_btn.setStyleSheet(Theme.button_primary())
        self.add_new_person_btn.clicked.connect(self._on_add_new_person)
        actions_layout.addWidget(self.add_new_person_btn)
        
        # === MANAGE DATABASE BUTTON ===
        self.manage_db_btn = QPushButton("Manage Database")
        self.manage_db_btn.setStyleSheet(Theme.button_secondary())
        self.manage_db_btn.clicked.connect(self._on_manage_database)
        actions_layout.addWidget(self.manage_db_btn)
        
        # Separator
        separator = QFrame()
        separator.setFrameShape(QFrame.Shape.HLine)
        separator.setStyleSheet(f"background-color: {Theme.BORDER_DEFAULT};")
        separator.setFixedHeight(1)
        actions_layout.addWidget(separator)
        
        # === ASSIGN NAME TO DETECTED PERSON ===
        assign_header = QLabel("Quick Assign")
        assign_header.setFont(QFont("Segoe UI", 9, QFont.Weight.Bold))
        assign_header.setStyleSheet(f"color: {Theme.TEXT_MUTED};")
        actions_layout.addWidget(assign_header)
        
        # Selected person indicator
        self.selected_label = QLabel("Select a person above")
        self.selected_label.setStyleSheet(f"color: {Theme.ACCENT_YELLOW}; font-size: 11px;")
        actions_layout.addWidget(self.selected_label)
        
        # First name input
        self.first_name_input = QLineEdit()
        self.first_name_input.setPlaceholderText("First name")
        actions_layout.addWidget(self.first_name_input)
        
        # Last name input
        self.last_name_input = QLineEdit()
        self.last_name_input.setPlaceholderText("Last name")
        actions_layout.addWidget(self.last_name_input)
        
        # Buttons layout
        buttons_layout = QHBoxLayout()
        
        # Add client button (assigns name to selected detected person)
        self.add_button = QPushButton("Assign Name")
        self.add_button.setStyleSheet(Theme.button_success())
        self.add_button.clicked.connect(self._on_add_client)
        self.add_button.setEnabled(False)
        buttons_layout.addWidget(self.add_button)
        
        actions_layout.addLayout(buttons_layout)
        layout.addWidget(actions_frame)
    
    def add_or_update_person(self, person_info: PersonInfo):
        """Add a new person or update existing one."""
        person_id = person_info.person_id
        
        if person_id in self._person_cards:
            # Update existing card
            self._person_cards[person_id].update_info(person_info)
        else:
            # Create new card
            card = PersonCardWidget(person_info)
            card.clicked.connect(self._on_person_clicked)
            
            # Insert before the stretch
            self.persons_layout.insertWidget(
                self.persons_layout.count() - 1, card
            )
            self._person_cards[person_id] = card
    
    def remove_person(self, person_id: int):
        """Remove a person from the list."""
        if person_id in self._person_cards:
            card = self._person_cards.pop(person_id)
            card.deleteLater()
            
            if self._selected_person_id == person_id:
                self._selected_person_id = None
                self._update_selection_ui()
    
    def update_stats(self, confirmed: int, candidates: int, known_persons: int = 0):
        """Update statistics label."""
        if known_persons > 0:
            self.stats_label.setText(f"Database: {known_persons}  ·  Active: {confirmed}  ·  Candidates: {candidates}")
        else:
            self.stats_label.setText(f"Confirmed: {confirmed}  ·  Candidates: {candidates}")
    
    def _on_person_clicked(self, person_id: int):
        """Handle person card click."""
        self._selected_person_id = person_id
        self._update_selection_ui()
        self.person_selected.emit(person_id)
    
    def _update_selection_ui(self):
        """Update UI to reflect current selection."""
        if self._selected_person_id is not None:
            card = self._person_cards.get(self._selected_person_id)
            if card:
                info = card.person_info
                if info.first_name or info.last_name:
                    name = f"{info.first_name} {info.last_name}".strip()
                else:
                    name = f"Person #{info.person_id}" if info.is_confirmed else f"Candidate #{abs(info.person_id)}"
                self.selected_label.setText(f"Selected: {name}")
                self.selected_label.setStyleSheet(f"color: {Theme.ACCENT_GREEN}; font-size: 11px;")
                self.add_button.setEnabled(True)
            else:
                self._selected_person_id = None
                self.selected_label.setText("Select a person above")
                self.selected_label.setStyleSheet(f"color: {Theme.ACCENT_YELLOW}; font-size: 11px;")
                self.add_button.setEnabled(False)
        else:
            self.selected_label.setText("Select a person above")
            self.selected_label.setStyleSheet(f"color: {Theme.ACCENT_YELLOW}; font-size: 11px;")
            self.add_button.setEnabled(False)
    
    def _on_add_client(self):
        """Handle add client button click."""
        if self._selected_person_id is None:
            return
        
        first_name = self.first_name_input.text().strip()
        last_name = self.last_name_input.text().strip()
        
        if not first_name and not last_name:
            QMessageBox.warning(
                self, "Error",
                "Please enter a first or last name."
            )
            return
        
        self.client_added.emit(self._selected_person_id, first_name, last_name)
        
        # Clear inputs
        self.first_name_input.clear()
        self.last_name_input.clear()
    
    def _on_add_new_person(self):
        """Handle add new person button click - emits signal to open dialog."""
        self.add_new_person_requested.emit()
    
    def _on_manage_database(self):
        """Handle manage database button click - emits signal to open dialog."""
        self.manage_database_requested.emit()


# ============================================================================
# Main Window
# ============================================================================

class MainWindow(QMainWindow):
    """Main application window."""
    
    def __init__(self):
        super().__init__()
        
        # Initialize AI components
        self.config = Config()
        
        # Only use cameras that are actually configured (don't extend to 4)
        self.num_active_cameras = len(self.config.CAMERA_URLS)
        logger.info(f"Configured cameras: {self.num_active_cameras}")
        
        logger.info("Loading AI models...")
        self.detector = RobustFaceDetector(self.config)
        self.embedding_engine = RobustFaceEmbeddingEngine(self.config)
        self.gallery = RobustFaceGallery(self.config)
        logger.info("AI models loaded successfully")
        
        # Initialize known faces manager for reference photo database
        self.known_faces_manager = None
        if KNOWN_FACES_AVAILABLE:
            logger.info("Initializing known faces database...")
            self.known_faces_manager = create_known_faces_manager(
                base_dir=str(Path.cwd()),
                embedding_engine=self.embedding_engine
            )
            # Load embeddings for all known faces
            num_embeddings = self.known_faces_manager.load_all_embeddings()
            logger.info(f"Loaded {num_embeddings} embeddings from known faces database")
            
            # Connect known faces manager to gallery
            self.gallery.set_known_faces_manager(self.known_faces_manager)
        else:
            logger.warning("Known faces database not available")
        
        # Person name mapping (for dynamically added names)
        self._person_names: Dict[int, Tuple[str, str]] = {}  # person_id -> (first, last)
        
        # Face crops for enrollment (camera_id -> last detected face crops)
        self._last_face_crops: Dict[int, List[np.ndarray]] = {}
        
        # Frame grabbers (one per camera)
        self.frame_grabbers: List[Optional[FrameGrabber]] = []
        
        # AI Processor (single thread for all cameras)
        self.ai_processor: Optional[AIProcessor] = None
        
        # Setup UI
        self._setup_ui()
        self._setup_camera_system()
        
        # Setup update timer for stats
        self.stats_timer = QTimer(self)
        self.stats_timer.timeout.connect(self._update_stats)
        self.stats_timer.start(1000)  # Update every second
    
    def _setup_ui(self):
        """Setup the main window UI."""
        self.setWindowTitle("Face Recognition System")
        self.setMinimumSize(1600, 900)
        self.resize(1920, 1080)
        self.setStyleSheet(Theme.get_main_stylesheet() + f"""
            QMainWindow {{
                background-color: {Theme.BG_DARKEST};
            }}
        """)
        
        # Central widget
        central_widget = QWidget()
        central_widget.setStyleSheet(f"background-color: {Theme.BG_DARKEST};")
        self.setCentralWidget(central_widget)
        
        # Main horizontal layout
        main_layout = QHBoxLayout(central_widget)
        main_layout.setContentsMargins(8, 8, 8, 8)
        main_layout.setSpacing(8)
        
        # ===== Left side: Camera Grid =====
        camera_container = QWidget()
        camera_container.setStyleSheet("background-color: transparent;")
        camera_layout = QGridLayout(camera_container)
        camera_layout.setSpacing(8)
        
        # Force equal space distribution for all 4 cells
        camera_layout.setRowStretch(0, 1)
        camera_layout.setRowStretch(1, 1)
        camera_layout.setColumnStretch(0, 1)
        camera_layout.setColumnStretch(1, 1)
        
        # Create 4 camera widgets in 2x2 grid
        self.camera_widgets: List[CameraWidget] = []
        for i in range(4):
            camera_widget = CameraWidget(i)
            row, col = i // 2, i % 2
            camera_layout.addWidget(camera_widget, row, col)
            self.camera_widgets.append(camera_widget)
        
        main_layout.addWidget(camera_container, stretch=3)
        
        # ===== Right side: Sidebar =====
        self.sidebar = SidebarPanel()
        self.sidebar.person_selected.connect(self._on_person_selected)
        self.sidebar.client_added.connect(self._on_client_added)
        self.sidebar.add_new_person_requested.connect(self._on_add_new_person_requested)
        self.sidebar.manage_database_requested.connect(self._on_manage_database_requested)
        main_layout.addWidget(self.sidebar)
    
    def _setup_camera_system(self):
        """Create frame grabbers and AI processor with new architecture."""
        # Create frame grabbers for each active camera
        for i in range(4):
            if i < self.num_active_cameras:
                url = self.config.CAMERA_URLS[i]
                grabber = FrameGrabber(camera_id=i, url=url, config=self.config)
                grabber.start()
                self.frame_grabbers.append(grabber)
                logger.info(f"Started frame grabber {i}")
            else:
                self.frame_grabbers.append(None)
                self.camera_widgets[i].set_status("DISABLED")
                logger.info(f"Camera {i} is disabled")
        
        # Give grabbers time to connect
        time.sleep(0.5)
        
        # Create single AI processor for all cameras
        active_grabbers = [g for g in self.frame_grabbers if g is not None]
        self.ai_processor = AIProcessor(
            grabbers=active_grabbers,
            config=self.config,
            detector=self.detector,
            embedding_engine=self.embedding_engine,
            gallery=self.gallery
        )
        
        # Connect AI processor signals
        self.ai_processor.frames_ready.connect(self._on_frames_ready)
        self.ai_processor.detections_ready.connect(self._on_detections_ready)
        self.ai_processor.connection_status.connect(self._on_connection_status)
        
        self.ai_processor.start()
        logger.info("Started AI processor (batched processing for all cameras)")
    
    @pyqtSlot(dict)
    def _on_frames_ready(self, frames: Dict[int, np.ndarray]):
        """Handle synchronized frames from all cameras."""
        for camera_id, frame in frames.items():
            if 0 <= camera_id < len(self.camera_widgets):
                self.camera_widgets[camera_id].update_frame(frame)
    
    @pyqtSlot(int, list)
    def _on_detections_ready(self, camera_id: int, detections: List[FaceDetection]):
        """Handle new detections from camera worker."""
        for detection in detections:
            if detection.person_id is not None and detection.face_crop is not None:
                # Check if this is a known person from the reference database
                is_known = self.gallery.is_known_person(detection.person_id)
                known_name = self.gallery.get_known_person_name(detection.person_id)
                
                if known_name:
                    first_name, last_name = known_name
                else:
                    # Get manually assigned name if any
                    first_name, last_name = self._person_names.get(
                        detection.person_id, ("", "")
                    )
                
                person_info = PersonInfo(
                    person_id=detection.person_id,
                    is_confirmed=detection.is_confirmed,
                    face_thumbnail=detection.face_crop.copy(),
                    first_name=first_name,
                    last_name=last_name,
                    last_seen_camera=camera_id,
                    last_seen_time=time.time(),
                    is_known=is_known
                )
                
                self.sidebar.add_or_update_person(person_info)
    
    @pyqtSlot(int, bool)
    def _on_connection_status(self, camera_id: int, connected: bool):
        """Handle camera connection status change."""
        if 0 <= camera_id < len(self.camera_widgets):
            self.camera_widgets[camera_id].set_connected(connected)
    
    @pyqtSlot(int, str)
    def _on_error_occurred(self, camera_id: int, error_msg: str):
        """Handle camera error."""
        logger.error(f"Camera {camera_id} error: {error_msg}")
        if 0 <= camera_id < len(self.camera_widgets):
            self.camera_widgets[camera_id].set_error(error_msg)
    
    @pyqtSlot(int)
    def _on_person_selected(self, person_id: int):
        """Handle person selection in sidebar."""
        logger.info(f"Person selected: {person_id}")
    
    @pyqtSlot(int, str, str)
    def _on_client_added(self, person_id: int, first_name: str, last_name: str):
        """Handle client addition - also adds to known faces database."""
        logger.info(f"Adding client: {first_name} {last_name} to person {person_id}")
        
        # Store the name mapping for GUI
        self._person_names[person_id] = (first_name, last_name)
        
        # If we have a known faces manager, add the person to the reference database
        if self.known_faces_manager is not None:
            # Get the face thumbnail from the sidebar
            face_crops = []
            cards = self.sidebar._person_cards
            if person_id in cards:
                card = cards[person_id]
                if card.person_info.face_thumbnail is not None:
                    face_crops.append(card.person_info.face_thumbnail)
            
            if face_crops:
                # Add as a new known person in the reference database
                known_person = self.known_faces_manager.add_person(
                    first_name=first_name,
                    last_name=last_name,
                    face_crops=face_crops
                )
                if known_person is not None:
                    logger.info(f"Added {first_name} {last_name} to known faces database as {known_person.person_id}")
                    QMessageBox.information(
                        self,
                        "Sukces",
                        f"Zapisano klienta do bazy danych:\n{first_name} {last_name}\n\n"
                        f"Osoba zostanie rozpoznawana automatycznie przy kolejnych wizytach."
                    )
                else:
                    logger.warning("Failed to add person to known faces database")
                    QMessageBox.warning(
                        self,
                        "Ostrzeżenie",
                        f"Przypisano dane, ale nie udało się zapisać do bazy danych."
                    )
            else:
                QMessageBox.information(
                    self,
                    "Sukces",
                    f"Przypisano dane klienta:\n{first_name} {last_name}\n\n"
                    f"Brak zdjęcia do zapisania w bazie."
                )
        else:
            QMessageBox.information(
                self,
                "Sukces",
                f"Przypisano dane klienta:\n{first_name} {last_name}"
            )
        
        # Update the sidebar card if it exists
        cards = self.sidebar._person_cards
        if person_id in cards:
            card = cards[person_id]
            card.person_info.first_name = first_name
            card.person_info.last_name = last_name
            card.update_info(card.person_info)
    
    @pyqtSlot()
    def _on_add_new_person_requested(self):
        """Handle request to add a new person - opens the AddPersonDialog."""
        dialog = AddPersonDialog(self)
        
        if dialog.exec() == QDialog.DialogCode.Accepted:
            first_name, last_name, notes, photo_paths = dialog.get_data()
            
            if not photo_paths:
                QMessageBox.warning(self, "Błąd", "Nie dodano żadnych zdjęć.")
                return
            
            if self.known_faces_manager is None:
                QMessageBox.warning(
                    self, "Błąd",
                    "Baza danych znanych osób nie jest dostępna."
                )
                return
            
            try:
                # Process photos and detect faces
                face_crops = []
                failed_photos = []
                
                for photo_path in photo_paths:
                    image = cv2.imread(photo_path)
                    if image is None:
                        failed_photos.append(Path(photo_path).name)
                        continue
                    
                    # Detect faces
                    detections = self.detector.detect(image)
                    reid_detections, _ = self.detector.filter_for_reid(detections)
                    
                    if reid_detections:
                        # Use the largest face
                        best_detection = max(reid_detections, key=lambda d: d.area)
                        face_crops.append(best_detection.face_crop)
                    else:
                        failed_photos.append(Path(photo_path).name)
                
                if not face_crops:
                    QMessageBox.warning(
                        self, "Błąd",
                        f"Nie wykryto twarzy na żadnym ze zdjęć.\n\n"
                        f"Upewnij się, że:\n"
                        f"- Twarze są dobrze widoczne\n"
                        f"- Zdjęcia są dobrej jakości\n"
                        f"- Twarze nie są zbyt małe"
                    )
                    return
                
                # Add person to known faces database
                person = self.known_faces_manager.add_person(
                    first_name=first_name,
                    last_name=last_name,
                    notes=notes,
                    photo_paths=photo_paths,
                    face_crops=face_crops
                )
                
                if person is None:
                    QMessageBox.warning(self, "Błąd", "Nie udało się dodać osoby.")
                    return
                
                # Reload embeddings for the new person
                self.known_faces_manager.load_all_embeddings()
                
                # Show success message
                success_msg = f"Dodano nową osobę do bazy danych!\n\n"
                success_msg += f"Imię: {first_name}\n"
                success_msg += f"Nazwisko: {last_name}\n"
                success_msg += f"ID: {person.person_id}\n"
                success_msg += f"Zdjęć: {len(face_crops)}"
                
                if failed_photos:
                    success_msg += f"\n\nNie wykryto twarzy w:\n" + "\n".join(failed_photos[:5])
                    if len(failed_photos) > 5:
                        success_msg += f"\n...i {len(failed_photos) - 5} innych"
                
                QMessageBox.information(self, "Sukces", success_msg)
                
                logger.info(f"Added new person: {person.full_name} with {len(face_crops)} photos")
                
            except Exception as e:
                logger.error(f"Error adding person: {e}")
                QMessageBox.warning(self, "Błąd", f"Błąd podczas dodawania: {str(e)}")
    
    @pyqtSlot()
    def _on_manage_database_requested(self):
        """Handle request to manage database - opens the ManageDatabaseDialog."""
        if self.known_faces_manager is None:
            QMessageBox.warning(
                self, "Błąd",
                "Baza danych znanych osób nie jest dostępna."
            )
            return
        
        dialog = ManageDatabaseDialog(self.known_faces_manager, self)
        dialog.person_deleted.connect(self._on_person_deleted_from_db)
        dialog.exec()
    
    @pyqtSlot(str)
    def _on_person_deleted_from_db(self, person_id: str):
        """Handle person deleted from database - refresh gallery mapping."""
        logger.info(f"Person {person_id} deleted from database")
        # Reload embeddings to update the internal state
        if self.known_faces_manager is not None:
            self.known_faces_manager.load_all_embeddings()
    
    def _update_stats(self):
        """Update gallery statistics in sidebar."""
        stats = self.gallery.get_stats()
        self.sidebar.update_stats(
            stats['confirmed_faces'],
            stats['candidates'],
            stats.get('known_persons', 0)
        )
    
    def closeEvent(self, event):
        """Handle window close event."""
        logger.info("Closing application...")
        
        # Stop AI processor
        if self.ai_processor is not None:
            self.ai_processor.stop()
            self.ai_processor.wait(2000)
        
        # Stop all frame grabbers
        for grabber in self.frame_grabbers:
            if grabber is not None:
                grabber.stop()
        
        # Wait for grabbers to finish
        for grabber in self.frame_grabbers:
            if grabber is not None:
                grabber.join(timeout=2.0)
        
        logger.info("Application closed")
        event.accept()


# ============================================================================
# Entry Point
# ============================================================================

def main():
    """Main entry point."""
    print("=" * 70)
    print("Multi-Camera Face Re-identification System - PyQt6 GUI")
    print("=" * 70)
    print()
    
    # Create application
    app = QApplication(sys.argv)
    
    # Set application style
    app.setStyle("Fusion")
    
    # Dark palette
    palette = QPalette()
    palette.setColor(QPalette.ColorRole.Window, QColor(15, 15, 26))
    palette.setColor(QPalette.ColorRole.WindowText, QColor(255, 255, 255))
    palette.setColor(QPalette.ColorRole.Base, QColor(30, 39, 46))
    palette.setColor(QPalette.ColorRole.AlternateBase, QColor(45, 52, 54))
    palette.setColor(QPalette.ColorRole.ToolTipBase, QColor(255, 255, 255))
    palette.setColor(QPalette.ColorRole.ToolTipText, QColor(255, 255, 255))
    palette.setColor(QPalette.ColorRole.Text, QColor(255, 255, 255))
    palette.setColor(QPalette.ColorRole.Button, QColor(45, 52, 54))
    palette.setColor(QPalette.ColorRole.ButtonText, QColor(255, 255, 255))
    palette.setColor(QPalette.ColorRole.BrightText, QColor(255, 0, 0))
    palette.setColor(QPalette.ColorRole.Link, QColor(52, 152, 219))
    palette.setColor(QPalette.ColorRole.Highlight, QColor(52, 152, 219))
    palette.setColor(QPalette.ColorRole.HighlightedText, QColor(255, 255, 255))
    app.setPalette(palette)
    
    # Create and show main window
    window = MainWindow()
    window.show()
    
    # Run application
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
