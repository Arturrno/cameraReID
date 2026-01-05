"""
PyQt6 Multi-Camera Face Re-identification GUI Application
==========================================================
A modern desktop application with:
- 2x2 camera grid layout
- Side panel with detected persons
- Add client functionality
- Multi-threaded AI processing

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

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QGridLayout, QLabel, QLineEdit, QPushButton, QScrollArea,
    QFrame, QSplitter, QFileDialog, QMessageBox, QSizePolicy
)
from PyQt6.QtCore import (
    Qt, QThread, pyqtSignal, pyqtSlot, QTimer, QSize, QMutex, QMutexLocker
)
from PyQt6.QtGui import QImage, QPixmap, QFont, QIcon, QPalette, QColor

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

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


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


# ============================================================================
# Camera Processing Thread
# ============================================================================

class CameraWorker(QThread):
    """
    Worker thread for processing a single camera stream.
    Emits signals with processed frames and detection info.
    """
    
    # Signals
    frame_ready = pyqtSignal(int, np.ndarray)  # camera_id, processed_frame
    detections_ready = pyqtSignal(int, list)   # camera_id, list of FaceDetection
    error_occurred = pyqtSignal(int, str)      # camera_id, error_message
    connection_status = pyqtSignal(int, bool)  # camera_id, is_connected
    
    def __init__(self, camera_id: int, url: str, config: Config,
                 detector: RobustFaceDetector,
                 embedding_engine: RobustFaceEmbeddingEngine,
                 gallery: RobustFaceGallery,
                 parent=None):
        super().__init__(parent)
        
        self.camera_id = camera_id
        self.url = url
        self.config = config
        self.detector = detector
        self.embedding_engine = embedding_engine
        self.gallery = gallery
        
        self._running = False
        self._paused = False
        self._mutex = QMutex()
        self._cap: Optional[cv2.VideoCapture] = None
        self._frame_count = 0
        self._reconnect_attempts = 0
        
        self.logger = logging.getLogger(f"CameraWorker-{camera_id}")
    
    def _connect(self) -> bool:
        """Attempt to connect to the camera stream with hardware decoding."""
        try:
            self.logger.info(f"Connecting to camera {self.camera_id}...")
            
            if self._cap is not None:
                self._cap.release()
            
            # Set environment variable for FFmpeg to use TCP transport
            # This MUST be set before opening the capture
            import os
            os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp"
            
            # Open with FFmpeg backend
            self._cap = cv2.VideoCapture(self.url, cv2.CAP_FFMPEG)
            
            # Minimal buffer to reduce latency
            self._cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            self._cap.set(cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, 15000)
            self._cap.set(cv2.CAP_PROP_READ_TIMEOUT_MSEC, 5000)  # Shorter timeout
            
            # Enable hardware acceleration (DXVA on Windows)
            self._cap.set(cv2.CAP_PROP_HW_ACCELERATION, cv2.VIDEO_ACCELERATION_ANY)
            
            if self._cap.isOpened():
                # Log actual resolution being received
                width = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                fps = self._cap.get(cv2.CAP_PROP_FPS)
                
                # Check if HW acceleration is active
                hw_accel = self._cap.get(cv2.CAP_PROP_HW_ACCELERATION)
                hw_str = "HW" if hw_accel > 0 else "SW"
                
                self._reconnect_attempts = 0
                self.connection_status.emit(self.camera_id, True)
                self.logger.info(f"Camera {self.camera_id} connected: {width}x{height} @ {fps:.1f}fps ({hw_str} decode)")
                return True
            else:
                self.connection_status.emit(self.camera_id, False)
                self.logger.error(f"Failed to open camera {self.camera_id}")
                return False
                
        except Exception as e:
            self.logger.error(f"Connection error for camera {self.camera_id}: {e}")
            self.connection_status.emit(self.camera_id, False)
            return False
    
    def _process_frame(self, frame: np.ndarray) -> Tuple[np.ndarray, List[FaceDetection]]:
        """Process a single frame through the AI pipeline."""
        try:
            # Update camera brightness statistics for adaptive preprocessing
            self.embedding_engine.update_camera_stats(self.camera_id, frame)
            
            # Detect all faces
            all_detections = self.detector.detect(frame)
            
            # Filter by size for ReID
            reid_detections, small_detections = self.detector.filter_for_reid(all_detections)
            
            # Track list for IOU updates
            current_tracks: List[Tuple[int, Tuple[int, int, int, int]]] = []
            
            if reid_detections:
                # Extract embeddings (with enhanced preprocessing)
                face_crops = [d.face_crop for d in reid_detections]
                embeddings = self.embedding_engine.extract_embeddings_batch(face_crops, self.camera_id)
                
                # Match and update gallery
                for detection, embedding in zip(reid_detections, embeddings):
                    detection.embedding = embedding
                    person_id, is_confirmed = self.gallery.match_and_update(
                        embedding, detection.bbox, self.camera_id
                    )
                    detection.person_id = person_id
                    detection.is_confirmed = is_confirmed
                    
                    # Add to track list for IOU memory
                    if is_confirmed:
                        current_tracks.append((person_id, detection.bbox))
            
            # Update IOU track memory
            self.gallery.update_tracks(self.camera_id, current_tracks)
            
            # Draw detections on frame
            processed_frame = self._draw_detections(frame, reid_detections, small_detections)
            
            return processed_frame, reid_detections
            
        except Exception as e:
            self.logger.error(f"Frame processing error: {e}")
            # Return original frame if processing fails
            return frame, []
    
    def _draw_detections(self, frame: np.ndarray, 
                         reid_detections: List[FaceDetection],
                         small_detections: List[FaceDetection]) -> np.ndarray:
        """Draw bounding boxes and labels on frame."""
        COLORS = [
            (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0),
            (255, 0, 255), (0, 255, 255), (128, 0, 0), (0, 128, 0),
        ]
        CANDIDATE_COLOR = (128, 128, 128)
        TOO_SMALL_COLOR = (100, 100, 100)
        
        # Draw ReID detections
        for detection in reid_detections:
            x1, y1, x2, y2 = detection.bbox
            person_id = detection.person_id or 0
            
            if detection.is_confirmed:
                color = COLORS[person_id % len(COLORS)]
                label = f"ID: {person_id}"
            else:
                color = CANDIDATE_COLOR
                label = f"ID: ? ({abs(person_id)})"
            
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            
            (label_w, label_h), _ = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
            )
            cv2.rectangle(frame, (x1, y1 - label_h - 10), 
                         (x1 + label_w + 5, y1), color, -1)
            cv2.putText(frame, label, (x1 + 2, y1 - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Draw small detections
        for detection in small_detections:
            x1, y1, x2, y2 = detection.bbox
            cv2.rectangle(frame, (x1, y1), (x2, y2), TOO_SMALL_COLOR, 1)
        
        return frame
    
    def run(self):
        """Main processing loop."""
        self._running = True
        connected = False
        
        while self._running:
            # Check if paused
            with QMutexLocker(self._mutex):
                if self._paused:
                    self.msleep(100)
                    continue
            
            # Try to connect if not connected
            if not connected:
                if self._reconnect_attempts >= self.config.MAX_RECONNECT_ATTEMPTS:
                    self.error_occurred.emit(
                        self.camera_id, 
                        f"Max reconnection attempts ({self.config.MAX_RECONNECT_ATTEMPTS}) reached"
                    )
                    break
                
                self._reconnect_attempts += 1
                connected = self._connect()
                
                if not connected:
                    self.msleep(int(self.config.RECONNECT_DELAY * 1000))
                    continue
            
            try:
                ret, frame = self._cap.read()
                
                if not ret or frame is None:
                    self.logger.warning(f"Frame read failed for camera {self.camera_id}")
                    connected = False
                    self.connection_status.emit(self.camera_id, False)
                    continue
                
                # Validate frame (skip corrupted frames from H.264 decode errors)
                if frame.size == 0 or np.mean(frame) < 1.0:
                    # Likely a corrupted/black frame, skip it
                    continue
                
                self._frame_count += 1
                
                # Skip frames for performance
                if self._frame_count % self.config.FRAME_SKIP != 0:
                    continue
                
                # Process frame
                processed_frame, detections = self._process_frame(frame)
                
                # Emit signals
                self.frame_ready.emit(self.camera_id, processed_frame)
                if detections:
                    self.detections_ready.emit(self.camera_id, detections)
                
            except Exception as e:
                self.logger.error(f"Processing error for camera {self.camera_id}: {e}")
                self.error_occurred.emit(self.camera_id, str(e))
                connected = False
        
        # Cleanup
        if self._cap is not None:
            self._cap.release()
        
        self.logger.info(f"Camera {self.camera_id} worker stopped")
    
    def stop(self):
        """Stop the worker thread."""
        self._running = False
    
    def pause(self):
        """Pause processing."""
        with QMutexLocker(self._mutex):
            self._paused = True
    
    def resume(self):
        """Resume processing."""
        with QMutexLocker(self._mutex):
            self._paused = False


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
        self.setFrameStyle(QFrame.Shape.Box | QFrame.Shadow.Sunken)
        self.setLineWidth(2)
        
        layout = QVBoxLayout(self)
        layout.setContentsMargins(2, 2, 2, 2)
        layout.setSpacing(2)
        
        # Camera label (title)
        self.title_label = QLabel(f"Kamera {self.camera_id + 1}")
        self.title_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.title_label.setFont(QFont("Arial", 10, QFont.Weight.Bold))
        self.title_label.setStyleSheet("color: white; background-color: #2c3e50; padding: 5px;")
        layout.addWidget(self.title_label)
        
        # Video display label
        self.video_label = QLabel()
        self.video_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.video_label.setMinimumSize(480, 360)  # Higher minimum for better quality
        self.video_label.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.video_label.setStyleSheet("background-color: #1a1a2e;")
        self.video_label.setText("Łączenie...")
        self.video_label.setScaledContents(False)  # We handle scaling manually
        layout.addWidget(self.video_label)
        
        # Status label
        self.status_label = QLabel("Status: Rozłączony")
        self.status_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.status_label.setStyleSheet("color: #e74c3c; font-size: 10px;")
        layout.addWidget(self.status_label)
    
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
            self.status_label.setText("Status: Połączony")
            self.status_label.setStyleSheet("color: #27ae60; font-size: 10px;")
        else:
            self.status_label.setText("Status: Rozłączony")
            self.status_label.setStyleSheet("color: #e74c3c; font-size: 10px;")
            self.video_label.setText("Łączenie...")
    
    def set_status(self, status: str):
        """Set a custom status for the camera (e.g., DISABLED)."""
        if status == "DISABLED":
            self.status_label.setText("Status: Wyłączona")
            self.status_label.setStyleSheet("color: #7f8c8d; font-size: 10px;")
            self.video_label.setText("Kamera wyłączona")
            self.video_label.setStyleSheet("background-color: #2c2c2c; color: #7f8c8d;")
    
    def set_error(self, error_msg: str):
        """Display error message."""
        self.status_label.setText(f"Błąd: {error_msg[:30]}...")
        self.status_label.setStyleSheet("color: #e74c3c; font-size: 10px;")


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
        self.setFrameStyle(QFrame.Shape.Box | QFrame.Shadow.Raised)
        self.setLineWidth(1)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        
        # Set style based on confirmation status
        if self.person_info.is_confirmed:
            self.setStyleSheet("""
                PersonCardWidget {
                    background-color: #2d3436;
                    border: 2px solid #27ae60;
                    border-radius: 5px;
                    margin: 2px;
                }
                PersonCardWidget:hover {
                    background-color: #3d4446;
                }
            """)
        else:
            self.setStyleSheet("""
                PersonCardWidget {
                    background-color: #2d3436;
                    border: 2px solid #f39c12;
                    border-radius: 5px;
                    margin: 2px;
                }
                PersonCardWidget:hover {
                    background-color: #3d4446;
                }
            """)
        
        layout = QHBoxLayout(self)
        layout.setContentsMargins(5, 5, 5, 5)
        layout.setSpacing(10)
        
        # Face thumbnail
        self.thumbnail_label = QLabel()
        self.thumbnail_label.setFixedSize(60, 60)
        self.thumbnail_label.setStyleSheet("border: 1px solid #7f8c8d; border-radius: 3px;")
        self._update_thumbnail()
        layout.addWidget(self.thumbnail_label)
        
        # Info layout
        info_layout = QVBoxLayout()
        info_layout.setSpacing(2)
        
        # ID / Name
        if self.person_info.first_name or self.person_info.last_name:
            name = f"{self.person_info.first_name} {self.person_info.last_name}".strip()
        else:
            if self.person_info.is_confirmed:
                name = f"ID: {self.person_info.person_id}"
            else:
                name = f"Kandydat #{abs(self.person_info.person_id)}"
        
        self.name_label = QLabel(name)
        self.name_label.setFont(QFont("Arial", 11, QFont.Weight.Bold))
        self.name_label.setStyleSheet("color: white;")
        info_layout.addWidget(self.name_label)
        
        # Status
        status = "✓ Potwierdzony" if self.person_info.is_confirmed else "⏳ Kandydat"
        self.status_label = QLabel(status)
        self.status_label.setStyleSheet(
            "color: #27ae60;" if self.person_info.is_confirmed else "color: #f39c12;"
        )
        info_layout.addWidget(self.status_label)
        
        # Camera info
        cam_text = f"Kamera: {self.person_info.last_seen_camera + 1}"
        self.camera_label = QLabel(cam_text)
        self.camera_label.setStyleSheet("color: #95a5a6; font-size: 10px;")
        info_layout.addWidget(self.camera_label)
        
        layout.addLayout(info_layout)
        layout.addStretch()
    
    def _update_thumbnail(self):
        """Update the face thumbnail."""
        if self.person_info.face_thumbnail is not None:
            thumb = self.person_info.face_thumbnail
            
            # Resize to fit
            thumb_resized = cv2.resize(thumb, (60, 60), interpolation=cv2.INTER_LINEAR)
            rgb_thumb = cv2.cvtColor(thumb_resized, cv2.COLOR_BGR2RGB)
            
            h, w = rgb_thumb.shape[:2]
            bytes_per_line = 3 * w
            q_image = QImage(rgb_thumb.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
            
            self.thumbnail_label.setPixmap(QPixmap.fromImage(q_image))
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
                name = f"ID: {person_info.person_id}"
            else:
                name = f"Kandydat #{abs(person_info.person_id)}"
        
        self.name_label.setText(name)


# ============================================================================
# Sidebar Panel
# ============================================================================

class SidebarPanel(QFrame):
    """Side panel with detected persons list and add client form."""
    
    person_selected = pyqtSignal(int)  # person_id
    client_added = pyqtSignal(int, str, str)  # person_id, first_name, last_name
    photo_uploaded = pyqtSignal(str)  # file_path
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self._person_cards: Dict[int, PersonCardWidget] = {}
        self._selected_person_id: Optional[int] = None
        self._setup_ui()
    
    def _setup_ui(self):
        """Setup the sidebar UI."""
        self.setMinimumWidth(280)
        self.setMaximumWidth(350)
        self.setStyleSheet("background-color: #1e272e;")
        
        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(10)
        
        # ===== Persons Section =====
        persons_header = QLabel("👥 Osoby")
        persons_header.setFont(QFont("Arial", 14, QFont.Weight.Bold))
        persons_header.setStyleSheet("color: white; padding: 5px;")
        layout.addWidget(persons_header)
        
        # Scroll area for persons
        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.scroll_area.setStyleSheet("""
            QScrollArea {
                border: none;
                background-color: #1e272e;
            }
            QScrollBar:vertical {
                background-color: #2d3436;
                width: 10px;
            }
            QScrollBar::handle:vertical {
                background-color: #636e72;
                border-radius: 5px;
            }
        """)
        
        self.persons_container = QWidget()
        self.persons_layout = QVBoxLayout(self.persons_container)
        self.persons_layout.setContentsMargins(0, 0, 0, 0)
        self.persons_layout.setSpacing(5)
        self.persons_layout.addStretch()
        
        self.scroll_area.setWidget(self.persons_container)
        layout.addWidget(self.scroll_area, stretch=1)
        
        # ===== Stats Section =====
        self.stats_label = QLabel("Potwierdzeni: 0 | Kandydaci: 0")
        self.stats_label.setStyleSheet("color: #95a5a6; font-size: 11px;")
        self.stats_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.stats_label)
        
        # ===== Add Client Section =====
        add_client_frame = QFrame()
        add_client_frame.setStyleSheet("""
            QFrame {
                background-color: #2d3436;
                border-radius: 8px;
                padding: 10px;
            }
        """)
        add_client_layout = QVBoxLayout(add_client_frame)
        add_client_layout.setSpacing(8)
        
        add_client_header = QLabel("➕ Dodaj Klienta")
        add_client_header.setFont(QFont("Arial", 12, QFont.Weight.Bold))
        add_client_header.setStyleSheet("color: white;")
        add_client_layout.addWidget(add_client_header)
        
        # Selected person indicator
        self.selected_label = QLabel("Wybierz osobę z listy powyżej")
        self.selected_label.setStyleSheet("color: #f39c12; font-size: 10px;")
        add_client_layout.addWidget(self.selected_label)
        
        # First name input
        self.first_name_input = QLineEdit()
        self.first_name_input.setPlaceholderText("Imię")
        self.first_name_input.setStyleSheet("""
            QLineEdit {
                background-color: #3d4446;
                color: white;
                border: 1px solid #636e72;
                border-radius: 5px;
                padding: 8px;
                font-size: 12px;
            }
            QLineEdit:focus {
                border: 1px solid #3498db;
            }
        """)
        add_client_layout.addWidget(self.first_name_input)
        
        # Last name input
        self.last_name_input = QLineEdit()
        self.last_name_input.setPlaceholderText("Nazwisko")
        self.last_name_input.setStyleSheet("""
            QLineEdit {
                background-color: #3d4446;
                color: white;
                border: 1px solid #636e72;
                border-radius: 5px;
                padding: 8px;
                font-size: 12px;
            }
            QLineEdit:focus {
                border: 1px solid #3498db;
            }
        """)
        add_client_layout.addWidget(self.last_name_input)
        
        # Buttons layout
        buttons_layout = QHBoxLayout()
        
        # Add client button
        self.add_button = QPushButton("Dodaj Klienta")
        self.add_button.setStyleSheet("""
            QPushButton {
                background-color: #27ae60;
                color: white;
                border: none;
                border-radius: 5px;
                padding: 10px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #2ecc71;
            }
            QPushButton:pressed {
                background-color: #1e8449;
            }
            QPushButton:disabled {
                background-color: #636e72;
            }
        """)
        self.add_button.clicked.connect(self._on_add_client)
        self.add_button.setEnabled(False)
        buttons_layout.addWidget(self.add_button)
        
        # Upload photo button
        self.upload_button = QPushButton("📷")
        self.upload_button.setFixedWidth(40)
        self.upload_button.setToolTip("Wczytaj zdjęcie z dysku")
        self.upload_button.setStyleSheet("""
            QPushButton {
                background-color: #3498db;
                color: white;
                border: none;
                border-radius: 5px;
                padding: 10px;
                font-size: 14px;
            }
            QPushButton:hover {
                background-color: #5dade2;
            }
            QPushButton:pressed {
                background-color: #2980b9;
            }
        """)
        self.upload_button.clicked.connect(self._on_upload_photo)
        buttons_layout.addWidget(self.upload_button)
        
        add_client_layout.addLayout(buttons_layout)
        layout.addWidget(add_client_frame)
    
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
    
    def update_stats(self, confirmed: int, candidates: int):
        """Update statistics label."""
        self.stats_label.setText(f"Potwierdzeni: {confirmed} | Kandydaci: {candidates}")
    
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
                    name = f"ID: {info.person_id}" if info.is_confirmed else f"Kandydat #{abs(info.person_id)}"
                self.selected_label.setText(f"Wybrano: {name}")
                self.selected_label.setStyleSheet("color: #27ae60; font-size: 10px;")
                self.add_button.setEnabled(True)
            else:
                self._selected_person_id = None
                self.selected_label.setText("Wybierz osobę z listy powyżej")
                self.selected_label.setStyleSheet("color: #f39c12; font-size: 10px;")
                self.add_button.setEnabled(False)
        else:
            self.selected_label.setText("Wybierz osobę z listy powyżej")
            self.selected_label.setStyleSheet("color: #f39c12; font-size: 10px;")
            self.add_button.setEnabled(False)
    
    def _on_add_client(self):
        """Handle add client button click."""
        if self._selected_person_id is None:
            return
        
        first_name = self.first_name_input.text().strip()
        last_name = self.last_name_input.text().strip()
        
        if not first_name and not last_name:
            QMessageBox.warning(
                self, "Błąd",
                "Proszę podać imię lub nazwisko."
            )
            return
        
        self.client_added.emit(self._selected_person_id, first_name, last_name)
        
        # Clear inputs
        self.first_name_input.clear()
        self.last_name_input.clear()
    
    def _on_upload_photo(self):
        """Handle upload photo button click."""
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Wybierz zdjęcie",
            "",
            "Obrazy (*.png *.jpg *.jpeg *.bmp)"
        )
        
        if file_path:
            self.photo_uploaded.emit(file_path)


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
        
        # Person name mapping
        self._person_names: Dict[int, Tuple[str, str]] = {}  # person_id -> (first, last)
        
        # Camera workers
        self.camera_workers: List[CameraWorker] = []
        
        # Setup UI
        self._setup_ui()
        self._setup_camera_workers()
        
        # Setup update timer for stats
        self.stats_timer = QTimer(self)
        self.stats_timer.timeout.connect(self._update_stats)
        self.stats_timer.start(1000)  # Update every second
    
    def _setup_ui(self):
        """Setup the main window UI."""
        self.setWindowTitle("Multi-Camera Face Re-identification System")
        self.setMinimumSize(1600, 900)  # Larger minimum for better video quality
        self.resize(1920, 1080)  # Default to Full HD
        self.setStyleSheet("background-color: #0f0f1a;")
        
        # Central widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # Main horizontal layout
        main_layout = QHBoxLayout(central_widget)
        main_layout.setContentsMargins(10, 10, 10, 10)
        main_layout.setSpacing(10)
        
        # ===== Left side: Camera Grid =====
        camera_container = QWidget()
        camera_layout = QGridLayout(camera_container)
        camera_layout.setSpacing(10)
        
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
        self.sidebar.photo_uploaded.connect(self._on_photo_uploaded)
        main_layout.addWidget(self.sidebar)
    
    def _setup_camera_workers(self):
        """Create and start camera worker threads."""
        # Only start workers for configured cameras
        for i in range(4):
            if i < self.num_active_cameras:
                url = self.config.CAMERA_URLS[i]
                worker = CameraWorker(
                    camera_id=i,
                    url=url,
                    config=self.config,
                    detector=self.detector,
                    embedding_engine=self.embedding_engine,
                    gallery=self.gallery
                )
                
                # Connect signals
                worker.frame_ready.connect(self._on_frame_ready)
                worker.detections_ready.connect(self._on_detections_ready)
                worker.connection_status.connect(self._on_connection_status)
                worker.error_occurred.connect(self._on_error_occurred)
                
                self.camera_workers.append(worker)
                worker.start()
                
                logger.info(f"Started camera worker {i}")
            else:
                # Mark disabled cameras
                self.camera_workers.append(None)  # Placeholder
                self.camera_widgets[i].set_status("DISABLED")
                logger.info(f"Camera {i} is disabled")
    
    @pyqtSlot(int, np.ndarray)
    def _on_frame_ready(self, camera_id: int, frame: np.ndarray):
        """Handle new frame from camera worker."""
        if 0 <= camera_id < len(self.camera_widgets):
            self.camera_widgets[camera_id].update_frame(frame)
    
    @pyqtSlot(int, list)
    def _on_detections_ready(self, camera_id: int, detections: List[FaceDetection]):
        """Handle new detections from camera worker."""
        for detection in detections:
            if detection.person_id is not None and detection.face_crop is not None:
                # Get name if assigned
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
                    last_seen_time=time.time()
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
        """Handle client addition."""
        logger.info(f"Adding client: {first_name} {last_name} to person {person_id}")
        
        # Store the name mapping
        self._person_names[person_id] = (first_name, last_name)
        
        # Update the sidebar card if it exists
        cards = self.sidebar._person_cards
        if person_id in cards:
            card = cards[person_id]
            card.person_info.first_name = first_name
            card.person_info.last_name = last_name
            card.update_info(card.person_info)
        
        QMessageBox.information(
            self,
            "Sukces",
            f"Przypisano dane klienta:\n{first_name} {last_name}"
        )
    
    @pyqtSlot(str)
    def _on_photo_uploaded(self, file_path: str):
        """Handle photo upload."""
        logger.info(f"Photo uploaded: {file_path}")
        
        try:
            # Load image
            image = cv2.imread(file_path)
            if image is None:
                QMessageBox.warning(self, "Błąd", "Nie można wczytać obrazu.")
                return
            
            # Detect faces in uploaded image
            detections = self.detector.detect(image)
            reid_detections, _ = self.detector.filter_for_reid(detections)
            
            if not reid_detections:
                QMessageBox.warning(
                    self,
                    "Brak twarzy",
                    "Nie wykryto twarzy na wczytanym zdjęciu."
                )
                return
            
            # Use the first (largest) face
            detection = max(reid_detections, key=lambda d: d.area)
            
            # Extract embedding
            embedding = self.embedding_engine.extract_embedding(detection.face_crop)
            
            # Add to gallery
            person_id, is_confirmed = self.gallery.match_and_update(
                embedding, detection.bbox, camera_id=-1
            )
            
            # Create person info
            first_name, last_name = self._person_names.get(person_id, ("", ""))
            person_info = PersonInfo(
                person_id=person_id,
                is_confirmed=is_confirmed,
                face_thumbnail=detection.face_crop.copy(),
                first_name=first_name,
                last_name=last_name,
                last_seen_camera=-1,
                last_seen_time=time.time()
            )
            
            self.sidebar.add_or_update_person(person_info)
            
            QMessageBox.information(
                self,
                "Sukces",
                f"Dodano twarz ze zdjęcia.\n"
                f"{'ID: ' + str(person_id) if is_confirmed else 'Kandydat #' + str(abs(person_id))}"
            )
            
        except Exception as e:
            logger.error(f"Error processing uploaded photo: {e}")
            QMessageBox.warning(self, "Błąd", f"Błąd przetwarzania: {str(e)}")
    
    def _update_stats(self):
        """Update gallery statistics in sidebar."""
        stats = self.gallery.get_stats()
        self.sidebar.update_stats(
            stats['confirmed_faces'],
            stats['candidates']
        )
    
    def closeEvent(self, event):
        """Handle window close event."""
        logger.info("Closing application...")
        
        # Stop all camera workers (skip None placeholders for disabled cameras)
        for worker in self.camera_workers:
            if worker is not None:
                worker.stop()
        
        # Wait for workers to finish
        for worker in self.camera_workers:
            if worker is not None:
                worker.wait(2000)
        
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
