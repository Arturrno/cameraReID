"""
Multi-Camera Person Re-identification System
=============================================
A production-grade system for tracking persons across multiple RTSP camera streams
using YOLOv8 for detection and OSNet for re-identification.

Author: Senior Python Developer
Date: 2024
"""

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from ultralytics import YOLO
import torchreid
from threading import Thread, Lock, Event
from queue import Queue, Empty
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from collections import OrderedDict
import time
import logging
from abc import ABC, abstractmethod
from scipy.spatial.distance import cosine

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ============================================================================
# Configuration
# ============================================================================

@dataclass
class Config:
    """Central configuration for the ReID system."""
    # RTSP Stream URLs
    CAMERA_URLS: List[str] = field(default_factory=lambda: [
        'rtsp://admin:admin123@192.168.1.53:554/Preview_01_sub',
        'rtsp://admin:admin123@192.168.1.54:554/Preview_01_sub',
    ])
    
    # Detection settings
    YOLO_MODEL: str = 'yolov8n.pt'  # Can use yolov8s.pt, yolov8m.pt for better accuracy
    DETECTION_CONFIDENCE: float = 0.5
    PERSON_CLASS_ID: int = 0  # COCO class ID for person
    
    # ReID settings
    REID_MODEL: str = 'osnet_x1_0'
    REID_INPUT_SIZE: Tuple[int, int] = (256, 128)  # Height x Width
    EMBEDDING_DIM: int = 512
    SIMILARITY_THRESHOLD: float = 0.45  # Cosine similarity threshold (lower for cross-camera)
    
    # Performance settings
    FRAME_SKIP: int = 3  # Process every Nth frame
    MAX_GALLERY_SIZE: int = 1000  # Maximum persons to track
    GALLERY_CLEANUP_THRESHOLD: int = 100  # Cleanup when gallery exceeds this after max
    FEATURE_HISTORY_SIZE: int = 10  # Number of features to keep per person for averaging
    
    # Stream settings
    RECONNECT_DELAY: float = 5.0  # Seconds to wait before reconnection attempt
    MAX_RECONNECT_ATTEMPTS: int = 10
    FRAME_QUEUE_SIZE: int = 2
    
    # Visualization
    WINDOW_WIDTH: int = 640
    WINDOW_HEIGHT: int = 480
    BBOX_THICKNESS: int = 2
    FONT_SCALE: float = 0.6
    
    # Device
    DEVICE: str = 'cuda' if torch.cuda.is_available() else 'cpu'


# ============================================================================
# Data Classes
# ============================================================================

@dataclass
class Detection:
    """Represents a detected person."""
    bbox: Tuple[int, int, int, int]  # x1, y1, x2, y2
    confidence: float
    crop: np.ndarray
    embedding: Optional[np.ndarray] = None
    person_id: Optional[int] = None


@dataclass
class TrackedPerson:
    """Represents a tracked person in the gallery."""
    person_id: int
    embeddings: List[np.ndarray] = field(default_factory=list)
    last_seen: float = field(default_factory=time.time)
    camera_id: int = -1
    hit_count: int = 1
    
    def get_average_embedding(self) -> np.ndarray:
        """Get the average embedding for this person."""
        if not self.embeddings:
            return np.zeros(512)
        return np.mean(self.embeddings, axis=0)
    
    def add_embedding(self, embedding: np.ndarray, max_history: int = 10):
        """Add a new embedding, maintaining a sliding window."""
        self.embeddings.append(embedding)
        if len(self.embeddings) > max_history:
            self.embeddings.pop(0)
        self.last_seen = time.time()
        self.hit_count += 1


# ============================================================================
# Gallery Manager (Thread-Safe Global Identity Store)
# ============================================================================

class Gallery:
    """
    Thread-safe global gallery for storing and managing person identities.
    Uses an LRU-style eviction policy when the gallery is full.
    """
    
    def __init__(self, config: Config):
        self.config = config
        self._persons: OrderedDict[int, TrackedPerson] = OrderedDict()
        self._lock = Lock()
        self._next_id = 1
        self._id_lock = Lock()
    
    def _generate_id(self) -> int:
        """Generate a new unique person ID."""
        with self._id_lock:
            new_id = self._next_id
            self._next_id += 1
            return new_id
    
    def _find_match_unlocked(self, embedding: np.ndarray) -> Tuple[Optional[int], float]:
        """
        Find the best matching person for the given embedding (without locking).
        Must be called while holding self._lock.
        
        Returns:
            Tuple of (person_id, similarity_score) or (None, 0.0) if no match
        """
        if not self._persons:
            return None, 0.0
        
        best_match_id = None
        best_similarity = 0.0
        
        # Normalize the query embedding
        query_norm = embedding / (np.linalg.norm(embedding) + 1e-8)
        
        for person_id, person in self._persons.items():
            avg_embedding = person.get_average_embedding()
            gallery_norm = avg_embedding / (np.linalg.norm(avg_embedding) + 1e-8)
            
            # Cosine similarity
            similarity = float(np.dot(query_norm, gallery_norm))
            
            if similarity > best_similarity:
                best_similarity = similarity
                best_match_id = person_id
        
        if best_similarity >= self.config.SIMILARITY_THRESHOLD:
            return best_match_id, best_similarity
        
        return None, best_similarity
    
    def find_match(self, embedding: np.ndarray) -> Tuple[Optional[int], float]:
        """
        Find the best matching person for the given embedding.
        
        Returns:
            Tuple of (person_id, similarity_score) or (None, 0.0) if no match
        """
        with self._lock:
            return self._find_match_unlocked(embedding)
    
    def add_or_update(self, embedding: np.ndarray, camera_id: int) -> int:
        """
        Add a new person or update an existing one.
        Atomic operation - find and update/create in single lock.
        
        Returns:
            The person ID (new or existing)
        """
        with self._lock:
            # Find match while holding the lock to prevent race conditions
            match_id, similarity = self._find_match_unlocked(embedding)
            
            if match_id is not None:
                # Update existing person
                person = self._persons[match_id]
                person.add_embedding(embedding, self.config.FEATURE_HISTORY_SIZE)
                person.camera_id = camera_id
                # Move to end (LRU)
                self._persons.move_to_end(match_id)
                logger.debug(f"Updated person {match_id} (cam {camera_id}) similarity {similarity:.3f}")
                return match_id
            else:
                # Create new person
                new_id = self._generate_id()
                new_person = TrackedPerson(
                    person_id=new_id,
                    embeddings=[embedding],
                    camera_id=camera_id
                )
                self._persons[new_id] = new_person
                logger.info(f"New person registered: ID {new_id} (cam {camera_id}, best_sim={similarity:.3f})")
                
                # Cleanup if needed
                self._cleanup_if_needed()
                
                return new_id
    
    def _cleanup_if_needed(self):
        """Remove oldest entries if gallery exceeds maximum size."""
        if len(self._persons) > self.config.MAX_GALLERY_SIZE:
            # Remove oldest entries (LRU eviction)
            items_to_remove = len(self._persons) - self.config.MAX_GALLERY_SIZE + self.config.GALLERY_CLEANUP_THRESHOLD
            for _ in range(items_to_remove):
                if self._persons:
                    oldest_id = next(iter(self._persons))
                    del self._persons[oldest_id]
                    logger.debug(f"Evicted person {oldest_id} from gallery (LRU)")
    
    def get_stats(self) -> Dict:
        """Get gallery statistics."""
        with self._lock:
            return {
                'total_persons': len(self._persons),
                'next_id': self._next_id
            }


# ============================================================================
# Person Detector (YOLOv8)
# ============================================================================

class PersonDetector:
    """
    High-performance person detector using YOLOv8.
    """
    
    def __init__(self, config: Config):
        self.config = config
        logger.info(f"Loading YOLO model: {config.YOLO_MODEL}")
        self.model = YOLO(config.YOLO_MODEL)
        self.model.to(config.DEVICE)
        logger.info(f"YOLO model loaded on {config.DEVICE}")
    
    def detect(self, frame: np.ndarray) -> List[Detection]:
        """
        Detect persons in the frame.
        
        Args:
            frame: BGR image as numpy array
            
        Returns:
            List of Detection objects
        """
        detections = []
        
        # Run inference
        results = self.model(frame, verbose=False, classes=[self.config.PERSON_CLASS_ID])
        
        for result in results:
            boxes = result.boxes
            if boxes is None:
                continue
                
            for box in boxes:
                confidence = float(box.conf[0])
                if confidence < self.config.DETECTION_CONFIDENCE:
                    continue
                
                # Get bounding box
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                
                # Ensure valid coordinates
                x1 = max(0, x1)
                y1 = max(0, y1)
                x2 = min(frame.shape[1], x2)
                y2 = min(frame.shape[0], y2)
                
                # Skip if box is too small
                if (x2 - x1) < 20 or (y2 - y1) < 40:
                    continue
                
                # Extract crop
                crop = frame[y1:y2, x1:x2].copy()
                
                detections.append(Detection(
                    bbox=(x1, y1, x2, y2),
                    confidence=confidence,
                    crop=crop
                ))
        
        return detections


# ============================================================================
# ReID Engine (OSNet Feature Extraction)
# ============================================================================

class ReIDEngine:
    """
    Re-identification engine using OSNet for feature extraction.
    Extracts 512-dimensional embeddings for person crops.
    """
    
    def __init__(self, config: Config):
        self.config = config
        self.device = torch.device(config.DEVICE)
        
        logger.info(f"Loading ReID model: {config.REID_MODEL}")
        
        # Build OSNet model using torchreid
        self.model = torchreid.models.build_model(
            name=config.REID_MODEL,
            num_classes=1000,  # Pretrained on ImageNet
            loss='softmax',
            pretrained=True
        )
        self.model = self.model.to(self.device)
        self.model.eval()
        
        logger.info(f"ReID model loaded on {config.DEVICE}")
        
        # Preprocessing transforms
        self.mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(self.device)
        self.std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(self.device)
    
    def preprocess(self, crop: np.ndarray) -> torch.Tensor:
        """
        Preprocess a person crop for the ReID model.
        
        Args:
            crop: BGR image as numpy array
            
        Returns:
            Preprocessed tensor ready for model inference
        """
        # Resize to model input size (256x128)
        resized = cv2.resize(crop, (self.config.REID_INPUT_SIZE[1], self.config.REID_INPUT_SIZE[0]))
        
        # Convert BGR to RGB
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        
        # Convert to tensor and normalize to [0, 1]
        tensor = torch.from_numpy(rgb).float().permute(2, 0, 1).unsqueeze(0) / 255.0
        tensor = tensor.to(self.device)
        
        # Normalize with ImageNet stats
        tensor = (tensor - self.mean) / self.std
        
        return tensor
    
    @torch.no_grad()
    def extract_embedding(self, crop: np.ndarray) -> np.ndarray:
        """
        Extract a 512-dimensional embedding for a person crop.
        
        Args:
            crop: BGR image as numpy array
            
        Returns:
            512-dimensional numpy array
        """
        tensor = self.preprocess(crop)
        features = self.model(tensor)
        
        # Normalize the embedding
        features = F.normalize(features, p=2, dim=1)
        
        return features.cpu().numpy().flatten()
    
    @torch.no_grad()
    def extract_embeddings_batch(self, crops: List[np.ndarray]) -> List[np.ndarray]:
        """
        Extract embeddings for multiple crops in a batch.
        
        Args:
            crops: List of BGR images
            
        Returns:
            List of 512-dimensional numpy arrays
        """
        if not crops:
            return []
        
        # Preprocess all crops
        tensors = [self.preprocess(crop) for crop in crops]
        batch = torch.cat(tensors, dim=0)
        
        # Extract features
        features = self.model(batch)
        features = F.normalize(features, p=2, dim=1)
        
        return [f.cpu().numpy().flatten() for f in features]


# ============================================================================
# Camera Stream Handler
# ============================================================================

class CameraStream:
    """
    Handles RTSP stream capture with automatic reconnection.
    Runs in a separate thread for non-blocking frame capture.
    """
    
    def __init__(self, camera_id: int, url: str, config: Config):
        self.camera_id = camera_id
        self.url = url
        self.config = config
        
        self._frame_queue: Queue = Queue(maxsize=config.FRAME_QUEUE_SIZE)
        self._stop_event = Event()
        self._thread: Optional[Thread] = None
        self._cap: Optional[cv2.VideoCapture] = None
        self._connected = False
        self._reconnect_attempts = 0
        self._frame_count = 0
        
        self.logger = logging.getLogger(f"Camera-{camera_id}")
    
    def _connect(self) -> bool:
        """Attempt to connect to the RTSP stream."""
        try:
            self.logger.info(f"Connecting to {self.url}")
            
            # Release existing capture if any
            if self._cap is not None:
                self._cap.release()
            
            # Create new capture
            self._cap = cv2.VideoCapture(self.url, cv2.CAP_FFMPEG)
            
            # Set buffer size to reduce latency
            self._cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            
            if self._cap.isOpened():
                self._connected = True
                self._reconnect_attempts = 0
                self.logger.info("Connected successfully")
                return True
            else:
                self._connected = False
                self.logger.error("Failed to open stream")
                return False
                
        except Exception as e:
            self.logger.error(f"Connection error: {e}")
            self._connected = False
            return False
    
    def _capture_loop(self):
        """Main capture loop running in a thread."""
        while not self._stop_event.is_set():
            # Check connection
            if not self._connected:
                if self._reconnect_attempts >= self.config.MAX_RECONNECT_ATTEMPTS:
                    self.logger.error("Max reconnection attempts reached. Stopping.")
                    break
                
                self._reconnect_attempts += 1
                self.logger.info(f"Reconnection attempt {self._reconnect_attempts}/{self.config.MAX_RECONNECT_ATTEMPTS}")
                
                if not self._connect():
                    time.sleep(self.config.RECONNECT_DELAY)
                    continue
            
            # Read frame
            try:
                ret, frame = self._cap.read()
                
                if not ret or frame is None:
                    self.logger.warning("Failed to read frame, reconnecting...")
                    self._connected = False
                    continue
                
                self._frame_count += 1
                
                # Apply frame skipping
                if self._frame_count % self.config.FRAME_SKIP != 0:
                    continue
                
                # Put frame in queue (non-blocking)
                try:
                    # Clear old frame if queue is full
                    if self._frame_queue.full():
                        try:
                            self._frame_queue.get_nowait()
                        except Empty:
                            pass
                    self._frame_queue.put_nowait((self._frame_count, frame))
                except:
                    pass
                    
            except Exception as e:
                self.logger.error(f"Capture error: {e}")
                self._connected = False
    
    def start(self):
        """Start the capture thread."""
        if self._thread is not None and self._thread.is_alive():
            return
        
        self._stop_event.clear()
        self._thread = Thread(target=self._capture_loop, daemon=True)
        self._thread.start()
        self.logger.info("Capture thread started")
    
    def stop(self):
        """Stop the capture thread."""
        self._stop_event.set()
        
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        
        if self._cap is not None:
            self._cap.release()
        
        self.logger.info("Capture thread stopped")
    
    def get_frame(self) -> Optional[Tuple[int, np.ndarray]]:
        """Get the latest frame from the queue."""
        try:
            return self._frame_queue.get_nowait()
        except Empty:
            return None
    
    @property
    def is_connected(self) -> bool:
        return self._connected


# ============================================================================
# Visualization
# ============================================================================

class Visualizer:
    """Handles rendering of detections and IDs on frames."""
    
    # Color palette for different person IDs
    COLORS = [
        (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0),
        (255, 0, 255), (0, 255, 255), (128, 0, 0), (0, 128, 0),
        (0, 0, 128), (128, 128, 0), (128, 0, 128), (0, 128, 128),
    ]
    
    def __init__(self, config: Config):
        self.config = config
    
    def get_color(self, person_id: int) -> Tuple[int, int, int]:
        """Get a consistent color for a person ID."""
        return self.COLORS[person_id % len(self.COLORS)]
    
    def draw_detection(self, frame: np.ndarray, detection: Detection) -> np.ndarray:
        """Draw bounding box and ID on the frame."""
        x1, y1, x2, y2 = detection.bbox
        person_id = detection.person_id or 0
        color = self.get_color(person_id)
        
        # Draw bounding box
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, self.config.BBOX_THICKNESS)
        
        # Prepare label
        label = f"ID: {person_id} ({detection.confidence:.2f})"
        
        # Get label size
        (label_w, label_h), baseline = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, self.config.FONT_SCALE, 2
        )
        
        # Draw label background
        cv2.rectangle(
            frame,
            (x1, y1 - label_h - 10),
            (x1 + label_w + 5, y1),
            color,
            -1
        )
        
        # Draw label text
        cv2.putText(
            frame,
            label,
            (x1 + 2, y1 - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            self.config.FONT_SCALE,
            (255, 255, 255),
            2
        )
        
        return frame
    
    def draw_info(self, frame: np.ndarray, camera_id: int, fps: float, gallery_stats: Dict) -> np.ndarray:
        """Draw camera info and stats on the frame."""
        info_text = f"Camera {camera_id} | FPS: {fps:.1f} | Gallery: {gallery_stats['total_persons']}"
        
        cv2.putText(
            frame,
            info_text,
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2
        )
        
        return frame


# ============================================================================
# Main ReID System
# ============================================================================

class MultiCameraReIDSystem:
    """
    Main system orchestrating multi-camera person re-identification.
    """
    
    def __init__(self, config: Optional[Config] = None):
        self.config = config or Config()
        
        # Initialize components
        logger.info("Initializing Multi-Camera ReID System...")
        
        self.detector = PersonDetector(self.config)
        self.reid_engine = ReIDEngine(self.config)
        self.gallery = Gallery(self.config)
        self.visualizer = Visualizer(self.config)
        
        # Initialize camera streams
        self.streams: List[CameraStream] = []
        for i, url in enumerate(self.config.CAMERA_URLS):
            stream = CameraStream(i, url, self.config)
            self.streams.append(stream)
        
        # Processing state
        self._running = False
        self._fps_counters: Dict[int, List[float]] = {i: [] for i in range(len(self.streams))}
        
        logger.info(f"System initialized with {len(self.streams)} cameras")
    
    def _process_frame(self, camera_id: int, frame: np.ndarray) -> np.ndarray:
        """
        Process a single frame: detect, extract features, match, visualize.
        """
        # Detect persons
        detections = self.detector.detect(frame)
        
        if detections:
            # Extract embeddings in batch
            crops = [d.crop for d in detections]
            embeddings = self.reid_engine.extract_embeddings_batch(crops)
            
            # Match with gallery and assign IDs
            for detection, embedding in zip(detections, embeddings):
                detection.embedding = embedding
                detection.person_id = self.gallery.add_or_update(embedding, camera_id)
        
        # Visualize
        for detection in detections:
            frame = self.visualizer.draw_detection(frame, detection)
        
        return frame
    
    def run(self):
        """Main run loop."""
        logger.info("Starting Multi-Camera ReID System...")
        self._running = True
        
        # Start all camera streams
        for stream in self.streams:
            stream.start()
        
        # Create windows
        for i in range(len(self.streams)):
            cv2.namedWindow(f"Camera {i}", cv2.WINDOW_NORMAL)
            cv2.resizeWindow(f"Camera {i}", self.config.WINDOW_WIDTH, self.config.WINDOW_HEIGHT)
        
        last_times = {i: time.time() for i in range(len(self.streams))}
        
        try:
            while self._running:
                for i, stream in enumerate(self.streams):
                    frame_data = stream.get_frame()
                    
                    if frame_data is None:
                        continue
                    
                    frame_count, frame = frame_data
                    
                    # Calculate FPS
                    current_time = time.time()
                    fps = 1.0 / (current_time - last_times[i] + 1e-8)
                    last_times[i] = current_time
                    
                    # Process frame
                    processed_frame = self._process_frame(i, frame)
                    
                    # Draw info overlay
                    processed_frame = self.visualizer.draw_info(
                        processed_frame, i, fps, self.gallery.get_stats()
                    )
                    
                    # Display
                    cv2.imshow(f"Camera {i}", processed_frame)
                
                # Handle key events
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    logger.info("Quit requested")
                    break
                elif key == ord('s'):
                    # Print gallery stats
                    stats = self.gallery.get_stats()
                    logger.info(f"Gallery Stats: {stats}")
        
        except KeyboardInterrupt:
            logger.info("Interrupted by user")
        
        finally:
            self.stop()
    
    def stop(self):
        """Stop the system and cleanup resources."""
        logger.info("Stopping system...")
        self._running = False
        
        # Stop all streams
        for stream in self.streams:
            stream.stop()
        
        # Close windows
        cv2.destroyAllWindows()
        
        logger.info("System stopped")


# ============================================================================
# Entry Point
# ============================================================================

def main():
    """Main entry point."""
    print("=" * 60)
    print("Multi-Camera Person Re-identification System")
    print("=" * 60)
    print()
    print("Controls:")
    print("  q - Quit")
    print("  s - Print gallery statistics")
    print()
    
    # Create and run system
    config = Config()
    
    # Print configuration
    print("Configuration:")
    print(f"  Cameras: {len(config.CAMERA_URLS)}")
    print(f"  Device: {config.DEVICE}")
    print(f"  YOLO Model: {config.YOLO_MODEL}")
    print(f"  ReID Model: {config.REID_MODEL}")
    print(f"  Similarity Threshold: {config.SIMILARITY_THRESHOLD}")
    print(f"  Frame Skip: {config.FRAME_SKIP}")
    print()
    
    system = MultiCameraReIDSystem(config)
    system.run()


if __name__ == "__main__":
    main()
