"""
Robust Multi-Camera Face Re-identification System
==================================================
A production-grade system with advanced robustness features for consistent
face recognition across multiple RTSP camera streams.

Features:
- CLAHE preprocessing for lighting robustness
- Momentum-based gallery updates (centroid memory)
- Temporal consistency with wait-and-see buffer
- Hysteresis thresholds (high to create, low to maintain)
- Minimum face size filtering for distance handling
- IOU-based tracking integration

Author: Senior Computer Vision Engineer
Date: 2024
"""

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from facenet_pytorch import MTCNN, InceptionResnetV1
from threading import Thread, Lock, Event
from queue import Queue, Empty
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from collections import OrderedDict, defaultdict
import time
import logging

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
    """Central configuration for the Robust Face ReID system."""
    # RTSP Stream URLs
    CAMERA_URLS: List[str] = field(default_factory=lambda: [
        'rtsp://admin:admin123@192.168.1.53:554/Preview_01_sub',
        'rtsp://admin:admin123@192.168.1.54:554/Preview_01_sub',
    ])
    
    # Face Detection settings (MTCNN)
    DETECTION_CONFIDENCE: float = 0.8  # Lowered for better far-distance detection
    MIN_FACE_SIZE: int = 15  # MTCNN minimum face size (lowered for distance)
    
    # Distance Handling - Minimum face size for ReID
    MIN_FACE_SIZE_FOR_REID: int = 25  # Lowered to allow ReID from further distances
    
    # Face Embedding settings (FaceNet)
    FACENET_INPUT_SIZE: int = 160
    EMBEDDING_DIM: int = 512
    
    # Hysteresis Thresholds (dual threshold strategy)
    HIGH_THRESHOLD: float = 0.70  # Threshold to CREATE a new ID (stricter)
    LOW_THRESHOLD: float = 0.55   # Threshold to MAINTAIN an existing ID (more lenient)
    
    # Momentum-based Gallery Updates
    MOMENTUM_ALPHA: float = 0.1  # gallery = (1-alpha)*gallery + alpha*current
    
    # Temporal Consistency - Wait-and-See Buffer
    MIN_FRAMES_TO_CONFIRM: int = 5  # Frames before committing new ID to gallery
    CANDIDATE_TIMEOUT: float = 2.0  # Seconds before candidate expires
    
    # IOU Tracking
    IOU_THRESHOLD: float = 0.3  # Minimum IOU to consider same track
    IOU_SIMILARITY_BOOST: float = 0.1  # Boost similarity if IOU matches
    
    # CLAHE Preprocessing
    CLAHE_CLIP_LIMIT: float = 2.0
    CLAHE_GRID_SIZE: Tuple[int, int] = (8, 8)
    
    # Performance settings
    FRAME_SKIP: int = 3
    MAX_GALLERY_SIZE: int = 500
    GALLERY_CLEANUP_THRESHOLD: int = 50
    
    # Stream settings
    RECONNECT_DELAY: float = 5.0
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
class FaceDetection:
    """Represents a detected face."""
    bbox: Tuple[int, int, int, int]  # x1, y1, x2, y2
    confidence: float
    face_crop: np.ndarray
    embedding: Optional[np.ndarray] = None
    person_id: Optional[int] = None
    is_confirmed: bool = True  # False if still in candidate buffer
    
    @property
    def width(self) -> int:
        return self.bbox[2] - self.bbox[0]
    
    @property
    def height(self) -> int:
        return self.bbox[3] - self.bbox[1]
    
    @property
    def area(self) -> int:
        return self.width * self.height


@dataclass
class TrackedFace:
    """Represents a confirmed face in the global gallery."""
    person_id: int
    centroid_embedding: np.ndarray  # Momentum-updated centroid
    last_seen: float = field(default_factory=time.time)
    camera_id: int = -1
    hit_count: int = 1
    last_bbox: Optional[Tuple[int, int, int, int]] = None
    
    def update_centroid(self, new_embedding: np.ndarray, alpha: float = 0.1):
        """Update centroid embedding using momentum."""
        # Ensure L2 normalized
        new_norm = new_embedding / (np.linalg.norm(new_embedding) + 1e-8)
        self.centroid_embedding = (1 - alpha) * self.centroid_embedding + alpha * new_norm
        # Re-normalize after update
        self.centroid_embedding = self.centroid_embedding / (np.linalg.norm(self.centroid_embedding) + 1e-8)
        self.last_seen = time.time()
        self.hit_count += 1


@dataclass
class CandidateFace:
    """Represents a candidate face in the wait-and-see buffer."""
    temp_id: int
    embeddings: List[np.ndarray] = field(default_factory=list)
    bboxes: List[Tuple[int, int, int, int]] = field(default_factory=list)
    first_seen: float = field(default_factory=time.time)
    last_seen: float = field(default_factory=time.time)
    camera_id: int = -1
    frame_count: int = 0
    
    def add_observation(self, embedding: np.ndarray, bbox: Tuple[int, int, int, int]):
        """Add a new observation to this candidate."""
        self.embeddings.append(embedding)
        self.bboxes.append(bbox)
        self.last_seen = time.time()
        self.frame_count += 1
    
    def get_average_embedding(self) -> np.ndarray:
        """Get average embedding from all observations."""
        if not self.embeddings:
            return np.zeros(512)
        avg = np.mean(self.embeddings, axis=0)
        return avg / (np.linalg.norm(avg) + 1e-8)
    
    def get_last_bbox(self) -> Optional[Tuple[int, int, int, int]]:
        """Get the most recent bounding box."""
        return self.bboxes[-1] if self.bboxes else None


# ============================================================================
# Utility Functions
# ============================================================================

def compute_iou(box1: Tuple[int, int, int, int], box2: Tuple[int, int, int, int]) -> float:
    """Compute Intersection over Union between two bounding boxes."""
    x1_1, y1_1, x2_1, y2_1 = box1
    x1_2, y1_2, x2_2, y2_2 = box2
    
    # Intersection
    xi1 = max(x1_1, x1_2)
    yi1 = max(y1_1, y1_2)
    xi2 = min(x2_1, x2_2)
    yi2 = min(y2_1, y2_2)
    
    if xi2 <= xi1 or yi2 <= yi1:
        return 0.0
    
    intersection = (xi2 - xi1) * (yi2 - yi1)
    
    # Union
    area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
    area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
    union = area1 + area2 - intersection
    
    return intersection / (union + 1e-8)


def apply_clahe(image: np.ndarray, clip_limit: float = 2.0, 
                grid_size: Tuple[int, int] = (8, 8)) -> np.ndarray:
    """
    Apply CLAHE (Contrast Limited Adaptive Histogram Equalization) to normalize lighting.
    """
    # Convert to LAB color space
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    
    # Split channels
    l, a, b = cv2.split(lab)
    
    # Apply CLAHE to L channel
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=grid_size)
    l_clahe = clahe.apply(l)
    
    # Merge channels
    lab_clahe = cv2.merge([l_clahe, a, b])
    
    # Convert back to BGR
    result = cv2.cvtColor(lab_clahe, cv2.COLOR_LAB2BGR)
    
    return result


# ============================================================================
# Face Gallery with Momentum Updates & Temporal Consistency
# ============================================================================

class RobustFaceGallery:
    """
    Thread-safe face gallery with:
    - Momentum-based centroid updates
    - Wait-and-see candidate buffer
    - Hysteresis thresholds
    - IOU-based tracking integration
    """
    
    def __init__(self, config: Config):
        self.config = config
        
        # Confirmed faces (global gallery)
        self._faces: OrderedDict[int, TrackedFace] = OrderedDict()
        
        # Candidate buffer (wait-and-see)
        self._candidates: Dict[int, CandidateFace] = {}
        
        # Per-camera track memory for IOU matching
        self._last_frame_tracks: Dict[int, List[Tuple[int, Tuple[int, int, int, int]]]] = defaultdict(list)
        
        self._lock = Lock()
        self._next_id = 1
        self._next_temp_id = -1  # Negative IDs for candidates
    
    def _generate_id(self) -> int:
        """Generate a new permanent ID."""
        new_id = self._next_id
        self._next_id += 1
        return new_id
    
    def _generate_temp_id(self) -> int:
        """Generate a new temporary ID for candidates."""
        temp_id = self._next_temp_id
        self._next_temp_id -= 1
        return temp_id
    
    def _compute_similarity(self, emb1: np.ndarray, emb2: np.ndarray) -> float:
        """Compute cosine similarity between two L2-normalized embeddings."""
        norm1 = emb1 / (np.linalg.norm(emb1) + 1e-8)
        norm2 = emb2 / (np.linalg.norm(emb2) + 1e-8)
        return float(np.dot(norm1, norm2))
    
    def _find_iou_match(self, bbox: Tuple[int, int, int, int], 
                        camera_id: int) -> Optional[int]:
        """Find a person ID based on IOU with previous frame tracks."""
        best_iou = 0.0
        best_id = None
        
        for person_id, last_bbox in self._last_frame_tracks[camera_id]:
            iou = compute_iou(bbox, last_bbox)
            if iou > best_iou and iou >= self.config.IOU_THRESHOLD:
                best_iou = iou
                best_id = person_id
        
        return best_id
    
    def _find_gallery_match(self, embedding: np.ndarray, 
                           bbox: Tuple[int, int, int, int],
                           camera_id: int) -> Tuple[Optional[int], float]:
        """
        Find best match in confirmed gallery.
        Uses hysteresis thresholds and IOU boosting.
        """
        best_match_id = None
        best_similarity = 0.0
        
        # Check for IOU match first
        iou_matched_id = self._find_iou_match(bbox, camera_id)
        
        for person_id, face in self._faces.items():
            similarity = self._compute_similarity(embedding, face.centroid_embedding)
            
            # Apply IOU boost if this face matches spatially
            if person_id == iou_matched_id:
                similarity = min(1.0, similarity + self.config.IOU_SIMILARITY_BOOST)
            
            if similarity > best_similarity:
                best_similarity = similarity
                best_match_id = person_id
        
        return best_match_id, best_similarity
    
    def _find_candidate_match(self, embedding: np.ndarray,
                              bbox: Tuple[int, int, int, int]) -> Tuple[Optional[int], float]:
        """Find best match among candidates."""
        best_match_id = None
        best_similarity = 0.0
        
        for temp_id, candidate in self._candidates.items():
            avg_emb = candidate.get_average_embedding()
            similarity = self._compute_similarity(embedding, avg_emb)
            
            # Also check IOU with last bbox
            last_bbox = candidate.get_last_bbox()
            if last_bbox:
                iou = compute_iou(bbox, last_bbox)
                if iou >= self.config.IOU_THRESHOLD:
                    similarity = min(1.0, similarity + self.config.IOU_SIMILARITY_BOOST)
            
            if similarity > best_similarity:
                best_similarity = similarity
                best_match_id = temp_id
        
        return best_match_id, best_similarity
    
    def _promote_candidate(self, temp_id: int) -> int:
        """Promote a candidate to confirmed gallery."""
        candidate = self._candidates.pop(temp_id)
        new_id = self._generate_id()
        
        # Create confirmed face with average embedding as initial centroid
        avg_embedding = candidate.get_average_embedding()
        new_face = TrackedFace(
            person_id=new_id,
            centroid_embedding=avg_embedding,
            camera_id=candidate.camera_id,
            hit_count=candidate.frame_count,
            last_bbox=candidate.get_last_bbox()
        )
        self._faces[new_id] = new_face
        
        logger.info(f"Promoted candidate {temp_id} -> confirmed ID {new_id} "
                   f"(observed {candidate.frame_count} frames)")
        
        return new_id
    
    def _cleanup_expired_candidates(self):
        """Remove candidates that have timed out."""
        current_time = time.time()
        expired = [
            temp_id for temp_id, cand in self._candidates.items()
            if (current_time - cand.last_seen) > self.config.CANDIDATE_TIMEOUT
        ]
        for temp_id in expired:
            del self._candidates[temp_id]
            logger.debug(f"Expired candidate {temp_id}")
    
    def _cleanup_gallery_if_needed(self):
        """Remove oldest entries if gallery exceeds maximum size."""
        if len(self._faces) > self.config.MAX_GALLERY_SIZE:
            items_to_remove = len(self._faces) - self.config.MAX_GALLERY_SIZE + self.config.GALLERY_CLEANUP_THRESHOLD
            for _ in range(items_to_remove):
                if self._faces:
                    oldest_id = next(iter(self._faces))
                    del self._faces[oldest_id]
                    logger.debug(f"Evicted face {oldest_id} from gallery")
    
    def match_and_update(self, embedding: np.ndarray, 
                         bbox: Tuple[int, int, int, int],
                         camera_id: int) -> Tuple[int, bool]:
        """
        Match embedding against gallery and candidates.
        Uses hysteresis thresholds and temporal consistency.
        
        Returns:
            Tuple of (person_id, is_confirmed)
            - person_id: positive for confirmed, negative for candidate
            - is_confirmed: True if ID is in confirmed gallery
        """
        with self._lock:
            # Cleanup expired candidates
            self._cleanup_expired_candidates()
            
            # 1. Try to match with confirmed gallery (using LOW threshold to maintain)
            gallery_match_id, gallery_sim = self._find_gallery_match(embedding, bbox, camera_id)
            
            if gallery_match_id is not None and gallery_sim >= self.config.LOW_THRESHOLD:
                # Update existing face with momentum
                face = self._faces[gallery_match_id]
                face.update_centroid(embedding, self.config.MOMENTUM_ALPHA)
                face.camera_id = camera_id
                face.last_bbox = bbox
                self._faces.move_to_end(gallery_match_id)
                
                logger.debug(f"Matched gallery ID {gallery_match_id} (sim={gallery_sim:.3f})")
                return gallery_match_id, True
            
            # 2. Try to match with candidates
            cand_match_id, cand_sim = self._find_candidate_match(embedding, bbox)
            
            if cand_match_id is not None and cand_sim >= self.config.LOW_THRESHOLD:
                # Update candidate
                candidate = self._candidates[cand_match_id]
                candidate.add_observation(embedding, bbox)
                candidate.camera_id = camera_id
                
                # Check if candidate should be promoted
                if candidate.frame_count >= self.config.MIN_FRAMES_TO_CONFIRM:
                    new_id = self._promote_candidate(cand_match_id)
                    self._cleanup_gallery_if_needed()
                    return new_id, True
                else:
                    logger.debug(f"Updated candidate {cand_match_id} "
                               f"(frames={candidate.frame_count}/{self.config.MIN_FRAMES_TO_CONFIRM})")
                    return cand_match_id, False
            
            # 3. Check if we should create a new candidate (HIGH threshold)
            # Only if no good match in gallery AND no good match in candidates
            if gallery_sim < self.config.HIGH_THRESHOLD and cand_sim < self.config.HIGH_THRESHOLD:
                # Create new candidate
                temp_id = self._generate_temp_id()
                new_candidate = CandidateFace(
                    temp_id=temp_id,
                    embeddings=[embedding],
                    bboxes=[bbox],
                    camera_id=camera_id,
                    frame_count=1
                )
                self._candidates[temp_id] = new_candidate
                
                logger.debug(f"Created new candidate {temp_id}")
                return temp_id, False
            
            # Edge case: similarity is between thresholds
            # Default to best match (even if not ideal)
            if gallery_sim >= cand_sim and gallery_match_id is not None:
                face = self._faces[gallery_match_id]
                face.update_centroid(embedding, self.config.MOMENTUM_ALPHA)
                face.last_bbox = bbox
                return gallery_match_id, True
            elif cand_match_id is not None:
                candidate = self._candidates[cand_match_id]
                candidate.add_observation(embedding, bbox)
                return cand_match_id, False
            else:
                # Fallback: create candidate
                temp_id = self._generate_temp_id()
                new_candidate = CandidateFace(
                    temp_id=temp_id,
                    embeddings=[embedding],
                    bboxes=[bbox],
                    camera_id=camera_id,
                    frame_count=1
                )
                self._candidates[temp_id] = new_candidate
                return temp_id, False
    
    def update_tracks(self, camera_id: int, tracks: List[Tuple[int, Tuple[int, int, int, int]]]):
        """Update the track memory for a camera."""
        with self._lock:
            self._last_frame_tracks[camera_id] = tracks
    
    def get_stats(self) -> Dict:
        """Get gallery statistics."""
        with self._lock:
            return {
                'confirmed_faces': len(self._faces),
                'candidates': len(self._candidates),
                'next_id': self._next_id
            }


# ============================================================================
# Face Detector with Size Filtering
# ============================================================================

class RobustFaceDetector:
    """
    Face detector with minimum size filtering for distance handling.
    """
    
    def __init__(self, config: Config):
        self.config = config
        self.device = torch.device(config.DEVICE)
        
        logger.info("Loading MTCNN face detector...")
        self.mtcnn = MTCNN(
            image_size=config.FACENET_INPUT_SIZE,
            margin=30,  # Increased margin for better face capture
            min_face_size=config.MIN_FACE_SIZE,
            thresholds=[0.5, 0.6, 0.6],  # Lowered thresholds for far distance
            factor=0.709,
            keep_all=True,
            device=self.device
        )
        logger.info(f"MTCNN loaded on {config.DEVICE}")
    
    def detect(self, frame: np.ndarray) -> List[FaceDetection]:
        """
        Detect faces in the frame.
        Returns all detected faces, but marks small ones as not suitable for ReID.
        """
        detections = []
        
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        boxes, probs = self.mtcnn.detect(rgb_frame)
        
        if boxes is None:
            return detections
        
        for box, prob in zip(boxes, probs):
            if prob < self.config.DETECTION_CONFIDENCE:
                continue
            
            x1, y1, x2, y2 = map(int, box)
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(frame.shape[1], x2)
            y2 = min(frame.shape[0], y2)
            
            if x2 <= x1 or y2 <= y1:
                continue
            
            face_crop = frame[y1:y2, x1:x2].copy()
            
            if face_crop.size == 0:
                continue
            
            detections.append(FaceDetection(
                bbox=(x1, y1, x2, y2),
                confidence=float(prob),
                face_crop=face_crop
            ))
        
        return detections
    
    def filter_for_reid(self, detections: List[FaceDetection]) -> Tuple[List[FaceDetection], List[FaceDetection]]:
        """
        Filter detections based on size for ReID.
        
        Returns:
            Tuple of (suitable_for_reid, too_small)
        """
        suitable = []
        too_small = []
        
        for det in detections:
            if min(det.width, det.height) >= self.config.MIN_FACE_SIZE_FOR_REID:
                suitable.append(det)
            else:
                too_small.append(det)
        
        return suitable, too_small


# ============================================================================
# Robust Face Embedding Engine with CLAHE
# ============================================================================

class RobustFaceEmbeddingEngine:
    """
    Face embedding extraction with CLAHE preprocessing for lighting robustness.
    All embeddings are L2-normalized.
    """
    
    def __init__(self, config: Config):
        self.config = config
        self.device = torch.device(config.DEVICE)
        
        logger.info("Loading FaceNet (InceptionResnetV1) model...")
        self.model = InceptionResnetV1(pretrained='vggface2').eval().to(self.device)
        logger.info(f"FaceNet loaded on {config.DEVICE}")
    
    def preprocess(self, face_crop: np.ndarray) -> torch.Tensor:
        """
        Preprocess face with CLAHE for lighting robustness.
        """
        # Apply CLAHE for lighting normalization
        clahe_crop = apply_clahe(
            face_crop, 
            self.config.CLAHE_CLIP_LIMIT,
            self.config.CLAHE_GRID_SIZE
        )
        
        # Resize to 160x160
        resized = cv2.resize(clahe_crop, (self.config.FACENET_INPUT_SIZE, self.config.FACENET_INPUT_SIZE))
        
        # Convert BGR to RGB
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        
        # Convert to tensor and normalize to [-1, 1]
        tensor = torch.from_numpy(rgb).float()
        tensor = tensor.permute(2, 0, 1)
        tensor = (tensor - 127.5) / 128.0
        tensor = tensor.unsqueeze(0).to(self.device)
        
        return tensor
    
    @torch.no_grad()
    def extract_embedding(self, face_crop: np.ndarray) -> np.ndarray:
        """Extract L2-normalized 512-D embedding."""
        tensor = self.preprocess(face_crop)
        embedding = self.model(tensor)
        embedding = F.normalize(embedding, p=2, dim=1)
        return embedding.cpu().numpy().flatten()
    
    @torch.no_grad()
    def extract_embeddings_batch(self, face_crops: List[np.ndarray]) -> List[np.ndarray]:
        """Extract L2-normalized embeddings for multiple faces."""
        if not face_crops:
            return []
        
        tensors = [self.preprocess(crop) for crop in face_crops]
        batch = torch.cat(tensors, dim=0)
        
        embeddings = self.model(batch)
        embeddings = F.normalize(embeddings, p=2, dim=1)
        
        return [e.cpu().numpy().flatten() for e in embeddings]


# ============================================================================
# Camera Stream Handler
# ============================================================================

class CameraStream:
    """Handles RTSP stream capture with automatic reconnection."""
    
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
        try:
            self.logger.info("Connecting to stream...")
            
            if self._cap is not None:
                self._cap.release()
            
            self._cap = cv2.VideoCapture(self.url, cv2.CAP_FFMPEG)
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
        while not self._stop_event.is_set():
            if not self._connected:
                if self._reconnect_attempts >= self.config.MAX_RECONNECT_ATTEMPTS:
                    self.logger.error("Max reconnection attempts reached")
                    break
                
                self._reconnect_attempts += 1
                self.logger.info(f"Reconnect attempt {self._reconnect_attempts}/{self.config.MAX_RECONNECT_ATTEMPTS}")
                
                if not self._connect():
                    time.sleep(self.config.RECONNECT_DELAY)
                    continue
            
            try:
                ret, frame = self._cap.read()
                
                if not ret or frame is None:
                    self.logger.warning("Frame read failed, reconnecting...")
                    self._connected = False
                    continue
                
                self._frame_count += 1
                
                if self._frame_count % self.config.FRAME_SKIP != 0:
                    continue
                
                if self._frame_queue.full():
                    try:
                        self._frame_queue.get_nowait()
                    except Empty:
                        pass
                self._frame_queue.put_nowait((self._frame_count, frame))
                    
            except Exception as e:
                self.logger.error(f"Capture error: {e}")
                self._connected = False
    
    def start(self):
        if self._thread is not None and self._thread.is_alive():
            return
        
        self._stop_event.clear()
        self._thread = Thread(target=self._capture_loop, daemon=True)
        self._thread.start()
        self.logger.info("Capture thread started")
    
    def stop(self):
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        if self._cap is not None:
            self._cap.release()
        self.logger.info("Capture thread stopped")
    
    def get_frame(self) -> Optional[Tuple[int, np.ndarray]]:
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
    """Handles rendering of face detections and IDs."""
    
    COLORS = [
        (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0),
        (255, 0, 255), (0, 255, 255), (128, 0, 0), (0, 128, 0),
        (0, 0, 128), (128, 128, 0), (128, 0, 128), (0, 128, 128),
    ]
    
    CANDIDATE_COLOR = (128, 128, 128)  # Gray for unconfirmed
    TOO_SMALL_COLOR = (100, 100, 100)  # Dark gray for too small
    
    def __init__(self, config: Config):
        self.config = config
    
    def get_color(self, person_id: int, is_confirmed: bool) -> Tuple[int, int, int]:
        if not is_confirmed:
            return self.CANDIDATE_COLOR
        return self.COLORS[person_id % len(self.COLORS)]
    
    def draw_detection(self, frame: np.ndarray, detection: FaceDetection, 
                       is_too_small: bool = False) -> np.ndarray:
        x1, y1, x2, y2 = detection.bbox
        
        if is_too_small:
            # Draw small faces differently
            cv2.rectangle(frame, (x1, y1), (x2, y2), self.TOO_SMALL_COLOR, 1)
            return frame
        
        person_id = detection.person_id or 0
        color = self.get_color(person_id, detection.is_confirmed)
        
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, self.config.BBOX_THICKNESS)
        
        # Label
        if detection.is_confirmed:
            label = f"ID: {person_id}"
        else:
            label = f"ID: ? ({abs(person_id)})"  # Show temp ID
        
        (label_w, label_h), _ = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, self.config.FONT_SCALE, 2
        )
        
        cv2.rectangle(
            frame,
            (x1, y1 - label_h - 10),
            (x1 + label_w + 5, y1),
            color,
            -1
        )
        
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
    
    def draw_info(self, frame: np.ndarray, camera_id: int, fps: float, 
                  gallery_stats: Dict) -> np.ndarray:
        info_text = (f"Cam {camera_id} | FPS: {fps:.1f} | "
                    f"Confirmed: {gallery_stats['confirmed_faces']} | "
                    f"Candidates: {gallery_stats['candidates']}")
        
        cv2.putText(
            frame,
            info_text,
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2
        )
        
        return frame


# ============================================================================
# Main Robust Face ReID System
# ============================================================================

class RobustMultiCameraFaceReIDSystem:
    """
    Main system with all robustness features:
    - CLAHE preprocessing
    - Momentum gallery updates
    - Temporal consistency
    - Hysteresis thresholds
    - Size filtering
    - IOU tracking
    """
    
    def __init__(self, config: Optional[Config] = None):
        self.config = config or Config()
        
        logger.info("Initializing Robust Multi-Camera Face ReID System...")
        
        self.detector = RobustFaceDetector(self.config)
        self.embedding_engine = RobustFaceEmbeddingEngine(self.config)
        self.gallery = RobustFaceGallery(self.config)
        self.visualizer = Visualizer(self.config)
        
        self.streams: List[CameraStream] = []
        for i, url in enumerate(self.config.CAMERA_URLS):
            stream = CameraStream(i, url, self.config)
            self.streams.append(stream)
        
        self._running = False
        
        logger.info(f"System initialized with {len(self.streams)} cameras")
    
    def _process_frame(self, camera_id: int, frame: np.ndarray) -> np.ndarray:
        """
        Process a single frame with all robustness features.
        """
        # Detect all faces
        all_detections = self.detector.detect(frame)
        
        # Filter by size for ReID
        reid_detections, small_detections = self.detector.filter_for_reid(all_detections)
        
        # Track list for IOU updates
        current_tracks: List[Tuple[int, Tuple[int, int, int, int]]] = []
        
        if reid_detections:
            # Extract embeddings (with CLAHE preprocessing)
            face_crops = [d.face_crop for d in reid_detections]
            embeddings = self.embedding_engine.extract_embeddings_batch(face_crops)
            
            # Match and update gallery
            for detection, embedding in zip(reid_detections, embeddings):
                detection.embedding = embedding
                person_id, is_confirmed = self.gallery.match_and_update(
                    embedding, detection.bbox, camera_id
                )
                detection.person_id = person_id
                detection.is_confirmed = is_confirmed
                
                # Add to track list for IOU memory
                if is_confirmed:
                    current_tracks.append((person_id, detection.bbox))
        
        # Update IOU track memory
        self.gallery.update_tracks(camera_id, current_tracks)
        
        # Visualize
        for detection in reid_detections:
            frame = self.visualizer.draw_detection(frame, detection)
        
        for detection in small_detections:
            frame = self.visualizer.draw_detection(frame, detection, is_too_small=True)
        
        return frame
    
    def run(self):
        logger.info("Starting Robust Multi-Camera Face ReID System...")
        self._running = True
        
        for stream in self.streams:
            stream.start()
        
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
                    
                    current_time = time.time()
                    fps = 1.0 / (current_time - last_times[i] + 1e-8)
                    last_times[i] = current_time
                    
                    processed_frame = self._process_frame(i, frame)
                    
                    processed_frame = self.visualizer.draw_info(
                        processed_frame, i, fps, self.gallery.get_stats()
                    )
                    
                    cv2.imshow(f"Camera {i}", processed_frame)
                
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    logger.info("Quit requested")
                    break
                elif key == ord('s'):
                    stats = self.gallery.get_stats()
                    logger.info(f"Gallery Stats: {stats}")
        
        except KeyboardInterrupt:
            logger.info("Interrupted by user")
        
        finally:
            self.stop()
    
    def stop(self):
        logger.info("Stopping system...")
        self._running = False
        
        for stream in self.streams:
            stream.stop()
        
        cv2.destroyAllWindows()
        logger.info("System stopped")


# ============================================================================
# Entry Point
# ============================================================================

def main():
    print("=" * 70)
    print("Robust Multi-Camera Face Re-identification System")
    print("=" * 70)
    print()
    print("Robustness Features:")
    print("  ✓ CLAHE preprocessing for lighting/shadow normalization")
    print("  ✓ Momentum-based gallery updates (centroid memory)")
    print("  ✓ Temporal consistency (wait-and-see buffer)")
    print("  ✓ Hysteresis thresholds (high to create, low to maintain)")
    print("  ✓ Minimum face size filtering (distance handling)")
    print("  ✓ IOU-based tracking integration")
    print()
    print("Controls:")
    print("  q - Quit")
    print("  s - Print gallery statistics")
    print()
    
    config = Config()
    
    print("Configuration:")
    print(f"  Cameras: {len(config.CAMERA_URLS)}")
    print(f"  Device: {config.DEVICE}")
    print(f"  High Threshold (create): {config.HIGH_THRESHOLD}")
    print(f"  Low Threshold (maintain): {config.LOW_THRESHOLD}")
    print(f"  Min Face Size for ReID: {config.MIN_FACE_SIZE_FOR_REID}px")
    print(f"  Frames to Confirm: {config.MIN_FRAMES_TO_CONFIRM}")
    print(f"  Momentum Alpha: {config.MOMENTUM_ALPHA}")
    print(f"  CLAHE Clip Limit: {config.CLAHE_CLIP_LIMIT}")
    print()
    
    system = RobustMultiCameraFaceReIDSystem(config)
    system.run()


if __name__ == "__main__":
    main()
