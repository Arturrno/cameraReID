"""
Optimized Multi-Camera Face Re-identification System
=====================================================
Enhanced system with advanced robustness features for consistent
face recognition across multiple RTSP cameras with varying lighting.

Optimizations:
- Enhanced preprocessing: CLAHE + Gamma Correction + LAB normalization
- Multi-embedding profiles (cluster of N diverse embeddings per person)
- Cosine similarity with adaptive thresholds
- Temporal decay and spatial priority for cross-camera handover
- CUDA-accelerated preprocessing for Jetson Orin Nano
- Camera-specific lighting profile calibration

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
from typing import Dict, List, Optional, Tuple, Set
from collections import OrderedDict, defaultdict
from enum import Enum
import time
import logging
import math

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ============================================================================
# Check for CUDA availability (for Jetson Orin Nano optimization)
# ============================================================================

def check_cuda_opencv():
    """Check if OpenCV was built with CUDA support."""
    try:
        count = cv2.cuda.getCudaEnabledDeviceCount()
        return count > 0
    except:
        return False

OPENCV_CUDA_AVAILABLE = check_cuda_opencv()
logger.info(f"OpenCV CUDA available: {OPENCV_CUDA_AVAILABLE}")


# ============================================================================
# Configuration
# ============================================================================

@dataclass
class OptimizedConfig:
    """Enhanced configuration for the Optimized Face ReID system."""
    
    # RTSP Stream URLs - USE MAIN STREAM FOR FULL RESOLUTION
    # Change 'sub' to 'main' for full 4K resolution
    # Common patterns: Preview_01_main, Streaming/Channels/101, cam/realmonitor?channel=1&subtype=0
    CAMERA_URLS: List[str] = field(default_factory=lambda: [
        'rtsp://admin:admin123@192.168.1.102:554/Preview_01_sub',   # Camera 0: Using sub stream (has network issues)
        'rtsp://admin:admin123@192.168.1.103:554/Preview_01_main',  # Camera 1: MAIN = 4K (works fine)
        # CAMERAS 3 & 4 DISABLED - uncomment when needed:
        # 'rtsp://admin:admin123@192.168.1.104:554/Preview_01_main',
        # 'rtsp://admin:admin123@192.168.1.105:554/Preview_01_main',
    ])
    
    # Camera Room Adjacency Map (for spatial priority)
    # Format: camera_id -> list of adjacent camera_ids
    CAMERA_ADJACENCY: Dict[int, List[int]] = field(default_factory=lambda: {
        0: [1],      # Camera 0 is adjacent to Camera 1
        1: [0, 2],   # Camera 1 is adjacent to Cameras 0 and 2
        2: [1, 3],   # Camera 2 is adjacent to Cameras 1 and 3
        3: [2],      # Camera 3 is adjacent to Camera 2
    })
    
    # Face Detection settings (MTCNN) - Optimized for long-distance security cameras
    DETECTION_CONFIDENCE: float = 0.7  # Lowered for distant faces
    MIN_FACE_SIZE: int = 12  # Minimum pixels for MTCNN to detect (very small for distance)
    MIN_FACE_SIZE_FOR_REID: int = 20  # Minimum face size for ReID (lowered for distance)
    
    # Detection scaling - downscale 4K for faster MTCNN, crop from original
    DETECTION_SCALE: float = 0.5  # Process detection at 50% resolution (e.g., 4K -> 1080p)
    MAX_DETECTION_WIDTH: int = 1920  # Max width for detection (scales down if larger)
    
    # Face Embedding settings
    FACENET_INPUT_SIZE: int = 160  # FaceNet model input (fixed)
    FACE_CROP_SIZE: int = 256  # Size for preprocessing (before resize to 160)
    FACE_MARGIN_PERCENT: float = 0.2  # 20% margin (smaller for distant faces)
    EMBEDDING_DIM: int = 512
    
    # Super-resolution / Enhancement for small faces
    ENHANCE_SMALL_FACES: bool = True  # Apply enhancement to small face crops
    SMALL_FACE_THRESHOLD: int = 80  # Faces smaller than this get enhanced
    UPSCALE_FACTOR: float = 2.0  # Upscale small faces before processing
    
    # Multi-Embedding Profile Settings
    MAX_EMBEDDINGS_PER_PROFILE: int = 10  # Store up to N diverse embeddings per person
    EMBEDDING_DIVERSITY_THRESHOLD: float = 0.15  # Min distance to add new embedding
    
    # Hysteresis Thresholds (base values - will be adapted per camera)
    BASE_HIGH_THRESHOLD: float = 0.72  # Base threshold to CREATE new ID
    BASE_LOW_THRESHOLD: float = 0.52   # Base threshold to MAINTAIN existing ID
    
    # Per-camera threshold adjustments (calibrated based on lighting)
    CAMERA_THRESHOLD_OFFSETS: Dict[int, float] = field(default_factory=lambda: {
        0: 0.0,   # Well-lit room
        1: -0.03, # Darker room - lower thresholds
        2: -0.05, # Very dark room
        3: 0.02,  # Bright room - higher thresholds
    })
    
    # Temporal Decay Settings
    TEMPORAL_DECAY_RATE: float = 0.1  # Decay per second
    TEMPORAL_BOOST_ADJACENT: float = 0.08  # Boost for adjacent camera match
    TEMPORAL_BOOST_RECENT: float = 0.12  # Boost if seen in last N seconds
    RECENT_SEEN_WINDOW: float = 10.0  # Seconds to consider "recently seen"
    
    # Momentum-based Gallery Updates
    MOMENTUM_ALPHA: float = 0.15
    
    # Temporal Consistency - Wait-and-See Buffer
    MIN_FRAMES_TO_CONFIRM: int = 4
    CANDIDATE_TIMEOUT: float = 3.0
    
    # IOU Tracking
    IOU_THRESHOLD: float = 0.3
    IOU_SIMILARITY_BOOST: float = 0.08
    
    # Enhanced Preprocessing Settings
    CLAHE_CLIP_LIMIT: float = 3.0
    CLAHE_GRID_SIZE: Tuple[int, int] = (8, 8)
    GAMMA_CORRECTION_TARGET: float = 0.5  # Target average brightness (0-1)
    GAMMA_RANGE: Tuple[float, float] = (0.4, 2.5)  # Min/max gamma values
    HISTOGRAM_EQUALIZATION: bool = True
    COLOR_NORMALIZATION: bool = True
    
    # Performance settings
    FRAME_SKIP: int = 2  # Process every 2nd frame
    MAX_GALLERY_SIZE: int = 500
    GALLERY_CLEANUP_THRESHOLD: int = 50
    USE_CUDA_PREPROCESSING: bool = True  # Use CUDA if available
    
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
    is_confirmed: bool = True
    
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
class EmbeddingProfile:
    """
    Multi-embedding profile storing diverse embeddings for a single person.
    Uses a cluster of embeddings captured at different angles/lighting.
    """
    embeddings: List[np.ndarray] = field(default_factory=list)
    camera_sources: List[int] = field(default_factory=list)  # Which camera each embedding came from
    timestamps: List[float] = field(default_factory=list)
    
    def add_embedding(self, embedding: np.ndarray, camera_id: int, 
                      max_size: int = 10, diversity_threshold: float = 0.15) -> bool:
        """
        Add a new embedding if it's sufficiently different from existing ones.
        Returns True if embedding was added.
        """
        embedding = embedding / (np.linalg.norm(embedding) + 1e-8)
        
        if len(self.embeddings) == 0:
            self.embeddings.append(embedding)
            self.camera_sources.append(camera_id)
            self.timestamps.append(time.time())
            return True
        
        # Check diversity - embedding should be different enough from existing
        min_distance = float('inf')
        for existing in self.embeddings:
            distance = 1.0 - np.dot(embedding, existing)
            min_distance = min(min_distance, distance)
        
        if min_distance >= diversity_threshold:
            if len(self.embeddings) >= max_size:
                # Replace oldest embedding
                oldest_idx = np.argmin(self.timestamps)
                self.embeddings[oldest_idx] = embedding
                self.camera_sources[oldest_idx] = camera_id
                self.timestamps[oldest_idx] = time.time()
            else:
                self.embeddings.append(embedding)
                self.camera_sources.append(camera_id)
                self.timestamps.append(time.time())
            return True
        
        return False
    
    def get_centroid(self) -> np.ndarray:
        """Get the centroid of all embeddings."""
        if not self.embeddings:
            return np.zeros(512)
        centroid = np.mean(self.embeddings, axis=0)
        return centroid / (np.linalg.norm(centroid) + 1e-8)
    
    def compute_similarity(self, query_embedding: np.ndarray) -> float:
        """
        Compute maximum cosine similarity against all embeddings in profile.
        Uses max similarity for better cross-lighting robustness.
        """
        if not self.embeddings:
            return 0.0
        
        query_norm = query_embedding / (np.linalg.norm(query_embedding) + 1e-8)
        
        # Compute similarity against all embeddings
        similarities = [np.dot(query_norm, emb) for emb in self.embeddings]
        
        # Return weighted combination: max + average boost
        max_sim = max(similarities)
        avg_sim = np.mean(similarities)
        
        # Weight towards max but include average for stability
        return 0.7 * max_sim + 0.3 * avg_sim


@dataclass
class TrackedFace:
    """Represents a confirmed face in the global gallery."""
    person_id: int
    profile: EmbeddingProfile  # Multi-embedding profile
    last_seen: float = field(default_factory=time.time)
    last_camera_id: int = -1
    hit_count: int = 1
    last_bbox: Optional[Tuple[int, int, int, int]] = None
    camera_history: List[Tuple[int, float]] = field(default_factory=list)  # (camera_id, timestamp)
    
    def update(self, embedding: np.ndarray, camera_id: int, bbox: Tuple[int, int, int, int],
               config: OptimizedConfig):
        """Update the tracked face with new observation."""
        # Add to profile if diverse enough
        self.profile.add_embedding(
            embedding, camera_id,
            config.MAX_EMBEDDINGS_PER_PROFILE,
            config.EMBEDDING_DIVERSITY_THRESHOLD
        )
        
        self.last_seen = time.time()
        self.last_camera_id = camera_id
        self.last_bbox = bbox
        self.hit_count += 1
        
        # Update camera history (keep last 20 entries)
        self.camera_history.append((camera_id, time.time()))
        if len(self.camera_history) > 20:
            self.camera_history = self.camera_history[-20:]
    
    def was_recently_at_camera(self, camera_id: int, window_seconds: float) -> bool:
        """Check if this person was recently seen at a specific camera."""
        current_time = time.time()
        for cam_id, timestamp in reversed(self.camera_history):
            if current_time - timestamp > window_seconds:
                break
            if cam_id == camera_id:
                return True
        return False
    
    def get_recent_cameras(self, window_seconds: float) -> Set[int]:
        """Get set of cameras where person was recently seen."""
        current_time = time.time()
        recent = set()
        for cam_id, timestamp in reversed(self.camera_history):
            if current_time - timestamp > window_seconds:
                break
            recent.add(cam_id)
        return recent


@dataclass
class CandidateFace:
    """Represents a candidate face in the wait-and-see buffer."""
    temp_id: int
    profile: EmbeddingProfile
    bboxes: List[Tuple[int, int, int, int]] = field(default_factory=list)
    first_seen: float = field(default_factory=time.time)
    last_seen: float = field(default_factory=time.time)
    camera_id: int = -1
    frame_count: int = 0
    
    def add_observation(self, embedding: np.ndarray, bbox: Tuple[int, int, int, int],
                        camera_id: int, config: OptimizedConfig):
        """Add a new observation to this candidate."""
        self.profile.add_embedding(
            embedding, camera_id,
            config.MAX_EMBEDDINGS_PER_PROFILE,
            config.EMBEDDING_DIVERSITY_THRESHOLD
        )
        self.bboxes.append(bbox)
        self.last_seen = time.time()
        self.camera_id = camera_id
        self.frame_count += 1
    
    def get_last_bbox(self) -> Optional[Tuple[int, int, int, int]]:
        return self.bboxes[-1] if self.bboxes else None


# ============================================================================
# Enhanced Preprocessing Engine (CUDA-optimized)
# ============================================================================

class EnhancedPreprocessor:
    """
    Advanced image preprocessing for lighting invariance.
    Supports CUDA acceleration for Jetson Orin Nano.
    """
    
    def __init__(self, config: OptimizedConfig):
        self.config = config
        self.use_cuda = config.USE_CUDA_PREPROCESSING and OPENCV_CUDA_AVAILABLE
        
        if self.use_cuda:
            logger.info("Using CUDA-accelerated preprocessing")
            # Pre-allocate CUDA matrices for performance
            self._gpu_mat = cv2.cuda_GpuMat()
            self._gpu_lab = cv2.cuda_GpuMat()
            self._gpu_result = cv2.cuda_GpuMat()
            
            # CUDA CLAHE
            self._cuda_clahe = cv2.cuda.createCLAHE(
                clipLimit=config.CLAHE_CLIP_LIMIT,
                tileGridSize=config.CLAHE_GRID_SIZE
            )
        else:
            logger.info("Using CPU preprocessing")
            self._clahe = cv2.createCLAHE(
                clipLimit=config.CLAHE_CLIP_LIMIT,
                tileGridSize=config.CLAHE_GRID_SIZE
            )
        
        # Per-camera brightness calibration (learned online)
        self._camera_brightness_stats: Dict[int, List[float]] = defaultdict(list)
        self._camera_gamma_cache: Dict[int, float] = {}
    
    def _estimate_gamma(self, image: np.ndarray, target: float = 0.5) -> float:
        """Estimate gamma correction value to achieve target brightness."""
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        # Calculate current average brightness (normalized 0-1)
        current_brightness = np.mean(gray) / 255.0
        
        if current_brightness < 0.01:
            return self.config.GAMMA_RANGE[1]  # Very dark, max gamma
        
        # Estimate gamma: target = current^gamma => gamma = log(target)/log(current)
        try:
            gamma = math.log(target) / math.log(current_brightness)
            gamma = np.clip(gamma, self.config.GAMMA_RANGE[0], self.config.GAMMA_RANGE[1])
        except:
            gamma = 1.0
        
        return gamma
    
    def _apply_gamma_correction(self, image: np.ndarray, gamma: float) -> np.ndarray:
        """Apply gamma correction using lookup table (fast)."""
        if abs(gamma - 1.0) < 0.05:
            return image
        
        # Build lookup table
        inv_gamma = 1.0 / gamma
        table = np.array([
            ((i / 255.0) ** inv_gamma) * 255
            for i in np.arange(0, 256)
        ]).astype("uint8")
        
        return cv2.LUT(image, table)
    
    def _apply_clahe_lab(self, image: np.ndarray) -> np.ndarray:
        """Apply CLAHE to L channel in LAB color space."""
        # Convert to LAB
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        
        # Split channels
        l, a, b = cv2.split(lab)
        
        # Apply CLAHE to L channel
        if self.use_cuda:
            self._gpu_mat.upload(l)
            l_clahe = self._cuda_clahe.apply(self._gpu_mat, cv2.cuda_Stream.Null())
            l = l_clahe.download()
        else:
            l = self._clahe.apply(l)
        
        # Merge and convert back
        lab = cv2.merge([l, a, b])
        result = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        
        return result
    
    def _normalize_color(self, image: np.ndarray) -> np.ndarray:
        """Normalize color distribution to reduce lighting color cast."""
        # Convert to LAB
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB).astype(np.float32)
        
        # Normalize a and b channels to center around 128
        l, a, b = cv2.split(lab)
        
        # Shift a and b to be centered (remove color cast)
        a = a - np.mean(a) + 128
        b = b - np.mean(b) + 128
        
        # Clip values
        l = np.clip(l, 0, 255).astype(np.uint8)
        a = np.clip(a, 0, 255).astype(np.uint8)
        b = np.clip(b, 0, 255).astype(np.uint8)
        
        # Merge and convert back
        lab = cv2.merge([l, a, b])
        result = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        
        return result
    
    def update_camera_stats(self, camera_id: int, image: np.ndarray):
        """Update brightness statistics for a camera (for adaptive gamma)."""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        brightness = np.mean(gray) / 255.0
        
        stats = self._camera_brightness_stats[camera_id]
        stats.append(brightness)
        
        # Keep last 100 samples
        if len(stats) > 100:
            self._camera_brightness_stats[camera_id] = stats[-100:]
        
        # Update cached gamma for this camera
        if len(stats) >= 10:
            avg_brightness = np.mean(stats[-50:])
            if avg_brightness > 0.01:
                try:
                    gamma = math.log(self.config.GAMMA_CORRECTION_TARGET) / math.log(avg_brightness)
                    gamma = np.clip(gamma, self.config.GAMMA_RANGE[0], self.config.GAMMA_RANGE[1])
                    self._camera_gamma_cache[camera_id] = gamma
                except:
                    pass
    
    def preprocess(self, face_crop: np.ndarray, camera_id: int = -1) -> np.ndarray:
        """
        Full preprocessing pipeline for lighting invariance.
        
        Pipeline:
        1. Gamma correction (adaptive or from camera cache)
        2. CLAHE on LAB L-channel
        3. Color normalization (optional)
        """
        if face_crop is None or face_crop.size == 0:
            return face_crop
        
        result = face_crop.copy()
        
        # 1. Gamma Correction
        if camera_id in self._camera_gamma_cache:
            gamma = self._camera_gamma_cache[camera_id]
        else:
            gamma = self._estimate_gamma(result, self.config.GAMMA_CORRECTION_TARGET)
        
        result = self._apply_gamma_correction(result, gamma)
        
        # 2. CLAHE on LAB L-channel
        result = self._apply_clahe_lab(result)
        
        # 3. Color normalization (reduce color cast from lighting)
        if self.config.COLOR_NORMALIZATION:
            result = self._normalize_color(result)
        
        return result
    
    def preprocess_batch(self, face_crops: List[np.ndarray], camera_id: int = -1) -> List[np.ndarray]:
        """Process multiple face crops (vectorized where possible)."""
        return [self.preprocess(crop, camera_id) for crop in face_crops]


# ============================================================================
# Optimized Face Embedding Engine
# ============================================================================

class OptimizedFaceEmbeddingEngine:
    """
    Face embedding extraction with enhanced preprocessing.
    All embeddings are L2-normalized for cosine similarity.
    """
    
    def __init__(self, config: OptimizedConfig):
        self.config = config
        self.device = torch.device(config.DEVICE)
        self.preprocessor = EnhancedPreprocessor(config)
        
        logger.info("Loading FaceNet (InceptionResnetV1) model...")
        self.model = InceptionResnetV1(pretrained='vggface2').eval().to(self.device)
        
        # Enable half precision on CUDA for Jetson optimization
        if config.DEVICE == 'cuda':
            try:
                self.model = self.model.half()
                self._use_half = True
                logger.info("Using FP16 for Jetson optimization")
            except:
                self._use_half = False
        else:
            self._use_half = False
        
        logger.info(f"FaceNet loaded on {config.DEVICE}")
    
    def _to_tensor(self, face_crop: np.ndarray) -> torch.Tensor:
        """Convert preprocessed face crop to tensor with high-quality downscaling."""
        h, w = face_crop.shape[:2]
        target_size = self.config.FACENET_INPUT_SIZE
        
        # First resize to intermediate size if much larger (preserves detail)
        if min(h, w) > target_size * 2:
            # Two-step resize for better quality from high-res
            intermediate = target_size * 2
            scale = intermediate / min(h, w)
            inter_w, inter_h = int(w * scale), int(h * scale)
            face_crop = cv2.resize(face_crop, (inter_w, inter_h), interpolation=cv2.INTER_AREA)
        
        # Final resize to 160x160 using INTER_AREA (best for downscaling)
        resized = cv2.resize(face_crop, (target_size, target_size), interpolation=cv2.INTER_AREA)
        
        # Convert BGR to RGB
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        
        # Convert to tensor and normalize to [-1, 1]
        tensor = torch.from_numpy(rgb).float()
        tensor = tensor.permute(2, 0, 1)
        tensor = (tensor - 127.5) / 128.0
        tensor = tensor.unsqueeze(0)
        
        if self._use_half:
            tensor = tensor.half()
        
        return tensor.to(self.device)
    
    @torch.no_grad()
    def extract_embedding(self, face_crop: np.ndarray, camera_id: int = -1) -> np.ndarray:
        """Extract L2-normalized 512-D embedding with preprocessing."""
        # Apply enhanced preprocessing
        preprocessed = self.preprocessor.preprocess(face_crop, camera_id)
        
        # Convert to tensor
        tensor = self._to_tensor(preprocessed)
        
        # Extract embedding
        embedding = self.model(tensor)
        
        if self._use_half:
            embedding = embedding.float()
        
        embedding = F.normalize(embedding, p=2, dim=1)
        return embedding.cpu().numpy().flatten()
    
    @torch.no_grad()
    def extract_embeddings_batch(self, face_crops: List[np.ndarray], 
                                  camera_id: int = -1) -> List[np.ndarray]:
        """Extract L2-normalized embeddings for multiple faces."""
        if not face_crops:
            return []
        
        # Preprocess all crops
        preprocessed = self.preprocessor.preprocess_batch(face_crops, camera_id)
        
        # Convert to batch tensor
        tensors = [self._to_tensor(crop) for crop in preprocessed]
        batch = torch.cat(tensors, dim=0)
        
        # Extract embeddings
        embeddings = self.model(batch)
        
        if self._use_half:
            embeddings = embeddings.float()
        
        embeddings = F.normalize(embeddings, p=2, dim=1)
        
        return [e.cpu().numpy().flatten() for e in embeddings]
    
    def update_camera_stats(self, camera_id: int, frame: np.ndarray):
        """Update camera brightness statistics for adaptive preprocessing."""
        self.preprocessor.update_camera_stats(camera_id, frame)


# ============================================================================
# Optimized Face Gallery with Multi-Embedding Profiles
# ============================================================================

class OptimizedFaceGallery:
    """
    Thread-safe face gallery with:
    - Multi-embedding profiles (cluster of diverse embeddings per person)
    - Cosine similarity matching
    - Temporal decay and spatial priority
    - Adaptive per-camera thresholds
    - Cross-camera handover optimization
    """
    
    def __init__(self, config: OptimizedConfig):
        self.config = config
        
        # Confirmed faces (global gallery)
        self._faces: OrderedDict[int, TrackedFace] = OrderedDict()
        
        # Candidate buffer (wait-and-see)
        self._candidates: Dict[int, CandidateFace] = {}
        
        # Per-camera track memory for IOU matching
        self._last_frame_tracks: Dict[int, List[Tuple[int, Tuple[int, int, int, int]]]] = defaultdict(list)
        
        self._lock = Lock()
        self._next_id = 1
        self._next_temp_id = -1
    
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
    
    def _get_adaptive_thresholds(self, camera_id: int) -> Tuple[float, float]:
        """Get camera-specific adaptive thresholds."""
        offset = self.config.CAMERA_THRESHOLD_OFFSETS.get(camera_id, 0.0)
        high = self.config.BASE_HIGH_THRESHOLD + offset
        low = self.config.BASE_LOW_THRESHOLD + offset
        return high, low
    
    def _compute_temporal_boost(self, face: TrackedFace, camera_id: int) -> float:
        """
        Compute temporal/spatial boost for cross-camera consistency.
        Higher boost if person was recently seen at this or adjacent camera.
        """
        boost = 0.0
        current_time = time.time()
        
        # Boost if recently seen anywhere
        time_since_seen = current_time - face.last_seen
        if time_since_seen < self.config.RECENT_SEEN_WINDOW:
            # Linear decay
            recency_factor = 1.0 - (time_since_seen / self.config.RECENT_SEEN_WINDOW)
            boost += self.config.TEMPORAL_BOOST_RECENT * recency_factor
        
        # Boost if last seen at adjacent camera
        adjacent_cameras = self.config.CAMERA_ADJACENCY.get(camera_id, [])
        if face.last_camera_id in adjacent_cameras:
            # Higher boost for adjacent camera handover
            if time_since_seen < self.config.RECENT_SEEN_WINDOW:
                boost += self.config.TEMPORAL_BOOST_ADJACENT
        
        # Small boost if same camera (tracking continuation)
        if face.last_camera_id == camera_id:
            boost += 0.03
        
        return boost
    
    def _find_iou_match(self, bbox: Tuple[int, int, int, int], 
                        camera_id: int) -> Optional[int]:
        """Find a person ID based on IOU with previous frame tracks."""
        best_iou = 0.0
        best_id = None
        
        for person_id, last_bbox in self._last_frame_tracks[camera_id]:
            iou = self._compute_iou(bbox, last_bbox)
            if iou > best_iou and iou >= self.config.IOU_THRESHOLD:
                best_iou = iou
                best_id = person_id
        
        return best_id
    
    @staticmethod
    def _compute_iou(box1: Tuple[int, int, int, int], 
                     box2: Tuple[int, int, int, int]) -> float:
        """Compute Intersection over Union between two bounding boxes."""
        x1_1, y1_1, x2_1, y2_1 = box1
        x1_2, y1_2, x2_2, y2_2 = box2
        
        xi1 = max(x1_1, x1_2)
        yi1 = max(y1_1, y1_2)
        xi2 = min(x2_1, x2_2)
        yi2 = min(y2_1, y2_2)
        
        if xi2 <= xi1 or yi2 <= yi1:
            return 0.0
        
        intersection = (xi2 - xi1) * (yi2 - yi1)
        area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
        area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
        union = area1 + area2 - intersection
        
        return intersection / (union + 1e-8)
    
    def _find_gallery_match(self, embedding: np.ndarray, 
                           bbox: Tuple[int, int, int, int],
                           camera_id: int) -> Tuple[Optional[int], float]:
        """
        Find best match in confirmed gallery using multi-embedding profiles.
        Applies temporal/spatial boosts for cross-camera consistency.
        """
        best_match_id = None
        best_similarity = 0.0
        
        # Check for IOU match first
        iou_matched_id = self._find_iou_match(bbox, camera_id)
        
        for person_id, face in self._faces.items():
            # Compute similarity against multi-embedding profile
            similarity = face.profile.compute_similarity(embedding)
            
            # Apply temporal/spatial boost
            temporal_boost = self._compute_temporal_boost(face, camera_id)
            similarity = min(1.0, similarity + temporal_boost)
            
            # Apply IOU boost if this face matches spatially
            if person_id == iou_matched_id:
                similarity = min(1.0, similarity + self.config.IOU_SIMILARITY_BOOST)
            
            if similarity > best_similarity:
                best_similarity = similarity
                best_match_id = person_id
        
        return best_match_id, best_similarity
    
    def _find_candidate_match(self, embedding: np.ndarray,
                              bbox: Tuple[int, int, int, int]) -> Tuple[Optional[int], float]:
        """Find best match among candidates using multi-embedding profiles."""
        best_match_id = None
        best_similarity = 0.0
        
        for temp_id, candidate in self._candidates.items():
            similarity = candidate.profile.compute_similarity(embedding)
            
            # IOU boost
            last_bbox = candidate.get_last_bbox()
            if last_bbox:
                iou = self._compute_iou(bbox, last_bbox)
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
        
        # Create confirmed face with candidate's profile
        new_face = TrackedFace(
            person_id=new_id,
            profile=candidate.profile,
            last_camera_id=candidate.camera_id,
            hit_count=candidate.frame_count,
            last_bbox=candidate.get_last_bbox()
        )
        self._faces[new_id] = new_face
        
        logger.info(f"Promoted candidate {temp_id} -> confirmed ID {new_id} "
                   f"(frames={candidate.frame_count}, embeddings={len(candidate.profile.embeddings)})")
        
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
    
    def _cleanup_gallery_if_needed(self):
        """Remove oldest entries if gallery exceeds maximum size."""
        if len(self._faces) > self.config.MAX_GALLERY_SIZE:
            items_to_remove = len(self._faces) - self.config.MAX_GALLERY_SIZE + self.config.GALLERY_CLEANUP_THRESHOLD
            
            # Sort by last_seen and remove oldest
            sorted_faces = sorted(self._faces.items(), key=lambda x: x[1].last_seen)
            for person_id, _ in sorted_faces[:items_to_remove]:
                del self._faces[person_id]
                logger.debug(f"Evicted face {person_id} from gallery")
    
    def match_and_update(self, embedding: np.ndarray, 
                         bbox: Tuple[int, int, int, int],
                         camera_id: int) -> Tuple[int, bool]:
        """
        Match embedding against gallery and candidates.
        Uses multi-embedding profiles and adaptive thresholds.
        
        Returns:
            Tuple of (person_id, is_confirmed)
        """
        with self._lock:
            self._cleanup_expired_candidates()
            
            # Get camera-specific thresholds
            high_threshold, low_threshold = self._get_adaptive_thresholds(camera_id)
            
            # 1. Try to match with confirmed gallery
            gallery_match_id, gallery_sim = self._find_gallery_match(embedding, bbox, camera_id)
            
            if gallery_match_id is not None and gallery_sim >= low_threshold:
                face = self._faces[gallery_match_id]
                face.update(embedding, camera_id, bbox, self.config)
                self._faces.move_to_end(gallery_match_id)
                
                logger.debug(f"Matched gallery ID {gallery_match_id} (sim={gallery_sim:.3f})")
                return gallery_match_id, True
            
            # 2. Try to match with candidates
            cand_match_id, cand_sim = self._find_candidate_match(embedding, bbox)
            
            if cand_match_id is not None and cand_sim >= low_threshold:
                candidate = self._candidates[cand_match_id]
                candidate.add_observation(embedding, bbox, camera_id, self.config)
                
                # Check if candidate should be promoted
                if candidate.frame_count >= self.config.MIN_FRAMES_TO_CONFIRM:
                    new_id = self._promote_candidate(cand_match_id)
                    self._cleanup_gallery_if_needed()
                    return new_id, True
                else:
                    return cand_match_id, False
            
            # 3. Create new candidate if no good match (using high threshold)
            if gallery_sim < high_threshold and cand_sim < high_threshold:
                temp_id = self._generate_temp_id()
                profile = EmbeddingProfile()
                profile.add_embedding(embedding, camera_id,
                                      self.config.MAX_EMBEDDINGS_PER_PROFILE,
                                      self.config.EMBEDDING_DIVERSITY_THRESHOLD)
                
                new_candidate = CandidateFace(
                    temp_id=temp_id,
                    profile=profile,
                    bboxes=[bbox],
                    camera_id=camera_id,
                    frame_count=1
                )
                self._candidates[temp_id] = new_candidate
                
                logger.debug(f"Created new candidate {temp_id}")
                return temp_id, False
            
            # Edge case handling
            if gallery_sim >= cand_sim and gallery_match_id is not None:
                face = self._faces[gallery_match_id]
                face.update(embedding, camera_id, bbox, self.config)
                return gallery_match_id, True
            elif cand_match_id is not None:
                candidate = self._candidates[cand_match_id]
                candidate.add_observation(embedding, bbox, camera_id, self.config)
                return cand_match_id, False
            else:
                temp_id = self._generate_temp_id()
                profile = EmbeddingProfile()
                profile.add_embedding(embedding, camera_id,
                                      self.config.MAX_EMBEDDINGS_PER_PROFILE,
                                      self.config.EMBEDDING_DIVERSITY_THRESHOLD)
                
                new_candidate = CandidateFace(
                    temp_id=temp_id,
                    profile=profile,
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
            total_embeddings = sum(
                len(face.profile.embeddings) for face in self._faces.values()
            )
            return {
                'confirmed_faces': len(self._faces),
                'candidates': len(self._candidates),
                'total_embeddings': total_embeddings,
                'next_id': self._next_id
            }
    
    def get_person_info(self, person_id: int) -> Optional[Dict]:
        """Get detailed info about a person."""
        with self._lock:
            if person_id in self._faces:
                face = self._faces[person_id]
                return {
                    'person_id': person_id,
                    'hit_count': face.hit_count,
                    'last_camera': face.last_camera_id,
                    'last_seen': face.last_seen,
                    'embedding_count': len(face.profile.embeddings),
                    'cameras_seen': list(set(face.profile.camera_sources))
                }
            return None


# ============================================================================
# Face Detector (reuse from original)
# ============================================================================

class RobustFaceDetector:
    """Face detector optimized for long-distance detection from security cameras."""
    
    def __init__(self, config: OptimizedConfig):
        self.config = config
        self.device = torch.device(config.DEVICE)
        
        logger.info("Loading MTCNN face detector (optimized for distance)...")
        
        # Calculate effective min_face_size for scaled detection
        # Since we scale down for detection, we need smaller min_face_size
        effective_min_face = max(8, int(config.MIN_FACE_SIZE * config.DETECTION_SCALE))
        
        # MTCNN with aggressive settings for small/distant faces
        self.mtcnn = MTCNN(
            image_size=config.FACE_CROP_SIZE,
            margin=0,  # We handle margin manually
            min_face_size=effective_min_face,  # Scaled for detection resolution
            thresholds=[0.4, 0.5, 0.5],  # Lower thresholds for distant faces
            factor=0.65,  # Smaller factor = more pyramid levels = better small face detection
            keep_all=True,
            device=self.device
        )
        logger.info(f"MTCNN loaded on {config.DEVICE} (min_face={effective_min_face}px at {config.DETECTION_SCALE:.0%} scale)")
    
    def _enhance_small_face(self, face_crop: np.ndarray) -> np.ndarray:
        """Enhance small face crops using upscaling and sharpening."""
        h, w = face_crop.shape[:2]
        
        if not self.config.ENHANCE_SMALL_FACES:
            return face_crop
        
        if min(h, w) >= self.config.SMALL_FACE_THRESHOLD:
            return face_crop
        
        # Upscale using INTER_CUBIC (better for faces than INTER_LINEAR)
        scale = self.config.UPSCALE_FACTOR
        new_w, new_h = int(w * scale), int(h * scale)
        upscaled = cv2.resize(face_crop, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
        
        # Apply mild sharpening to recover detail
        # Using unsharp mask: sharpened = original + (original - blurred) * amount
        blurred = cv2.GaussianBlur(upscaled, (0, 0), 2.0)
        sharpened = cv2.addWeighted(upscaled, 1.3, blurred, -0.3, 0)
        
        # Apply bilateral filter to reduce noise while preserving edges
        enhanced = cv2.bilateralFilter(sharpened, 5, 50, 50)
        
        return enhanced
    
    def detect(self, frame: np.ndarray) -> List[FaceDetection]:
        """
        Detect faces using scaled-down frame for speed, but crop from original 4K.
        This is the key optimization for long-distance detection on high-res streams.
        """
        detections = []
        
        h, w = frame.shape[:2]
        
        # Calculate scale factor for detection
        # Downscale large frames for faster MTCNN, but crop from original
        if w > self.config.MAX_DETECTION_WIDTH:
            scale = self.config.MAX_DETECTION_WIDTH / w
        else:
            scale = min(1.0, self.config.DETECTION_SCALE)
        
        # Create scaled frame for detection
        if scale < 1.0:
            scaled_w, scaled_h = int(w * scale), int(h * scale)
            scaled_frame = cv2.resize(frame, (scaled_w, scaled_h), interpolation=cv2.INTER_AREA)
            rgb_frame = cv2.cvtColor(scaled_frame, cv2.COLOR_BGR2RGB)
            inv_scale = 1.0 / scale
        else:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            inv_scale = 1.0
        
        # Run MTCNN on scaled frame
        boxes, probs = self.mtcnn.detect(rgb_frame)
        
        if boxes is None:
            return detections
        
        for box, prob in zip(boxes, probs):
            if prob < self.config.DETECTION_CONFIDENCE:
                continue
            
            # Scale detection box back to original 4K coordinates
            x1, y1, x2, y2 = map(int, [coord * inv_scale for coord in box])
            
            # Calculate face dimensions
            face_w = x2 - x1
            face_h = y2 - y1
            
            # Add margin (percentage-based for better coverage)
            margin_x = int(face_w * self.config.FACE_MARGIN_PERCENT)
            margin_y = int(face_h * self.config.FACE_MARGIN_PERCENT)
            
            # Expand bounding box with margin
            x1_expanded = max(0, x1 - margin_x)
            y1_expanded = max(0, y1 - margin_y)
            x2_expanded = min(w, x2 + margin_x)
            y2_expanded = min(h, y2 + margin_y)
            
            if x2_expanded <= x1_expanded or y2_expanded <= y1_expanded:
                continue
            
            # Extract face crop from full resolution frame
            face_crop = frame[y1_expanded:y2_expanded, x1_expanded:x2_expanded].copy()
            
            if face_crop.size == 0:
                continue
            
            # Enhance small faces (for distant detection)
            face_crop = self._enhance_small_face(face_crop)
            
            # Store original tight bbox for tracking, but use enhanced crop for embedding
            detections.append(FaceDetection(
                bbox=(x1, y1, x2, y2),  # Original tight box for display/tracking
                confidence=float(prob),
                face_crop=face_crop  # Enhanced crop for embedding
            ))
        
        return detections
    
    def filter_for_reid(self, detections: List[FaceDetection]) -> Tuple[List[FaceDetection], List[FaceDetection]]:
        """Filter detections based on size for ReID. Uses original bbox size, not enhanced crop."""
        suitable = []
        too_small = []
        
        for det in detections:
            # Use original detection size (before enhancement) for filtering
            if min(det.width, det.height) >= self.config.MIN_FACE_SIZE_FOR_REID:
                suitable.append(det)
            else:
                too_small.append(det)
        
        return suitable, too_small
        
        return suitable, too_small


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
    
    CANDIDATE_COLOR = (128, 128, 128)
    TOO_SMALL_COLOR = (100, 100, 100)
    
    def __init__(self, config: OptimizedConfig):
        self.config = config
    
    def get_color(self, person_id: int, is_confirmed: bool) -> Tuple[int, int, int]:
        if not is_confirmed:
            return self.CANDIDATE_COLOR
        return self.COLORS[person_id % len(self.COLORS)]
    
    def draw_detection(self, frame: np.ndarray, detection: FaceDetection, 
                       is_too_small: bool = False) -> np.ndarray:
        x1, y1, x2, y2 = detection.bbox
        
        if is_too_small:
            cv2.rectangle(frame, (x1, y1), (x2, y2), self.TOO_SMALL_COLOR, 1)
            return frame
        
        person_id = detection.person_id or 0
        color = self.get_color(person_id, detection.is_confirmed)
        
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, self.config.BBOX_THICKNESS)
        
        if detection.is_confirmed:
            label = f"ID: {person_id}"
        else:
            label = f"ID: ? ({abs(person_id)})"
        
        (label_w, label_h), _ = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, self.config.FONT_SCALE, 2
        )
        
        cv2.rectangle(frame, (x1, y1 - label_h - 10), (x1 + label_w + 5, y1), color, -1)
        cv2.putText(frame, label, (x1 + 2, y1 - 5),
                   cv2.FONT_HERSHEY_SIMPLEX, self.config.FONT_SCALE, (255, 255, 255), 2)
        
        return frame
    
    def draw_info(self, frame: np.ndarray, camera_id: int, fps: float, 
                  gallery_stats: Dict) -> np.ndarray:
        info_text = (f"Cam {camera_id} | FPS: {fps:.1f} | "
                    f"IDs: {gallery_stats['confirmed_faces']} | "
                    f"Embs: {gallery_stats.get('total_embeddings', 0)}")
        
        cv2.putText(frame, info_text, (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        return frame


# ============================================================================
# Camera Stream Handler
# ============================================================================

class CameraStream:
    """Handles RTSP stream capture with automatic reconnection."""
    
    def __init__(self, camera_id: int, url: str, config: OptimizedConfig):
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
                
                if not self._connect():
                    time.sleep(self.config.RECONNECT_DELAY)
                    continue
            
            try:
                ret, frame = self._cap.read()
                
                if not ret or frame is None:
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
    
    def stop(self):
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        if self._cap is not None:
            self._cap.release()
    
    def get_frame(self) -> Optional[Tuple[int, np.ndarray]]:
        try:
            return self._frame_queue.get_nowait()
        except Empty:
            return None
    
    @property
    def is_connected(self) -> bool:
        return self._connected


# ============================================================================
# Main Optimized System
# ============================================================================

class OptimizedMultiCameraFaceReIDSystem:
    """
    Optimized multi-camera face re-identification system with:
    - Enhanced preprocessing for lighting invariance
    - Multi-embedding profiles
    - Cross-camera handover optimization
    - CUDA acceleration support
    """
    
    def __init__(self, config: Optional[OptimizedConfig] = None):
        self.config = config or OptimizedConfig()
        
        logger.info("Initializing Optimized Multi-Camera Face ReID System...")
        
        self.detector = RobustFaceDetector(self.config)
        self.embedding_engine = OptimizedFaceEmbeddingEngine(self.config)
        self.gallery = OptimizedFaceGallery(self.config)
        self.visualizer = Visualizer(self.config)
        
        self.streams: List[CameraStream] = []
        for i, url in enumerate(self.config.CAMERA_URLS):
            stream = CameraStream(i, url, self.config)
            self.streams.append(stream)
        
        self._running = False
        
        logger.info(f"System initialized with {len(self.streams)} cameras")
        logger.info(f"Device: {self.config.DEVICE}, OpenCV CUDA: {OPENCV_CUDA_AVAILABLE}")
    
    def _process_frame(self, camera_id: int, frame: np.ndarray) -> np.ndarray:
        """Process a single frame with optimized pipeline."""
        # Update camera brightness statistics
        self.embedding_engine.update_camera_stats(camera_id, frame)
        
        # Detect all faces
        all_detections = self.detector.detect(frame)
        
        # Filter by size for ReID
        reid_detections, small_detections = self.detector.filter_for_reid(all_detections)
        
        # Track list for IOU updates
        current_tracks: List[Tuple[int, Tuple[int, int, int, int]]] = []
        
        if reid_detections:
            face_crops = [d.face_crop for d in reid_detections]
            embeddings = self.embedding_engine.extract_embeddings_batch(face_crops, camera_id)
            
            for detection, embedding in zip(reid_detections, embeddings):
                detection.embedding = embedding
                person_id, is_confirmed = self.gallery.match_and_update(
                    embedding, detection.bbox, camera_id
                )
                detection.person_id = person_id
                detection.is_confirmed = is_confirmed
                
                if is_confirmed:
                    current_tracks.append((person_id, detection.bbox))
        
        self.gallery.update_tracks(camera_id, current_tracks)
        
        # Visualize
        for detection in reid_detections:
            frame = self.visualizer.draw_detection(frame, detection)
        
        for detection in small_detections:
            frame = self.visualizer.draw_detection(frame, detection, is_too_small=True)
        
        return frame
    
    def run(self):
        logger.info("Starting Optimized Multi-Camera Face ReID System...")
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


# ============================================================================
# Entry Point
# ============================================================================

def main():
    print("=" * 70)
    print("Optimized Multi-Camera Face Re-identification System")
    print("=" * 70)
    print()
    print("Optimizations:")
    print("  ✓ Enhanced preprocessing (CLAHE + Gamma + LAB normalization)")
    print("  ✓ Multi-embedding profiles (cluster per person)")
    print("  ✓ Cosine similarity with max-pooling")
    print("  ✓ Temporal decay & spatial priority for cross-camera")
    print("  ✓ Adaptive per-camera thresholds")
    print(f"  ✓ CUDA preprocessing: {'Enabled' if OPENCV_CUDA_AVAILABLE else 'Disabled'}")
    print()
    
    config = OptimizedConfig()
    
    print("Configuration:")
    print(f"  Cameras: {len(config.CAMERA_URLS)}")
    print(f"  Device: {config.DEVICE}")
    print(f"  Base High Threshold: {config.BASE_HIGH_THRESHOLD}")
    print(f"  Base Low Threshold: {config.BASE_LOW_THRESHOLD}")
    print(f"  Max Embeddings per Profile: {config.MAX_EMBEDDINGS_PER_PROFILE}")
    print(f"  Temporal Boost (adjacent): {config.TEMPORAL_BOOST_ADJACENT}")
    print()
    
    system = OptimizedMultiCameraFaceReIDSystem(config)
    system.run()


if __name__ == "__main__":
    main()
