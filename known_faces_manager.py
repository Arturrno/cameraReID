"""
Known Faces Manager - Reference Photo Database
===============================================
Provides consistent face recognition by matching against
a database of known reference photos.

This is how professional face recognition systems work:
1. Enroll people with high-quality reference photos
2. Compute embeddings for all reference photos
3. Match live detections against known references FIRST
4. Only create new IDs for truly unknown people

Directory Structure:
    known_faces/
        person_001/
            photo1.jpg
            photo2.jpg
        person_002/
            photo1.jpg
        ...
    known_faces.json  # Metadata: names, notes, etc.

Author: Senior Computer Vision Engineer
Date: 2024
"""

import os
import json
import logging
import numpy as np
import cv2
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field, asdict
from threading import Lock
import time
import shutil

logger = logging.getLogger(__name__)


# ============================================================================
# Data Classes
# ============================================================================

@dataclass
class KnownPerson:
    """Represents a known person in the reference database."""
    person_id: str  # Unique identifier (e.g., "person_001")
    first_name: str = ""
    last_name: str = ""
    notes: str = ""
    photo_paths: List[str] = field(default_factory=list)
    embeddings: List[np.ndarray] = field(default_factory=list)
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    
    @property
    def full_name(self) -> str:
        """Get full name or ID if no name set."""
        if self.first_name or self.last_name:
            return f"{self.first_name} {self.last_name}".strip()
        return self.person_id
    
    @property
    def num_photos(self) -> int:
        return len(self.photo_paths)
    
    @property
    def num_embeddings(self) -> int:
        return len(self.embeddings)
    
    def get_centroid_embedding(self) -> Optional[np.ndarray]:
        """Get average embedding (centroid) for this person."""
        if not self.embeddings:
            return None
        centroid = np.mean(self.embeddings, axis=0)
        # L2 normalize
        norm = np.linalg.norm(centroid)
        if norm > 0:
            centroid = centroid / norm
        return centroid
    
    def compute_similarity(self, query_embedding: np.ndarray) -> float:
        """
        Compute similarity against all reference embeddings.
        Returns maximum similarity (best match among all photos).
        """
        if not self.embeddings:
            return 0.0
        
        max_sim = 0.0
        for ref_emb in self.embeddings:
            # Cosine similarity (embeddings should be L2-normalized)
            sim = float(np.dot(query_embedding, ref_emb))
            max_sim = max(max_sim, sim)
        
        return max_sim
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization (without embeddings)."""
        return {
            'person_id': self.person_id,
            'first_name': self.first_name,
            'last_name': self.last_name,
            'notes': self.notes,
            'photo_paths': self.photo_paths,
            'created_at': self.created_at,
            'updated_at': self.updated_at
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'KnownPerson':
        """Create from dictionary (loaded from JSON)."""
        return cls(
            person_id=data['person_id'],
            first_name=data.get('first_name', ''),
            last_name=data.get('last_name', ''),
            notes=data.get('notes', ''),
            photo_paths=data.get('photo_paths', []),
            created_at=data.get('created_at', time.time()),
            updated_at=data.get('updated_at', time.time())
        )


@dataclass
class MatchResult:
    """Result of matching against known faces database."""
    matched: bool
    person_id: Optional[str] = None
    person: Optional[KnownPerson] = None
    similarity: float = 0.0
    is_known: bool = False  # True if matched against reference database


# ============================================================================
# Known Faces Manager
# ============================================================================

class KnownFacesManager:
    """
    Manages a database of known faces with reference photos.
    
    Key features:
    - Load reference photos from directory structure
    - Compute and cache embeddings for all references
    - Fast matching against known faces
    - Add new people from GUI or live detection
    - Persist metadata to JSON
    """
    
    DEFAULT_DIR = "known_faces"
    METADATA_FILE = "known_faces.json"
    
    # Matching thresholds - lower = more lenient matching to known faces
    # This is LOWER than the dynamic gallery threshold so known faces are preferred
    KNOWN_MATCH_THRESHOLD = 0.55  # Match known person more easily than creating new ID
    KNOWN_STRONG_MATCH = 0.70     # Very confident match
    
    def __init__(self, 
                 base_dir: str = None,
                 embedding_engine = None):
        """
        Initialize the Known Faces Manager.
        
        Args:
            base_dir: Base directory for known_faces folder. Defaults to current dir.
            embedding_engine: Face embedding engine for computing embeddings.
        """
        self.base_dir = Path(base_dir) if base_dir else Path.cwd()
        self.known_faces_dir = self.base_dir / self.DEFAULT_DIR
        self.metadata_path = self.base_dir / self.METADATA_FILE
        
        self.embedding_engine = embedding_engine
        
        # Known persons database
        self._persons: Dict[str, KnownPerson] = {}
        self._lock = Lock()
        
        # ID counter for new persons
        self._next_id = 1
        
        # Statistics
        self._match_count = 0
        self._known_match_count = 0
        
        # Initialize
        self._ensure_directory_exists()
        self._load_metadata()
    
    def _ensure_directory_exists(self):
        """Create known_faces directory if it doesn't exist."""
        self.known_faces_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Known faces directory: {self.known_faces_dir}")
    
    def _load_metadata(self):
        """Load metadata from JSON file."""
        if not self.metadata_path.exists():
            logger.info("No known_faces.json found, starting fresh")
            self._save_metadata()
            return
        
        try:
            with open(self.metadata_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Load persons
            for person_data in data.get('persons', []):
                person = KnownPerson.from_dict(person_data)
                self._persons[person.person_id] = person
            
            # Update next ID counter
            if self._persons:
                max_id = max(
                    int(pid.split('_')[-1]) 
                    for pid in self._persons.keys() 
                    if pid.startswith('person_')
                )
                self._next_id = max_id + 1
            
            logger.info(f"Loaded {len(self._persons)} known persons from metadata")
            
        except Exception as e:
            logger.error(f"Error loading metadata: {e}")
    
    def _save_metadata(self):
        """Save metadata to JSON file."""
        try:
            data = {
                'version': '1.0',
                'updated_at': time.time(),
                'persons': [p.to_dict() for p in self._persons.values()]
            }
            
            with open(self.metadata_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            
            logger.debug(f"Saved metadata for {len(self._persons)} persons")
            
        except Exception as e:
            logger.error(f"Error saving metadata: {e}")
    
    def _generate_person_id(self) -> str:
        """Generate a new unique person ID."""
        person_id = f"person_{self._next_id:03d}"
        self._next_id += 1
        return person_id
    
    def set_embedding_engine(self, engine):
        """Set the embedding engine (can be set after init)."""
        self.embedding_engine = engine
    
    def load_all_embeddings(self) -> int:
        """
        Load and compute embeddings for all known persons.
        Call this after setting the embedding engine.
        
        Returns:
            Number of embeddings computed
        """
        if self.embedding_engine is None:
            logger.error("Embedding engine not set, cannot load embeddings")
            return 0
        
        total_embeddings = 0
        
        with self._lock:
            for person_id, person in self._persons.items():
                person.embeddings.clear()
                
                # First try to load from person's directory
                person_dir = self.known_faces_dir / person_id
                if person_dir.exists():
                    # Scan directory for photos
                    photo_files = list(person_dir.glob("*.jpg")) + \
                                  list(person_dir.glob("*.jpeg")) + \
                                  list(person_dir.glob("*.png")) + \
                                  list(person_dir.glob("*.bmp"))
                    
                    # Update photo paths
                    person.photo_paths = [str(p.relative_to(self.base_dir)) for p in photo_files]
                
                # Compute embeddings for each photo
                for photo_path in person.photo_paths:
                    full_path = self.base_dir / photo_path
                    if not full_path.exists():
                        logger.warning(f"Photo not found: {full_path}")
                        continue
                    
                    embedding = self._compute_embedding_from_photo(str(full_path))
                    if embedding is not None:
                        person.embeddings.append(embedding)
                        total_embeddings += 1
                
                if person.embeddings:
                    logger.info(f"Loaded {len(person.embeddings)} embeddings for {person.full_name}")
                else:
                    logger.warning(f"No valid embeddings for {person.full_name}")
        
        # Save updated metadata
        self._save_metadata()
        
        logger.info(f"Total embeddings loaded: {total_embeddings} for {len(self._persons)} persons")
        return total_embeddings
    
    def _compute_embedding_from_photo(self, photo_path: str) -> Optional[np.ndarray]:
        """
        Compute face embedding from a photo file.
        Uses face detection first to find and crop the face.
        """
        if self.embedding_engine is None:
            return None
        
        try:
            # Load image
            image = cv2.imread(photo_path)
            if image is None:
                logger.warning(f"Could not load image: {photo_path}")
                return None
            
            # For reference photos, we need to detect the face first
            # Import here to avoid circular import
            from multi_camera_face_reid_optimized import RobustFaceDetector, OptimizedConfig
            
            # Use a temporary detector if not available
            # Note: In production, this should be passed in or cached
            config = OptimizedConfig()
            detector = RobustFaceDetector(config)
            
            # Detect faces
            detections = detector.detect(image)
            
            if not detections:
                # If no face detected, try to use the whole image (assuming it's a face crop)
                logger.warning(f"No face detected in {photo_path}, using full image")
                face_crop = cv2.resize(image, (160, 160))
            else:
                # Use the largest/most confident face
                best_detection = max(detections, key=lambda d: d.confidence * d.area)
                face_crop = best_detection.face_crop
            
            # Compute embedding
            embedding = self.embedding_engine.extract_embedding(face_crop, camera_id=-1)
            
            return embedding
            
        except Exception as e:
            logger.error(f"Error computing embedding from {photo_path}: {e}")
            return None
    
    def match(self, query_embedding: np.ndarray) -> MatchResult:
        """
        Match a query embedding against all known faces.
        
        Args:
            query_embedding: L2-normalized face embedding
            
        Returns:
            MatchResult with match details
        """
        self._match_count += 1
        
        with self._lock:
            best_match: Optional[KnownPerson] = None
            best_similarity = 0.0
            
            for person in self._persons.values():
                if not person.embeddings:
                    continue
                
                similarity = person.compute_similarity(query_embedding)
                
                if similarity > best_similarity:
                    best_similarity = similarity
                    best_match = person
            
            # Check if above threshold
            if best_match is not None and best_similarity >= self.KNOWN_MATCH_THRESHOLD:
                self._known_match_count += 1
                return MatchResult(
                    matched=True,
                    person_id=best_match.person_id,
                    person=best_match,
                    similarity=best_similarity,
                    is_known=True
                )
            
            return MatchResult(
                matched=False,
                similarity=best_similarity
            )
    
    def add_person(self, 
                   first_name: str = "",
                   last_name: str = "",
                   notes: str = "",
                   photo_paths: List[str] = None,
                   face_crops: List[np.ndarray] = None) -> Optional[KnownPerson]:
        """
        Add a new person to the known faces database.
        
        Args:
            first_name: Person's first name
            last_name: Person's last name
            notes: Additional notes
            photo_paths: List of photo file paths to copy
            face_crops: List of face crop images to save
            
        Returns:
            The created KnownPerson, or None on failure
        """
        with self._lock:
            person_id = self._generate_person_id()
            
            # Create person directory
            person_dir = self.known_faces_dir / person_id
            person_dir.mkdir(parents=True, exist_ok=True)
            
            saved_paths = []
            embeddings = []
            
            # Copy provided photos
            if photo_paths:
                for i, src_path in enumerate(photo_paths):
                    src = Path(src_path)
                    if src.exists():
                        dst = person_dir / f"photo_{i+1:02d}{src.suffix}"
                        shutil.copy2(src, dst)
                        saved_paths.append(str(dst.relative_to(self.base_dir)))
            
            # Save face crops as photos
            if face_crops:
                for i, crop in enumerate(face_crops):
                    if crop is not None and crop.size > 0:
                        filename = f"capture_{i+1:02d}.jpg"
                        filepath = person_dir / filename
                        cv2.imwrite(str(filepath), crop)
                        saved_paths.append(str(filepath.relative_to(self.base_dir)))
                        
                        # Compute embedding
                        if self.embedding_engine is not None:
                            emb = self.embedding_engine.extract_embedding(crop, camera_id=-1)
                            if emb is not None:
                                embeddings.append(emb)
            
            # Create person object
            person = KnownPerson(
                person_id=person_id,
                first_name=first_name,
                last_name=last_name,
                notes=notes,
                photo_paths=saved_paths,
                embeddings=embeddings
            )
            
            self._persons[person_id] = person
            self._save_metadata()
            
            logger.info(f"Added new known person: {person.full_name} ({person_id}) with {len(saved_paths)} photos")
            
            return person
    
    def add_photo_to_person(self, 
                            person_id: str, 
                            photo_path: str = None,
                            face_crop: np.ndarray = None) -> bool:
        """
        Add a photo to an existing person.
        
        Args:
            person_id: ID of the person
            photo_path: Path to photo file to copy
            face_crop: Face crop image to save
            
        Returns:
            True on success
        """
        with self._lock:
            if person_id not in self._persons:
                logger.error(f"Person not found: {person_id}")
                return False
            
            person = self._persons[person_id]
            person_dir = self.known_faces_dir / person_id
            person_dir.mkdir(parents=True, exist_ok=True)
            
            # Determine next photo number
            existing_photos = len(list(person_dir.glob("*")))
            
            if photo_path:
                src = Path(photo_path)
                if src.exists():
                    dst = person_dir / f"photo_{existing_photos+1:02d}{src.suffix}"
                    shutil.copy2(src, dst)
                    person.photo_paths.append(str(dst.relative_to(self.base_dir)))
                    
                    # Compute embedding
                    emb = self._compute_embedding_from_photo(str(dst))
                    if emb is not None:
                        person.embeddings.append(emb)
            
            if face_crop is not None and face_crop.size > 0:
                filepath = person_dir / f"capture_{existing_photos+1:02d}.jpg"
                cv2.imwrite(str(filepath), face_crop)
                person.photo_paths.append(str(filepath.relative_to(self.base_dir)))
                
                if self.embedding_engine is not None:
                    emb = self.embedding_engine.extract_embedding(face_crop, camera_id=-1)
                    if emb is not None:
                        person.embeddings.append(emb)
            
            person.updated_at = time.time()
            self._save_metadata()
            
            logger.info(f"Added photo to {person.full_name}")
            return True
    
    def update_person(self, 
                      person_id: str,
                      first_name: str = None,
                      last_name: str = None,
                      notes: str = None) -> bool:
        """Update person metadata."""
        with self._lock:
            if person_id not in self._persons:
                return False
            
            person = self._persons[person_id]
            
            if first_name is not None:
                person.first_name = first_name
            if last_name is not None:
                person.last_name = last_name
            if notes is not None:
                person.notes = notes
            
            person.updated_at = time.time()
            self._save_metadata()
            
            return True
    
    def remove_person(self, person_id: str) -> bool:
        """Remove a person and their photos."""
        with self._lock:
            if person_id not in self._persons:
                return False
            
            # Remove directory
            person_dir = self.known_faces_dir / person_id
            if person_dir.exists():
                shutil.rmtree(person_dir)
            
            del self._persons[person_id]
            self._save_metadata()
            
            logger.info(f"Removed person: {person_id}")
            return True
    
    def get_person(self, person_id: str) -> Optional[KnownPerson]:
        """Get a known person by ID."""
        with self._lock:
            return self._persons.get(person_id)
    
    def get_all_persons(self) -> List[KnownPerson]:
        """Get all known persons."""
        with self._lock:
            return list(self._persons.values())
    
    def get_stats(self) -> Dict:
        """Get statistics about the known faces database."""
        with self._lock:
            total_photos = sum(p.num_photos for p in self._persons.values())
            total_embeddings = sum(p.num_embeddings for p in self._persons.values())
            
            return {
                'num_persons': len(self._persons),
                'total_photos': total_photos,
                'total_embeddings': total_embeddings,
                'match_attempts': self._match_count,
                'known_matches': self._known_match_count,
                'match_rate': self._known_match_count / max(1, self._match_count)
            }
    
    def scan_directory_for_new_persons(self) -> int:
        """
        Scan known_faces directory for new person folders not in metadata.
        Useful for manually adding photos.
        
        Returns:
            Number of new persons discovered
        """
        new_count = 0
        
        with self._lock:
            for person_dir in self.known_faces_dir.iterdir():
                if not person_dir.is_dir():
                    continue
                
                person_id = person_dir.name
                
                if person_id in self._persons:
                    continue
                
                # New person found
                photo_files = list(person_dir.glob("*.jpg")) + \
                              list(person_dir.glob("*.jpeg")) + \
                              list(person_dir.glob("*.png")) + \
                              list(person_dir.glob("*.bmp"))
                
                if not photo_files:
                    continue
                
                person = KnownPerson(
                    person_id=person_id,
                    photo_paths=[str(p.relative_to(self.base_dir)) for p in photo_files]
                )
                
                self._persons[person_id] = person
                new_count += 1
                
                logger.info(f"Discovered new person: {person_id} with {len(photo_files)} photos")
            
            if new_count > 0:
                self._save_metadata()
        
        return new_count


# ============================================================================
# Integration Helper
# ============================================================================

def create_known_faces_manager(base_dir: str = None, 
                               embedding_engine = None) -> KnownFacesManager:
    """
    Factory function to create and initialize a KnownFacesManager.
    
    Usage:
        from known_faces_manager import create_known_faces_manager
        
        manager = create_known_faces_manager(
            base_dir="path/to/project",
            embedding_engine=your_embedding_engine
        )
        manager.load_all_embeddings()
    """
    manager = KnownFacesManager(base_dir=base_dir, embedding_engine=embedding_engine)
    
    # Scan for manually added persons
    new_persons = manager.scan_directory_for_new_persons()
    if new_persons > 0:
        logger.info(f"Discovered {new_persons} new persons from directory scan")
    
    return manager
