"""
Multi-face detection and processing utilities
Uses InsightFace for face detection (already included in LivePortrait)
"""
import cv2
import numpy as np
from pathlib import Path
import logging
from typing import List, Tuple, Optional
import json

try:
    import onnxruntime as ort
    ONNXRUNTIME_AVAILABLE = True
except ImportError:
    ONNXRUNTIME_AVAILABLE = False
    logging.warning("ONNX Runtime not available. Face detection will be limited.")

try:
    from insightface import app as insightface_app
    from insightface.utils import face_align
    INSIGHTFACE_AVAILABLE = True
except ImportError:
    INSIGHTFACE_AVAILABLE = False
    logging.warning("InsightFace not available. Using basic face detection.")


class FaceDetector:
    """Face detection and processing using InsightFace or OpenCV."""
    
    def __init__(self, model_path=None):
        """Initialize face detector."""
        self.model_path = model_path
        self.face_app = None
        self.detector = None
        
        # Try to initialize InsightFace
        if INSIGHTFACE_AVAILABLE and model_path:
            try:
                self.face_app = insightface_app.FaceAnalysis(
                    name='buffalo_l',
                    root=str(Path(model_path).parent.parent) if model_path else None,
                    providers=['CUDAExecutionProvider', 'CPUExecutionProvider']
                )
                self.face_app.prepare(ctx_id=0, det_size=(640, 640))
                logging.info("InsightFace initialized successfully")
            except Exception as e:
                logging.warning(f"Failed to initialize InsightFace: {e}")
                self._init_opencv_detector()
        else:
            self._init_opencv_detector()
    
    def _init_opencv_detector(self):
        """Initialize OpenCV face detector as fallback."""
        try:
            # Try to load OpenCV DNN face detector
            dnn_path = Path(__file__).parent / 'models' / 'opencv_face_detector.pbtxt'
            weights_path = Path(__file__).parent / 'models' / 'opencv_face_detector_uint8.pb'
            
            if dnn_path.exists() and weights_path.exists():
                self.detector = cv2.dnn.readNetFromTensorflow(str(weights_path), str(dnn_path))
                logging.info("OpenCV DNN face detector initialized")
            else:
                # Use Haar Cascade as last resort
                cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
                self.detector = cv2.CascadeClassifier(cascade_path)
                logging.info("OpenCV Haar Cascade face detector initialized")
        except Exception as e:
            logging.error(f"Failed to initialize OpenCV detector: {e}")
            self.detector = None
    
    def detect_faces(self, image_path: str) -> List[dict]:
        """
        Detect all faces in an image.
        
        Returns:
            List of face dictionaries with keys:
            - bbox: [x1, y1, x2, y2] bounding box
            - landmarks: 5 facial landmarks
            - confidence: detection confidence
            - index: face index (0-based)
        """
        if not Path(image_path).exists():
            raise FileNotFoundError(f"Image not found: {image_path}")
        
        # Read image
        img = cv2.imread(str(image_path))
        if img is None:
            raise ValueError(f"Could not read image: {image_path}")
        
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        height, width = img.shape[:2]
        
        faces = []
        
        # Use InsightFace if available
        if self.face_app is not None:
            try:
                detected_faces = self.face_app.get(img_rgb)
                
                for idx, face in enumerate(detected_faces):
                    bbox = face.bbox.astype(int)
                    landmarks = face.kps if hasattr(face, 'kps') else None
                    confidence = face.det_score if hasattr(face, 'det_score') else 1.0
                    
                    faces.append({
                        'bbox': [int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])],
                        'landmarks': landmarks.tolist() if landmarks is not None else None,
                        'confidence': float(confidence),
                        'index': idx,
                        'area': (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
                    })
                
                logging.info(f"Detected {len(faces)} faces using InsightFace")
                return faces
            
            except Exception as e:
                logging.warning(f"InsightFace detection failed: {e}, falling back to OpenCV")
        
        # Fallback to OpenCV
        if self.detector is not None:
            try:
                if isinstance(self.detector, cv2.dnn.Net):
                    # DNN detector
                    blob = cv2.dnn.blobFromImage(img, 1.0, (300, 300), [104, 117, 123])
                    self.detector.setInput(blob)
                    detections = self.detector.forward()
                    
                    for idx in range(detections.shape[2]):
                        confidence = detections[0, 0, idx, 2]
                        if confidence > 0.5:
                            x1 = int(detections[0, 0, idx, 3] * width)
                            y1 = int(detections[0, 0, idx, 4] * height)
                            x2 = int(detections[0, 0, idx, 5] * width)
                            y2 = int(detections[0, 0, idx, 6] * height)
                            
                            faces.append({
                                'bbox': [x1, y1, x2, y2],
                                'landmarks': None,
                                'confidence': float(confidence),
                                'index': len(faces),
                                'area': (x2 - x1) * (y2 - y1)
                            })
                
                elif isinstance(self.detector, cv2.CascadeClassifier):
                    # Haar Cascade detector
                    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                    detected = self.detector.detectMultiScale(gray, 1.1, 4)
                    
                    for idx, (x, y, w, h) in enumerate(detected):
                        faces.append({
                            'bbox': [int(x), int(y), int(x + w), int(y + h)],
                            'landmarks': None,
                            'confidence': 0.8,  # Haar doesn't provide confidence
                            'index': idx,
                            'area': w * h
                        })
                
                logging.info(f"Detected {len(faces)} faces using OpenCV")
                return faces
            
            except Exception as e:
                logging.error(f"OpenCV detection failed: {e}")
        
        # If no detector available, return empty list
        logging.warning("No face detector available")
        return []
    
    def draw_face_boxes(self, image_path: str, faces: List[dict], 
                       selected_indices: List[int] = None) -> np.ndarray:
        """
        Draw bounding boxes on image for detected faces.
        
        Args:
            image_path: Path to image
            faces: List of face dictionaries
            selected_indices: List of face indices to highlight
            
        Returns:
            Image with drawn boxes
        """
        img = cv2.imread(str(image_path))
        if img is None:
            return None
        
        selected_indices = selected_indices or []
        
        for face in faces:
            idx = face['index']
            bbox = face['bbox']
            x1, y1, x2, y2 = bbox
            
            # Color: green for selected, blue for others
            color = (0, 255, 0) if idx in selected_indices else (255, 0, 0)
            thickness = 3 if idx in selected_indices else 2
            
            # Draw bounding box
            cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)
            
            # Draw face index
            label = f"Face {idx + 1}"
            if idx in selected_indices:
                label += " (Selected)"
            
            # Background for text
            (text_width, text_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(img, (x1, y1 - text_height - 10), 
                         (x1 + text_width, y1), color, -1)
            
            # Text
            cv2.putText(img, label, (x1, y1 - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            # Draw landmarks if available
            if face['landmarks'] is not None:
                for landmark in face['landmarks']:
                    x, y = int(landmark[0]), int(landmark[1])
                    cv2.circle(img, (x, y), 3, (0, 255, 255), -1)
        
        return img
    
    def crop_face(self, image_path: str, face: dict, padding: float = 0.2) -> np.ndarray:
        """
        Crop face from image with padding.
        
        Args:
            image_path: Path to image
            face: Face dictionary with bbox
            padding: Padding ratio (0.2 = 20% padding)
        
        Returns:
            Cropped face image
        """
        img = cv2.imread(str(image_path))
        if img is None:
            return None
        
        height, width = img.shape[:2]
        x1, y1, x2, y2 = face['bbox']
        
        # Add padding
        face_width = x2 - x1
        face_height = y2 - y1
        pad_x = int(face_width * padding)
        pad_y = int(face_height * padding)
        
        x1 = max(0, x1 - pad_x)
        y1 = max(0, y1 - pad_y)
        x2 = min(width, x2 + pad_x)
        y2 = min(height, y2 + pad_y)
        
        cropped = img[y1:y2, x1:x2]
        return cropped
    
    def get_largest_face(self, faces: List[dict]) -> Optional[dict]:
        """Get the largest face by area."""
        if not faces:
            return None
        return max(faces, key=lambda f: f['area'])
    
    def get_face_at_position(self, faces: List[dict], x: int, y: int) -> Optional[dict]:
        """Get face at specific position (for click selection)."""
        for face in faces:
            x1, y1, x2, y2 = face['bbox']
            if x1 <= x <= x2 and y1 <= y <= y2:
                return face
        return None


def detect_faces_in_image(image_path: str, model_path: str = None) -> List[dict]:
    """Convenience function to detect faces in an image."""
    detector = FaceDetector(model_path)
    return detector.detect_faces(image_path)


def save_face_detection_results(image_path: str, faces: List[dict], output_path: str):
    """Save face detection results to JSON file."""
    results = {
        'image_path': str(image_path),
        'num_faces': len(faces),
        'faces': faces
    }
    
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    logging.info(f"Saved face detection results to {output_path}")

