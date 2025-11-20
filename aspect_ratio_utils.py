"""
Aspect ratio and cropping utilities for non-square video generation
"""
import cv2
import numpy as np
from pathlib import Path
import logging
from typing import Tuple, Optional

try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False
    logging.warning("OpenCV not available for aspect ratio processing")


class AspectRatioProcessor:
    """Handle aspect ratio conversion and smart cropping."""
    
    # Common aspect ratios
    ASPECT_RATIOS = {
        '1:1': (1.0, 1.0, 'Square (Instagram)'),
        '16:9': (16.0, 9.0, 'Widescreen (YouTube)'),
        '9:16': (9.0, 16.0, 'Portrait (TikTok/Stories)'),
        '4:3': (4.0, 3.0, 'Classic (TV)'),
        '21:9': (21.0, 9.0, 'Ultrawide'),
        'custom': (None, None, 'Custom')
    }
    
    def __init__(self):
        """Initialize aspect ratio processor."""
        pass
    
    def calculate_target_size(self, original_width: int, original_height: int, 
                            aspect_ratio: str, custom_width: int = None, 
                            custom_height: int = None, max_dimension: int = 1024) -> Tuple[int, int]:
        """
        Calculate target size maintaining aspect ratio.
        
        Args:
            original_width: Original image width
            original_height: Original image height
            aspect_ratio: Aspect ratio string (e.g., '16:9')
            custom_width: Custom width (if aspect_ratio is 'custom')
            custom_height: Custom height (if aspect_ratio is 'custom')
            max_dimension: Maximum dimension to scale to
        
        Returns:
            Tuple of (target_width, target_height)
        """
        if aspect_ratio == 'custom' and custom_width and custom_height:
            # Use custom dimensions
            return (custom_width, custom_height)
        
        if aspect_ratio not in self.ASPECT_RATIOS:
            aspect_ratio = '1:1'  # Default to square
        
        target_aspect = self.ASPECT_RATIOS[aspect_ratio]
        target_w_ratio, target_h_ratio = target_aspect[0], target_aspect[1]
        target_aspect_value = target_w_ratio / target_h_ratio
        
        # Calculate dimensions maintaining aspect ratio
        if original_width / original_height > target_aspect_value:
            # Image is wider than target - fit to height
            target_height = min(max_dimension, original_height)
            target_width = int(target_height * target_aspect_value)
        else:
            # Image is taller than target - fit to width
            target_width = min(max_dimension, original_width)
            target_height = int(target_width / target_aspect_value)
        
        return (target_width, target_height)
    
    def smart_crop(self, image_path: str, target_width: int, target_height: int,
                  crop_mode: str = 'center', face_bbox: Optional[list] = None) -> np.ndarray:
        """
        Smart crop image to target dimensions.
        
        Args:
            image_path: Path to image
            target_width: Target width
            target_height: Target height
            crop_mode: 'center', 'face', 'top', 'bottom', 'left', 'right'
            face_bbox: Face bounding box [x1, y1, x2, y2] for face-aware cropping
        
        Returns:
            Cropped image
        """
        if not CV2_AVAILABLE:
            raise ImportError("OpenCV required for cropping")
        
        img = cv2.imread(str(image_path))
        if img is None:
            raise ValueError(f"Could not read image: {image_path}")
        
        original_height, original_width = img.shape[:2]
        
        # Calculate crop region
        if crop_mode == 'face' and face_bbox:
            # Face-aware cropping - center crop around face
            face_center_x = (face_bbox[0] + face_bbox[2]) // 2
            face_center_y = (face_bbox[1] + face_bbox[3]) // 2
            
            x1 = max(0, face_center_x - target_width // 2)
            y1 = max(0, face_center_y - target_height // 2)
            x2 = min(original_width, x1 + target_width)
            y2 = min(original_height, y1 + target_height)
            
            # Adjust if we hit boundaries
            if x2 - x1 < target_width:
                x1 = max(0, x2 - target_width)
            if y2 - y1 < target_height:
                y1 = max(0, y2 - target_height)
        
        elif crop_mode == 'center':
            # Center crop
            x1 = (original_width - target_width) // 2
            y1 = (original_height - target_height) // 2
            x2 = x1 + target_width
            y2 = y1 + target_height
        
        elif crop_mode == 'top':
            x1 = (original_width - target_width) // 2
            y1 = 0
            x2 = x1 + target_width
            y2 = target_height
        
        elif crop_mode == 'bottom':
            x1 = (original_width - target_width) // 2
            y1 = original_height - target_height
            x2 = x1 + target_width
            y2 = original_height
        
        elif crop_mode == 'left':
            x1 = 0
            y1 = (original_height - target_height) // 2
            x2 = target_width
            y2 = y1 + target_height
        
        elif crop_mode == 'right':
            x1 = original_width - target_width
            y1 = (original_height - target_height) // 2
            x2 = original_width
            y2 = y1 + target_height
        
        else:
            # Default to center
            x1 = (original_width - target_width) // 2
            y1 = (original_height - target_height) // 2
            x2 = x1 + target_width
            y2 = y1 + target_height
        
        # Ensure valid crop
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(original_width, x2)
        y2 = min(original_height, y2)
        
        cropped = img[y1:y2, x1:x2]
        
        # Resize if needed (in case crop was smaller than target)
        if cropped.shape[1] != target_width or cropped.shape[0] != target_height:
            cropped = cv2.resize(cropped, (target_width, target_height), interpolation=cv2.INTER_LANCZOS4)
        
        return cropped
    
    def preserve_background(self, original_image_path: str, animated_face: np.ndarray,
                          face_bbox: list, crop_region: list) -> np.ndarray:
        """
        Paste animated face back into original image preserving background.
        
        Args:
            original_image_path: Path to original image
            animated_face: Animated face region
            face_bbox: Original face bounding box [x1, y1, x2, y2]
            crop_region: Crop region used [x1, y1, x2, y2]
        
        Returns:
            Composite image with animated face on original background
        """
        if not CV2_AVAILABLE:
            raise ImportError("OpenCV required for background preservation")
        
        original = cv2.imread(str(original_image_path))
        if original is None:
            raise ValueError(f"Could not read original image: {original_image_path}")
        
        # Calculate position to paste animated face
        face_x1, face_y1, face_x2, face_y2 = face_bbox
        crop_x1, crop_y1, crop_x2, crop_y2 = crop_region
        
        # Calculate offset within crop
        offset_x = face_x1 - crop_x1
        offset_y = face_y1 - crop_y1
        
        # Resize animated face to match original face size
        face_width = face_x2 - face_x1
        face_height = face_y2 - face_y1
        
        # Extract face region from animated crop
        animated_face_region = animated_face[offset_y:offset_y+face_height, 
                                             offset_x:offset_x+face_width]
        
        if animated_face_region.size == 0:
            # Fallback: resize entire animated face
            animated_face_region = cv2.resize(animated_face, (face_width, face_height))
        
        # Create mask for blending
        mask = np.ones((face_height, face_width), dtype=np.uint8) * 255
        
        # Paste animated face onto original
        result = original.copy()
        
        # Use seamless cloning for better blending
        try:
            center = (face_x1 + face_width // 2, face_y1 + face_height // 2)
            result = cv2.seamlessClone(animated_face_region, result, mask, center, cv2.NORMAL_CLONE)
        except:
            # Fallback to simple paste
            result[face_y1:face_y1+face_height, face_x1:face_x1+face_width] = animated_face_region
        
        return result
    
    def resize_to_aspect(self, image_path: str, aspect_ratio: str,
                        custom_width: int = None, custom_height: int = None,
                        max_dimension: int = 1024, preserve_background: bool = True,
                        crop_mode: str = 'center', face_bbox: Optional[list] = None) -> np.ndarray:
        """
        Resize image to target aspect ratio with smart cropping.
        
        Args:
            image_path: Path to image
            aspect_ratio: Target aspect ratio
            custom_width: Custom width (if aspect_ratio is 'custom')
            custom_height: Custom height (if aspect_ratio is 'custom')
            max_dimension: Maximum dimension
            preserve_background: Whether to preserve background (adds padding)
            crop_mode: Crop mode for smart cropping
            face_bbox: Face bounding box for face-aware cropping
        
        Returns:
            Processed image
        """
        if not CV2_AVAILABLE:
            raise ImportError("OpenCV required")
        
        img = cv2.imread(str(image_path))
        if img is None:
            raise ValueError(f"Could not read image: {image_path}")
        
        original_height, original_width = img.shape[:2]
        
        # Calculate target size
        target_width, target_height = self.calculate_target_size(
            original_width, original_height, aspect_ratio,
            custom_width, custom_height, max_dimension
        )
        
        if preserve_background:
            # Add padding to maintain aspect ratio
            scale = min(target_width / original_width, target_height / original_height)
            new_width = int(original_width * scale)
            new_height = int(original_height * scale)
            
            # Resize image
            resized = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_LANCZOS4)
            
            # Create canvas with target aspect ratio
            canvas = np.zeros((target_height, target_width, 3), dtype=np.uint8)
            
            # Center the resized image
            x_offset = (target_width - new_width) // 2
            y_offset = (target_height - new_height) // 2
            
            canvas[y_offset:y_offset+new_height, x_offset:x_offset+new_width] = resized
            
            return canvas
        else:
            # Smart crop
            return self.smart_crop(image_path, target_width, target_height, crop_mode, face_bbox)

