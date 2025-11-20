"""
UI utilities for real-time preview, comparison view, and keyboard shortcuts
"""
import logging
import cv2
import numpy as np
from pathlib import Path
from typing import Optional, Tuple, List
import base64
import io
from PIL import Image

try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False
    logging.warning("OpenCV not available for UI utilities")


def create_comparison_image(image1_path: str, image2_path: str, 
                           output_path: Optional[str] = None,
                           labels: Tuple[str, str] = ("Before", "After"),
                           layout: str = "side") -> Optional[np.ndarray]:
    """
    Create side-by-side or stacked comparison image.
    
    Args:
        image1_path: Path to first image
        image2_path: Path to second image
        output_path: Optional path to save comparison
        labels: Labels for images (before_label, after_label)
        layout: "side" for horizontal, "stack" for vertical
    
    Returns:
        Comparison image as numpy array
    """
    if not CV2_AVAILABLE:
        return None
    
    try:
        img1 = cv2.imread(str(image1_path))
        img2 = cv2.imread(str(image2_path))
        
        if img1 is None or img2 is None:
            return None
        
        # Resize images to same height (for side layout) or width (for stack layout)
        if layout == "side":
            # Resize to same height
            height = min(img1.shape[0], img2.shape[0])
            width1 = int(img1.shape[1] * height / img1.shape[0])
            width2 = int(img2.shape[1] * height / img2.shape[0])
            img1 = cv2.resize(img1, (width1, height))
            img2 = cv2.resize(img2, (width2, height))
            
            # Combine horizontally
            comparison = np.hstack([img1, img2])
            total_width = comparison.shape[1]
            
            # Add labels
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 1.0
            thickness = 2
            color = (255, 255, 255)
            
            # Calculate text positions
            text_y = 30
            text1_x = total_width // 4
            text2_x = 3 * total_width // 4
            
            cv2.putText(comparison, labels[0], (text1_x, text_y), 
                       font, font_scale, color, thickness)
            cv2.putText(comparison, labels[1], (text2_x, text_y), 
                       font, font_scale, color, thickness)
        
        else:  # stack layout
            # Resize to same width
            width = min(img1.shape[1], img2.shape[1])
            height1 = int(img1.shape[0] * width / img1.shape[1])
            height2 = int(img2.shape[0] * width / img2.shape[1])
            img1 = cv2.resize(img1, (width, height1))
            img2 = cv2.resize(img2, (width, height2))
            
            # Combine vertically
            comparison = np.vstack([img1, img2])
            
            # Add labels
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 1.0
            thickness = 2
            color = (255, 255, 255)
            
            text_x = width // 2
            text1_y = 30
            text2_y = height1 + 30
            
            cv2.putText(comparison, labels[0], (text_x, text1_y), 
                       font, font_scale, color, thickness)
            cv2.putText(comparison, labels[1], (text_x, text2_y), 
                       font, font_scale, color, thickness)
        
        if output_path:
            cv2.imwrite(str(output_path), comparison)
        
        return comparison
    
    except Exception as e:
        logging.error(f"Failed to create comparison image: {e}")
        return None


def extract_video_frame(video_path: str, frame_number: int = 0) -> Optional[np.ndarray]:
    """
    Extract a frame from video for preview.
    
    Args:
        video_path: Path to video
        frame_number: Frame number (0-based)
    
    Returns:
        Frame as numpy array
    """
    if not CV2_AVAILABLE:
        return None
    
    try:
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            return None
        
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        ret, frame = cap.read()
        cap.release()
        
        if ret:
            return frame
        return None
    
    except Exception as e:
        logging.error(f"Failed to extract frame: {e}")
        return None


def create_preview_grid(images: List[str], output_path: Optional[str] = None,
                       grid_size: Optional[Tuple[int, int]] = None) -> Optional[np.ndarray]:
    """
    Create a grid preview from multiple images.
    
    Args:
        images: List of image paths
        output_path: Optional path to save grid
        grid_size: Optional (rows, cols) tuple. Auto-calculated if None
    
    Returns:
        Grid image as numpy array
    """
    if not CV2_AVAILABLE or not images:
        return None
    
    try:
        # Load images
        loaded_images = []
        for img_path in images:
            img = cv2.imread(str(img_path))
            if img is not None:
                loaded_images.append(img)
        
        if not loaded_images:
            return None
        
        # Calculate grid size
        if grid_size is None:
            cols = int(np.ceil(np.sqrt(len(loaded_images))))
            rows = int(np.ceil(len(loaded_images) / cols))
        else:
            rows, cols = grid_size
        
        # Resize all images to same size
        target_size = (256, 256)  # Preview size
        resized_images = []
        for img in loaded_images:
            resized = cv2.resize(img, target_size)
            resized_images.append(resized)
        
        # Create grid
        grid_rows = []
        for i in range(rows):
            row_images = []
            for j in range(cols):
                idx = i * cols + j
                if idx < len(resized_images):
                    row_images.append(resized_images[idx])
                else:
                    # Empty cell
                    row_images.append(np.zeros((target_size[1], target_size[0], 3), dtype=np.uint8))
            
            if row_images:
                row = np.hstack(row_images)
                grid_rows.append(row)
        
        if grid_rows:
            grid = np.vstack(grid_rows)
            
            if output_path:
                cv2.imwrite(str(output_path), grid)
            
            return grid
        
        return None
    
    except Exception as e:
        logging.error(f"Failed to create preview grid: {e}")
        return None


def image_to_base64(image_array: np.ndarray) -> str:
    """Convert numpy image array to base64 string."""
    try:
        from PIL import Image
        import io
        import base64
        
        # Convert BGR to RGB if needed
        if len(image_array.shape) == 3 and image_array.shape[2] == 3:
            image_rgb = cv2.cvtColor(image_array, cv2.COLOR_BGR2RGB)
        else:
            image_rgb = image_array
        
        pil_image = Image.fromarray(image_rgb)
        buffer = io.BytesIO()
        pil_image.save(buffer, format='PNG')
        img_bytes = buffer.getvalue()
        img_base64 = base64.b64encode(img_bytes).decode('utf-8')
        return f"data:image/png;base64,{img_base64}"
    
    except Exception as e:
        logging.error(f"Failed to convert image to base64: {e}")
        return ""

