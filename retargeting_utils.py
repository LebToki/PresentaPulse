"""
Advanced retargeting utilities for expression control, blink frequency, head movement, and gaze direction
"""
import logging
import numpy as np
from typing import Optional, Dict, List
from dataclasses import dataclass

@dataclass
class RetargetingParams:
    """Parameters for advanced retargeting."""
    eye_open_ratio: float = 0.0
    lip_open_ratio: float = 0.0
    expression_intensity: float = 1.0  # 0.0 to 2.0
    blink_frequency: float = 0.0  # 0.0 = no blinks, 1.0 = normal, 2.0 = frequent
    head_movement_intensity: float = 1.0  # 0.0 = no movement, 1.0 = normal, 2.0 = exaggerated
    gaze_direction_x: float = 0.0  # -1.0 (left) to 1.0 (right)
    gaze_direction_y: float = 0.0  # -1.0 (down) to 1.0 (up)
    preset_name: Optional[str] = None


class ExpressionPresets:
    """Predefined expression presets."""
    
    PRESETS = {
        'natural': {
            'eye_open_ratio': 0.0,
            'lip_open_ratio': 0.0,
            'expression_intensity': 1.0,
            'blink_frequency': 1.0,
            'head_movement_intensity': 1.0,
            'gaze_direction_x': 0.0,
            'gaze_direction_y': 0.0,
            'description': 'Natural, subtle expressions'
        },
        'excited': {
            'eye_open_ratio': 0.3,
            'lip_open_ratio': 0.4,
            'expression_intensity': 1.5,
            'blink_frequency': 1.2,
            'head_movement_intensity': 1.3,
            'gaze_direction_x': 0.0,
            'gaze_direction_y': 0.2,
            'description': 'Energetic and excited'
        },
        'calm': {
            'eye_open_ratio': 0.0,
            'lip_open_ratio': 0.0,
            'expression_intensity': 0.7,
            'blink_frequency': 0.8,
            'head_movement_intensity': 0.6,
            'gaze_direction_x': 0.0,
            'gaze_direction_y': -0.1,
            'description': 'Calm and relaxed'
        },
        'surprised': {
            'eye_open_ratio': 0.6,
            'lip_open_ratio': 0.5,
            'expression_intensity': 1.8,
            'blink_frequency': 0.5,
            'head_movement_intensity': 1.2,
            'gaze_direction_x': 0.0,
            'gaze_direction_y': 0.3,
            'description': 'Surprised expression'
        },
        'focused': {
            'eye_open_ratio': 0.2,
            'lip_open_ratio': 0.0,
            'expression_intensity': 1.1,
            'blink_frequency': 0.6,
            'head_movement_intensity': 0.5,
            'gaze_direction_x': 0.0,
            'gaze_direction_y': 0.1,
            'description': 'Focused and attentive'
        },
        'playful': {
            'eye_open_ratio': 0.4,
            'lip_open_ratio': 0.3,
            'expression_intensity': 1.3,
            'blink_frequency': 1.5,
            'head_movement_intensity': 1.4,
            'gaze_direction_x': 0.2,
            'gaze_direction_y': 0.1,
            'description': 'Playful and animated'
        }
    }
    
    @classmethod
    def get_preset(cls, name: str) -> Optional[Dict]:
        """Get preset by name."""
        return cls.PRESETS.get(name)
    
    @classmethod
    def list_presets(cls) -> List[str]:
        """List all available presets."""
        return list(cls.PRESETS.keys())
    
    @classmethod
    def get_preset_description(cls, name: str) -> str:
        """Get description for a preset."""
        preset = cls.get_preset(name)
        if preset:
            return preset.get('description', '')
        return ''


def apply_expression_intensity(base_value: float, intensity: float) -> float:
    """
    Apply expression intensity multiplier.
    
    Args:
        base_value: Base value (0.0 to 1.0)
        intensity: Intensity multiplier (0.0 to 2.0)
    
    Returns:
        Adjusted value
    """
    # Clamp intensity
    intensity = max(0.0, min(2.0, intensity))
    
    # Apply intensity: 1.0 = no change, <1.0 = reduce, >1.0 = amplify
    adjusted = base_value * intensity
    
    # Clamp result
    return max(0.0, min(0.8, adjusted))


def calculate_blink_pattern(frame_count: int, fps: float, blink_frequency: float) -> List[int]:
    """
    Calculate blink frame indices based on frequency.
    
    Args:
        frame_count: Total number of frames
        fps: Frames per second
        blink_frequency: Blink frequency multiplier (0.0 = no blinks, 1.0 = normal ~20 blinks/min, 2.0 = frequent)
    
    Returns:
        List of frame indices where blinks should occur
    """
    if blink_frequency <= 0:
        return []
    
    # Normal blink rate: ~20 blinks per minute = 1 blink per 3 seconds
    # Blink duration: ~100-400ms = 3-12 frames at 30fps
    normal_blink_interval = 3.0 * fps  # frames between blinks
    blink_duration = 6  # frames (200ms at 30fps)
    
    # Adjust interval based on frequency
    blink_interval = normal_blink_interval / blink_frequency
    
    blink_frames = []
    current_frame = int(blink_interval)
    
    while current_frame < frame_count:
        # Add blink frames (open -> close -> open)
        for i in range(blink_duration):
            if current_frame + i < frame_count:
                blink_frames.append(current_frame + i)
        current_frame += int(blink_interval)
    
    return blink_frames


def apply_head_movement_modification(base_motion: np.ndarray, intensity: float) -> np.ndarray:
    """
    Modify head movement intensity.
    
    Args:
        base_motion: Base motion data (numpy array)
        intensity: Movement intensity (0.0 to 2.0)
    
    Returns:
        Modified motion data
    """
    if intensity <= 0:
        # No movement - return neutral
        return np.zeros_like(base_motion)
    
    # Scale motion by intensity
    modified = base_motion * intensity
    
    # Clamp to reasonable range
    return np.clip(modified, -1.0, 1.0)


def apply_gaze_direction_modification(base_gaze: np.ndarray, direction_x: float, direction_y: float) -> np.ndarray:
    """
    Modify gaze direction.
    
    Args:
        base_gaze: Base gaze data (numpy array)
        direction_x: X direction (-1.0 = left, 1.0 = right)
        direction_y: Y direction (-1.0 = down, 1.0 = up)
    
    Returns:
        Modified gaze data
    """
    # Clamp directions
    direction_x = max(-1.0, min(1.0, direction_x))
    direction_y = max(-1.0, min(1.0, direction_y))
    
    # Create offset based on direction
    offset = np.array([direction_x * 0.1, direction_y * 0.1])  # Small offset
    
    # Apply offset to base gaze
    if base_gaze.ndim == 1:
        modified = base_gaze + offset[:len(base_gaze)]
    else:
        modified = base_gaze + offset
    
    # Clamp result
    return np.clip(modified, -1.0, 1.0)


def create_retargeting_params_from_preset(preset_name: str) -> RetargetingParams:
    """
    Create retargeting parameters from preset.
    
    Args:
        preset_name: Name of preset
    
    Returns:
        RetargetingParams object
    """
    preset = ExpressionPresets.get_preset(preset_name)
    if not preset:
        # Return default if preset not found
        return RetargetingParams()
    
    return RetargetingParams(
        eye_open_ratio=preset.get('eye_open_ratio', 0.0),
        lip_open_ratio=preset.get('lip_open_ratio', 0.0),
        expression_intensity=preset.get('expression_intensity', 1.0),
        blink_frequency=preset.get('blink_frequency', 1.0),
        head_movement_intensity=preset.get('head_movement_intensity', 1.0),
        gaze_direction_x=preset.get('gaze_direction_x', 0.0),
        gaze_direction_y=preset.get('gaze_direction_y', 0.0),
        preset_name=preset_name
    )

