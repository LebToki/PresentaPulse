import pytest
from aspect_ratio_utils import AspectRatioProcessor

def test_calculate_target_size_custom():
    processor = AspectRatioProcessor()


    # Test custom dimensions
    width, height = processor.calculate_target_size(
        original_width=800,
        original_height=600,
        aspect_ratio='custom',
        custom_width=1280,
        custom_height=720,
    )
    assert width == 1280
    assert height == 720

def test_calculate_target_size_missing_custom_dims():
    processor = AspectRatioProcessor()


    # Test custom aspect ratio but missing custom dimensions
    # Should default to 1:1, so width=600, height=600 for original 800x600
    width, height = processor.calculate_target_size(
        original_width=800,
        original_height=600,
        aspect_ratio='custom',
        custom_width=None,
        custom_height=720,  # One missing is enough
    )
    assert width == 600
    assert height == 600


    width2, height2 = processor.calculate_target_size(
        original_width=800,
        original_height=600,
        aspect_ratio='custom',
    )
    assert width2 == 600
    assert height2 == 600

def test_calculate_target_size_invalid_ratio():
    processor = AspectRatioProcessor()


    # Test fallback to 1:1 for invalid ratio
    width, height = processor.calculate_target_size(
        original_width=800,
        original_height=600,
        aspect_ratio='invalid_ratio',
    )
    # Target aspect ratio is 1:1, original is 800:600 (4:3) -> wider than target
    # Will fit to height, so target_height = 600, target_width = 600
    assert width == 600
    assert height == 600

def test_calculate_target_size_wider_than_target():
    processor = AspectRatioProcessor()


    # Test wider image fitting to target aspect ratio (9:16)
    width, height = processor.calculate_target_size(
        original_width=1920,
        original_height=1080,
        aspect_ratio='9:16',
        max_dimension=1024
    )
    # Original is 16:9, target is 9:16
    # Original (1920/1080) > Target (9/16) -> wider than target
    # Fits to height: target_height = min(1024, 1080) = 1024
    # target_width = 1024 * (9/16) = 576
    assert width == 576
    assert height == 1024

def test_calculate_target_size_taller_than_target():
    processor = AspectRatioProcessor()


    # Test taller image fitting to target aspect ratio (16:9)
    width, height = processor.calculate_target_size(
        original_width=1080,
        original_height=1920,
        aspect_ratio='16:9',
        max_dimension=1024
    )
    # Original is 9:16, target is 16:9
    # Original (1080/1920) < Target (16/9) -> taller than target
    # Fits to width: target_width = min(1024, 1080) = 1024
    # target_height = 1024 / (16/9) = 576
    assert width == 1024
    assert height == 576

def test_calculate_target_size_respects_max_dimension():
    processor = AspectRatioProcessor()


    # Test that max_dimension limits both width and height when needed
    width, height = processor.calculate_target_size(
        original_width=2000,
        original_height=2000,
        aspect_ratio='1:1',
        max_dimension=500
    )
    assert width == 500
    assert height == 500

def test_calculate_target_size_no_scale_needed():
    processor = AspectRatioProcessor()


    # Test when dimensions are smaller than max_dimension
    width, height = processor.calculate_target_size(
        original_width=800,
        original_height=800,
        aspect_ratio='1:1',
        max_dimension=1024
    )
    assert width == 800
    assert height == 800

def test_calculate_target_size_exact_ratio():
    processor = AspectRatioProcessor()


    # Test when the original image has the exact same aspect ratio as target
    width, height = processor.calculate_target_size(
        original_width=1920,
        original_height=1080,
        aspect_ratio='16:9',
        max_dimension=1024
    )
    # Original is 16:9 (1920/1080 = 1.777...)
    # Target is 16:9 (16/9 = 1.777...)
    # Because 1920/1080 is NOT > 16/9 (it is equal), it falls into the else block:
    # "Image is taller than target - fit to width" (or exact match)
    # target_width = min(1024, 1920) = 1024
    # target_height = 1024 / (16/9) = 576
    assert width == 1024
    assert height == 576
