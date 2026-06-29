import unittest
from aspect_ratio_utils import AspectRatioProcessor

class TestAspectRatioUtils(unittest.TestCase):
    def setUp(self):
        self.processor = AspectRatioProcessor()

    def test_calculate_target_size_valid_aspect_ratio(self):
        """Test valid aspect ratio"""
        # 16:9 aspect ratio
        result = self.processor.calculate_target_size(
            original_width=800,
            original_height=600,
            aspect_ratio='16:9',
            max_dimension=1024
        )
        self.assertEqual(result, (800, 450))

    def test_calculate_target_size_invalid_aspect_ratio(self):
        """Test that invalid aspect ratio falls back to 1:1"""
        result = self.processor.calculate_target_size(
            original_width=800,
            original_height=600,
            aspect_ratio='invalid_ratio',
            max_dimension=1024
        )
        self.assertEqual(result, (600, 600))

    def test_calculate_target_size_custom(self):
        """Test custom aspect ratio"""
        result = self.processor.calculate_target_size(
            original_width=800,
            original_height=600,
            aspect_ratio='custom',
            custom_width=555,
            custom_height=444
        )
        self.assertEqual(result, (555, 444))

    def test_calculate_target_size_custom_missing_dimensions(self):
        """Test custom aspect ratio with missing dimensions falls back to 1:1"""
        # Both None
        result1 = self.processor.calculate_target_size(
            original_width=800,
            original_height=600,
            aspect_ratio='custom',
            custom_width=None,
            custom_height=None,
            max_dimension=1024
        )
        self.assertEqual(result1, (600, 600))

        # One None
        result2 = self.processor.calculate_target_size(
            original_width=800,
            original_height=600,
            aspect_ratio='custom',
            custom_width=800,
            custom_height=None,
            max_dimension=1024
        )
        self.assertEqual(result2, (600, 600))

if __name__ == '__main__':
    unittest.main()
