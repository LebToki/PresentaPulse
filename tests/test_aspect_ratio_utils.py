import unittest
from unittest.mock import patch

from aspect_ratio_utils import AspectRatioProcessor

class TestAspectRatioUtils(unittest.TestCase):
    @patch('aspect_ratio_utils.CV2_AVAILABLE', False)
    def test_smart_crop_missing_cv2(self):
        processor = AspectRatioProcessor()
        with self.assertRaises(ImportError) as context:
            processor.smart_crop('dummy_path.jpg', 100, 100)

        self.assertEqual(str(context.exception), "OpenCV required for cropping")

if __name__ == '__main__':
    unittest.main()
