import unittest
from unittest.mock import patch
import sys
import os

# Add parent directory to path so we can import export_utils
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from export_utils import find_ffmpeg

class TestFindFFmpeg(unittest.TestCase):

    @patch('shutil.which')
    def test_ffmpeg_found_in_path(self, mock_which):
        """Test when ffmpeg is found in PATH via shutil.which"""
        mock_which.return_value = '/mock/path/to/ffmpeg'
        result = find_ffmpeg()
        self.assertEqual(result, '/mock/path/to/ffmpeg')
        mock_which.assert_called_once_with('ffmpeg')

    @patch('shutil.which', return_value=None)
    @patch('platform.system', return_value='Windows')
    @patch('os.path.exists')
    def test_windows_fallback_found(self, mock_exists, mock_system, mock_which):
        """Test Windows fallback path when ffmpeg is found in common locations"""
        # Make os.path.exists return True for the first common path
        mock_exists.side_effect = lambda path: path == 'C:\\ffmpeg\\bin\\ffmpeg.exe'

        result = find_ffmpeg()
        self.assertEqual(result, 'C:\\ffmpeg\\bin\\ffmpeg.exe')
        mock_which.assert_called_once_with('ffmpeg')
        mock_system.assert_called_once()
        mock_exists.assert_called_once_with('C:\\ffmpeg\\bin\\ffmpeg.exe')

    @patch('shutil.which', return_value=None)
    @patch('platform.system', return_value='Windows')
    @patch('os.path.exists', return_value=False)
    def test_windows_fallback_not_found(self, mock_exists, mock_system, mock_which):
        """Test Windows fallback when ffmpeg is NOT found in common locations"""
        result = find_ffmpeg()
        self.assertEqual(result, 'ffmpeg') # Default return value
        mock_which.assert_called_once_with('ffmpeg')
        mock_system.assert_called_once()
        self.assertEqual(mock_exists.call_count, 2) # Two common Windows paths checked

    @patch('shutil.which', return_value=None)
    @patch('platform.system', return_value='Linux')
    @patch('os.path.exists')
    def test_linux_fallback_found(self, mock_exists, mock_system, mock_which):
        """Test Linux/Mac fallback path when ffmpeg is found in common locations"""
        # Make os.path.exists return True for the second common path
        mock_exists.side_effect = lambda path: path == '/usr/local/bin/ffmpeg'

        result = find_ffmpeg()
        self.assertEqual(result, '/usr/local/bin/ffmpeg')
        mock_which.assert_called_once_with('ffmpeg')
        mock_system.assert_called_once()
        self.assertEqual(mock_exists.call_count, 2) # Checked first, then found on second

    @patch('shutil.which', return_value=None)
    @patch('platform.system', return_value='Linux')
    @patch('os.path.exists', return_value=False)
    def test_linux_fallback_not_found(self, mock_exists, mock_system, mock_which):
        """Test Linux/Mac fallback when ffmpeg is NOT found in common locations"""
        result = find_ffmpeg()
        self.assertEqual(result, 'ffmpeg') # Default return value
        mock_which.assert_called_once_with('ffmpeg')
        mock_system.assert_called_once()
        self.assertEqual(mock_exists.call_count, 2) # Two common Linux paths checked

if __name__ == '__main__':
    unittest.main()
