"""
History and gallery management for generated videos and images
"""
import json
import logging
import shutil
from pathlib import Path
from typing import List, Dict, Optional
from datetime import datetime
from dataclasses import dataclass, asdict
import hashlib

@dataclass
class GenerationHistory:
    """Represents a single generation history entry."""
    id: str
    timestamp: str
    image_path: str
    video_path: str
    output_path: str
    parameters: Dict
    thumbnail_path: Optional[str] = None
    preview_path: Optional[str] = None
    
    def to_dict(self):
        """Convert to dictionary."""
        return asdict(self)


class HistoryManager:
    """Manage generation history and gallery."""
    
    def __init__(self, history_dir: Path, max_entries: int = 100):
        """
        Initialize history manager.
        
        Args:
            history_dir: Directory to store history files
            max_entries: Maximum number of history entries to keep
        """
        self.history_dir = Path(history_dir)
        self.history_dir.mkdir(parents=True, exist_ok=True)
        self.history_file = self.history_dir / 'history.json'
        self.max_entries = max_entries
        self.history: List[GenerationHistory] = []
        self.load_history()
    
    def load_history(self):
        """Load history from file."""
        try:
            if self.history_file.exists():
                with open(self.history_file, 'r') as f:
                    data = json.load(f)
                    self.history = [GenerationHistory(**entry) for entry in data]
                    # Sort by timestamp (newest first)
                    self.history.sort(key=lambda x: x.timestamp, reverse=True)
                    # Keep only max_entries
                    if len(self.history) > self.max_entries:
                        self.history = self.history[:self.max_entries]
        except Exception as e:
            logging.error(f"Failed to load history: {e}")
            self.history = []
    
    def save_history(self):
        """Save history to file."""
        try:
            # Keep only max_entries
            if len(self.history) > self.max_entries:
                # Remove oldest entries
                to_remove = self.history[self.max_entries:]
                for entry in to_remove:
                    self._cleanup_entry(entry)
                self.history = self.history[:self.max_entries]
            
            data = [entry.to_dict() for entry in self.history]
            with open(self.history_file, 'w') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            logging.error(f"Failed to save history: {e}")
    
    def add_entry(self, image_path: str, video_path: str, output_path: str,
                  parameters: Dict, thumbnail_path: Optional[str] = None,
                  preview_path: Optional[str] = None) -> str:
        """
        Add a new history entry.
        
        Returns:
            Entry ID
        """
        # Generate ID from timestamp and paths
        timestamp = datetime.now().isoformat()
        id_string = f"{timestamp}_{image_path}_{video_path}"
        entry_id = hashlib.md5(id_string.encode()).hexdigest()[:12]
        
        entry = GenerationHistory(
            id=entry_id,
            timestamp=timestamp,
            image_path=str(image_path),
            video_path=str(video_path),
            output_path=str(output_path),
            parameters=parameters,
            thumbnail_path=str(thumbnail_path) if thumbnail_path else None,
            preview_path=str(preview_path) if preview_path else None
        )
        
        # Add to beginning (newest first)
        self.history.insert(0, entry)
        self.save_history()
        
        return entry_id
    
    def get_entry(self, entry_id: str) -> Optional[GenerationHistory]:
        """Get history entry by ID."""
        for entry in self.history:
            if entry.id == entry_id:
                return entry
        return None
    
    def get_recent(self, limit: int = 10) -> List[GenerationHistory]:
        """Get recent history entries."""
        return self.history[:limit]
    
    def get_all(self) -> List[GenerationHistory]:
        """Get all history entries."""
        return self.history
    
    def delete_entry(self, entry_id: str) -> bool:
        """Delete history entry."""
        for i, entry in enumerate(self.history):
            if entry.id == entry_id:
                self._cleanup_entry(entry)
                self.history.pop(i)
                self.save_history()
                return True
        return False
    
    def _cleanup_entry(self, entry: GenerationHistory):
        """Clean up files associated with entry."""
        # Don't delete original output files, just remove from history
        pass
    
    def clear_history(self):
        """Clear all history."""
        self.history = []
        self.save_history()
    
    def create_thumbnail(self, video_path: str, output_path: str, frame_number: int = 1) -> bool:
        """
        Create thumbnail from video.
        
        Args:
            video_path: Path to video
            output_path: Path to save thumbnail
            frame_number: Frame number to extract (1-based)
        
        Returns:
            True if successful
        """
        try:
            import subprocess
            import shutil
            
            ffmpeg_path = shutil.which('ffmpeg') or 'ffmpeg'
            
            command = [
                ffmpeg_path,
                '-i', str(video_path),
                '-vf', f'select=eq(n\\,{frame_number-1})',
                '-vframes', '1',
                '-y',
                str(output_path)
            ]
            
            subprocess.run(command, capture_output=True, check=True)
            return True
        except Exception as e:
            logging.error(f"Failed to create thumbnail: {e}")
            return False

