"""
Enhanced batch processing with queue system and per-item progress tracking
"""
import os
import zipfile
import logging
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
from queue import Queue
import threading
import time

@dataclass
class BatchItem:
    """Represents a single batch processing item."""
    item_id: int
    image_path: str
    video_path: str
    image_idx: int
    video_idx: int
    status: str = 'pending'  # pending, processing, completed, failed
    result_path: Optional[str] = None
    error: Optional[str] = None
    start_time: Optional[float] = None
    end_time: Optional[float] = None

class BatchProcessor:
    """Enhanced batch processor with queue system."""
    
    def __init__(self, output_dir: Path):
        self.output_dir = output_dir
        self.output_dir.mkdir(exist_ok=True)
        self.queue = Queue()
        self.items: List[BatchItem] = []
        self.results: List[str] = []
        self.processing = False
        
    def create_batch_queue(self, images: List, videos: List) -> List[BatchItem]:
        """Create queue of batch items from images and videos."""
        items = []
        item_id = 0
        
        for img_idx, image_file in enumerate(images):
            for vid_idx, video_file in enumerate(videos):
                item_id += 1
                image_path = image_file.name if hasattr(image_file, 'name') else str(image_file)
                video_path = video_file.name if hasattr(video_file, 'name') else str(video_file)
                
                item = BatchItem(
                    item_id=item_id,
                    image_path=image_path,
                    video_path=video_path,
                    image_idx=img_idx + 1,
                    video_idx=vid_idx + 1
                )
                items.append(item)
                self.queue.put(item)
        
        self.items = items
        return items
    
    def get_queue_status(self) -> Dict:
        """Get current queue status."""
        total = len(self.items)
        pending = sum(1 for item in self.items if item.status == 'pending')
        processing = sum(1 for item in self.items if item.status == 'processing')
        completed = sum(1 for item in self.items if item.status == 'completed')
        failed = sum(1 for item in self.items if item.status == 'failed')
        
        return {
            'total': total,
            'pending': pending,
            'processing': processing,
            'completed': completed,
            'failed': failed,
            'progress': (completed / total * 100) if total > 0 else 0
        }
    
    def get_item_status_text(self) -> str:
        """Get formatted status text for all items."""
        lines = []
        lines.append(f"Queue Status: {len(self.items)} items")
        lines.append("")
        
        status = self.get_queue_status()
        lines.append(f"📊 Progress: {status['completed']}/{status['total']} completed ({status['progress']:.1f}%)")
        lines.append(f"⏳ Pending: {status['pending']} | 🔄 Processing: {status['processing']} | ❌ Failed: {status['failed']}")
        lines.append("")
        lines.append("Item Details:")
        lines.append("-" * 60)
        
        for item in self.items:
            status_icon = {
                'pending': '⏸️',
                'processing': '🔄',
                'completed': '✅',
                'failed': '❌'
            }.get(item.status, '❓')
            
            elapsed = ""
            if item.start_time and item.end_time:
                elapsed = f" ({item.end_time - item.start_time:.1f}s)"
            elif item.start_time:
                elapsed = f" (running: {time.time() - item.start_time:.1f}s)"
            
            lines.append(
                f"{status_icon} Item {item.item_id}: Image {item.image_idx} × Video {item.video_idx} "
                f"- {item.status.upper()}{elapsed}"
            )
            
            if item.error:
                lines.append(f"   Error: {item.error}")
            if item.result_path and os.path.exists(item.result_path):
                file_size = os.path.getsize(item.result_path) / (1024 * 1024)  # MB
                lines.append(f"   Output: {os.path.basename(item.result_path)} ({file_size:.2f} MB)")
        
        return "\n".join(lines)
    
    def create_zip_archive(self, zip_filename: Optional[str] = None) -> Optional[str]:
        """Create ZIP archive of all completed results."""
        if not self.results:
            return None
        
        if zip_filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            zip_filename = f"batch_results_{timestamp}.zip"
        
        zip_path = self.output_dir / zip_filename
        
        try:
            with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
                for idx, result_file in enumerate(self.results):
                    if os.path.exists(result_file):
                        # Create descriptive filename
                        filename = os.path.basename(result_file)
                        # Try to match with batch item
                        item_num = idx + 1
                        arcname = f"result_{item_num:03d}_{filename}"
                        zipf.write(result_file, arcname)
            
            logging.info(f"Created ZIP archive: {zip_path} with {len(self.results)} files")
            return str(zip_path)
        except Exception as e:
            logging.error(f"Failed to create ZIP archive: {str(e)}")
            return None

