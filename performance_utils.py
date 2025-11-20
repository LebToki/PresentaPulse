"""
Performance optimization utilities for GPU memory management, processing queues, and multi-GPU support
"""
import logging
import os
import json
import pickle
from pathlib import Path
from typing import Optional, Dict, List, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime
import threading
from queue import Queue, PriorityQueue

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    logging.warning("PyTorch not available for GPU memory management")

try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False
    logging.warning("psutil not available for system monitoring")


@dataclass
class ProcessingState:
    """State information for resuming interrupted processing."""
    job_id: str
    image_path: str
    video_path: str
    parameters: Dict
    current_step: str
    progress: float
    checkpoint_path: Optional[str] = None
    created_at: str = None
    updated_at: str = None
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now().isoformat()
        if self.updated_at is None:
            self.updated_at = datetime.now().isoformat()


class GPUMemoryManager:
    """Manage GPU memory allocation and cleanup."""
    
    def __init__(self, device_id: int = 0, low_memory_mode: bool = False):
        """
        Initialize GPU memory manager.
        
        Args:
            device_id: GPU device ID
            low_memory_mode: Enable low-memory optimizations
        """
        self.device_id = device_id
        self.low_memory_mode = low_memory_mode
        self.device = None
        
        if TORCH_AVAILABLE and torch.cuda.is_available():
            self.device = torch.device(f'cuda:{device_id}')
            self._initial_memory = self.get_memory_usage()
        else:
            logging.warning("CUDA not available, GPU memory management disabled")
    
    def get_memory_usage(self) -> Dict[str, float]:
        """Get current GPU memory usage in MB."""
        if not TORCH_AVAILABLE or not torch.cuda.is_available():
            return {'allocated': 0.0, 'reserved': 0.0, 'free': 0.0}
        
        allocated = torch.cuda.memory_allocated(self.device) / (1024 ** 2)  # MB
        reserved = torch.cuda.memory_reserved(self.device) / (1024 ** 2)  # MB
        total = torch.cuda.get_device_properties(self.device).total_memory / (1024 ** 2)  # MB
        free = total - reserved
        
        return {
            'allocated': allocated,
            'reserved': reserved,
            'free': free,
            'total': total
        }
    
    def clear_cache(self):
        """Clear GPU cache."""
        if TORCH_AVAILABLE and torch.cuda.is_available():
            torch.cuda.empty_cache()
            logging.info("GPU cache cleared")
    
    def optimize_for_low_memory(self):
        """Apply low-memory optimizations."""
        if not TORCH_AVAILABLE or not torch.cuda.is_available():
            return
        
        # Enable memory-efficient attention if available
        try:
            torch.backends.cuda.enable_flash_sdp(False)  # Disable flash attention (uses more memory)
        except:
            pass
        
        # Set memory fraction for low-memory mode
        if self.low_memory_mode:
            try:
                torch.cuda.set_per_process_memory_fraction(0.7, self.device)  # Use max 70% of GPU
            except:
                pass
        
        self.clear_cache()
        logging.info("Low-memory optimizations applied")
    
    def get_memory_info_string(self) -> str:
        """Get formatted memory usage string."""
        mem = self.get_memory_usage()
        return f"GPU Memory: {mem['allocated']:.1f}MB allocated, {mem['reserved']:.1f}MB reserved, {mem['free']:.1f}MB free"
    
    def check_memory_available(self, required_mb: float) -> bool:
        """Check if required memory is available."""
        mem = self.get_memory_usage()
        return mem['free'] >= required_mb


class ProcessingQueue:
    """Thread-safe processing queue with priority support."""
    
    def __init__(self, max_size: int = 100):
        self.queue = Queue(maxsize=max_size)
        self.processing = False
        self.current_job = None
        self.lock = threading.Lock()
    
    def add_job(self, job_data: Dict, priority: int = 0):
        """Add job to queue with priority."""
        try:
            job_with_priority = (priority, job_data)
            self.queue.put_nowait(job_with_priority)
            return True
        except:
            return False
    
    def get_job(self) -> Optional[Dict]:
        """Get next job from queue (highest priority first)."""
        if self.queue.empty():
            return None
        
        # Get all jobs and sort by priority
        jobs = []
        while not self.queue.empty():
            jobs.append(self.queue.get())
        
        # Sort by priority (higher priority first)
        jobs.sort(key=lambda x: x[0], reverse=True)
        
        # Put remaining jobs back
        for job in jobs[1:]:
            self.queue.put(job)
        
        if jobs:
            return jobs[0][1]  # Return job data (without priority)
        return None
    
    def size(self) -> int:
        """Get queue size."""
        return self.queue.qsize()
    
    def clear(self):
        """Clear all jobs from queue."""
        while not self.queue.empty():
            try:
                self.queue.get_nowait()
            except:
                break


class ProcessingCheckpoint:
    """Manage processing checkpoints for resuming interrupted jobs."""
    
    def __init__(self, checkpoint_dir: Path):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    def save_checkpoint(self, state: ProcessingState, checkpoint_data: Optional[Dict] = None):
        """Save processing checkpoint."""
        try:
            checkpoint_file = self.checkpoint_dir / f"{state.job_id}.pkl"
            
            # Save state
            with open(checkpoint_file, 'wb') as f:
                pickle.dump(state, f)
            
            # Save additional data if provided
            if checkpoint_data:
                data_file = self.checkpoint_dir / f"{state.job_id}_data.json"
                with open(data_file, 'w') as f:
                    json.dump(checkpoint_data, f)
            
            logging.info(f"Checkpoint saved: {checkpoint_file}")
            return True
        except Exception as e:
            logging.error(f"Failed to save checkpoint: {e}")
            return False
    
    def load_checkpoint(self, job_id: str) -> Optional[ProcessingState]:
        """Load processing checkpoint."""
        try:
            checkpoint_file = self.checkpoint_dir / f"{job_id}.pkl"
            if not checkpoint_file.exists():
                return None
            
            with open(checkpoint_file, 'rb') as f:
                state = pickle.load(f)
            
            logging.info(f"Checkpoint loaded: {checkpoint_file}")
            return state
        except Exception as e:
            logging.error(f"Failed to load checkpoint: {e}")
            return None
    
    def list_checkpoints(self) -> List[str]:
        """List all available checkpoints."""
        checkpoints = []
        for file in self.checkpoint_dir.glob("*.pkl"):
            checkpoints.append(file.stem)
        return checkpoints
    
    def delete_checkpoint(self, job_id: str):
        """Delete checkpoint."""
        checkpoint_file = self.checkpoint_dir / f"{job_id}.pkl"
        data_file = self.checkpoint_dir / f"{job_id}_data.json"
        
        if checkpoint_file.exists():
            checkpoint_file.unlink()
        if data_file.exists():
            data_file.unlink()


class MultiGPUSupport:
    """Manage multi-GPU processing."""
    
    def __init__(self):
        self.available_devices = []
        self.device_loads = {}
        
        if TORCH_AVAILABLE and torch.cuda.is_available():
            self.available_devices = list(range(torch.cuda.device_count()))
            for device_id in self.available_devices:
                self.device_loads[device_id] = 0.0
    
    def get_available_devices(self) -> List[int]:
        """Get list of available GPU devices."""
        return self.available_devices
    
    def get_device_load(self, device_id: int) -> float:
        """Get current load on device (0.0 to 1.0)."""
        if device_id not in self.device_loads:
            return 1.0  # Assume fully loaded if unknown
        
        return self.device_loads[device_id]
    
    def select_best_device(self) -> Optional[int]:
        """Select device with lowest load."""
        if not self.available_devices:
            return None
        
        best_device = None
        lowest_load = 1.0
        
        for device_id in self.available_devices:
            load = self.get_device_load(device_id)
            if load < lowest_load:
                lowest_load = load
                best_device = device_id
        
        return best_device
    
    def update_device_load(self, device_id: int, load: float):
        """Update device load."""
        if device_id in self.device_loads:
            self.device_loads[device_id] = max(0.0, min(1.0, load))
    
    def get_device_memory_info(self, device_id: int) -> Dict[str, float]:
        """Get memory info for specific device."""
        if not TORCH_AVAILABLE or not torch.cuda.is_available():
            return {'allocated': 0.0, 'reserved': 0.0, 'free': 0.0}
        
        device = torch.device(f'cuda:{device_id}')
        allocated = torch.cuda.memory_allocated(device) / (1024 ** 2)
        reserved = torch.cuda.memory_reserved(device) / (1024 ** 2)
        total = torch.cuda.get_device_properties(device).total_memory / (1024 ** 2)
        free = total - reserved
        
        return {
            'allocated': allocated,
            'reserved': reserved,
            'free': free,
            'total': total
        }


class SystemMonitor:
    """Monitor system resources."""
    
    def __init__(self):
        self.monitoring = False
        self.monitor_thread = None
    
    def get_cpu_usage(self) -> float:
        """Get CPU usage percentage."""
        if PSUTIL_AVAILABLE:
            return psutil.cpu_percent(interval=0.1)
        return 0.0
    
    def get_ram_usage(self) -> Dict[str, float]:
        """Get RAM usage in MB."""
        if PSUTIL_AVAILABLE:
            mem = psutil.virtual_memory()
            return {
                'total': mem.total / (1024 ** 2),
                'used': mem.used / (1024 ** 2),
                'free': mem.free / (1024 ** 2),
                'percent': mem.percent
            }
        return {'total': 0.0, 'used': 0.0, 'free': 0.0, 'percent': 0.0}
    
    def get_system_info(self) -> Dict:
        """Get comprehensive system information."""
        info = {
            'cpu_usage': self.get_cpu_usage(),
            'ram': self.get_ram_usage(),
        }
        
        if TORCH_AVAILABLE and torch.cuda.is_available():
            info['gpu_count'] = torch.cuda.device_count()
            info['gpus'] = []
            for i in range(torch.cuda.device_count()):
                props = torch.cuda.get_device_properties(i)
                mem_manager = GPUMemoryManager(device_id=i)
                mem_info = mem_manager.get_memory_usage()
                info['gpus'].append({
                    'id': i,
                    'name': props.name,
                    'memory': mem_info,
                    'compute_capability': f"{props.major}.{props.minor}"
                })
        
        return info

