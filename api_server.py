"""
REST API server for PresentaPulse
Provides API endpoints for programmatic access to animation generation
"""
from flask import Flask, request, jsonify, send_file
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from functools import wraps
import os
import logging
import hashlib
import secrets
from datetime import datetime, timedelta
from pathlib import Path
import json
from typing import Dict, Optional
import threading
from queue import Queue

try:
    from werkzeug.security import check_password_hash, generate_password_hash
    SECURITY_AVAILABLE = True
except ImportError:
    SECURITY_AVAILABLE = False
    logging.warning("werkzeug not available, using basic API key storage")

app = Flask(__name__)
app.config['SECRET_KEY'] = os.getenv('API_SECRET_KEY', secrets.token_hex(32))

# Rate limiting
limiter = Limiter(
    app=app,
    key_func=get_remote_address,
    default_limits=["1000 per hour"],
    storage_uri="memory://"
)

# API keys storage (in production, use a database)
API_KEYS_FILE = Path('api_keys.json')
api_keys: Dict[str, Dict] = {}
usage_stats: Dict[str, Dict] = {}


def load_api_keys():
    """Load API keys from file."""
    global api_keys
    if API_KEYS_FILE.exists():
        try:
            with open(API_KEYS_FILE, 'r') as f:
                api_keys = json.load(f)
        except Exception as e:
            logging.error(f"Failed to load API keys: {e}")
            api_keys = {}


def save_api_keys():
    """Save API keys to file."""
    try:
        with open(API_KEYS_FILE, 'w') as f:
            json.dump(api_keys, f, indent=2)
    except Exception as e:
        logging.error(f"Failed to save API keys: {e}")


def load_usage_stats():
    """Load usage statistics."""
    global usage_stats
    stats_file = Path('api_usage_stats.json')
    if stats_file.exists():
        try:
            with open(stats_file, 'r') as f:
                usage_stats = json.load(f)
        except Exception as e:
            logging.error(f"Failed to load usage stats: {e}")
            usage_stats = {}


def save_usage_stats():
    """Save usage statistics."""
    stats_file = Path('api_usage_stats.json')
    try:
        with open(stats_file, 'w') as f:
            json.dump(usage_stats, f, indent=2)
    except Exception as e:
        logging.error(f"Failed to save usage stats: {e}")


def track_usage(api_key: str, endpoint: str):
    """Track API usage."""
    if api_key not in usage_stats:
        usage_stats[api_key] = {
            'total_requests': 0,
            'endpoints': {},
            'last_used': None
        }
    
    usage_stats[api_key]['total_requests'] += 1
    usage_stats[api_key]['last_used'] = datetime.now().isoformat()
    
    if endpoint not in usage_stats[api_key]['endpoints']:
        usage_stats[api_key]['endpoints'][endpoint] = 0
    usage_stats[api_key]['endpoints'][endpoint] += 1
    
    save_usage_stats()


def require_api_key(f):
    """Decorator to require API key authentication."""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        api_key = request.headers.get('X-API-Key') or request.args.get('api_key')
        
        if not api_key:
            return jsonify({'error': 'API key required'}), 401
        
        if api_key not in api_keys:
            return jsonify({'error': 'Invalid API key'}), 401
        
        # Check if key is active
        if not api_keys[api_key].get('active', True):
            return jsonify({'error': 'API key is inactive'}), 403
        
        # Track usage
        track_usage(api_key, request.endpoint)
        
        return f(*args, **kwargs)
    
    return decorated_function


# Initialize on startup
load_api_keys()
load_usage_stats()


@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint."""
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'version': '1.0.0'
    })


@app.route('/api/v1/generate', methods=['POST'])
@limiter.limit("10 per minute")
@require_api_key
def generate_animation():
    """
    Generate animation from image and video.
    
    Request body:
    {
        "image_path": "path/to/image.jpg",
        "video_path": "path/to/video.mp4",
        "parameters": {
            "relative_motion": true,
            "do_crop": true,
            "remap": true,
            "crop_driving_video": false,
            "smoothing_strength": 0.0,
            "denoise_strength": 0.0,
            "stabilize": false
        }
    }
    
    Response:
    {
        "job_id": "abc123",
        "status": "processing",
        "estimated_time": 60
    }
    """
    try:
        data = request.get_json()
        
        if not data or 'image_path' not in data or 'video_path' not in data:
            return jsonify({'error': 'Missing required fields: image_path, video_path'}), 400
        
        # Generate job ID
        job_id = secrets.token_hex(8)
        
        # Queue job for processing (implement actual processing queue)
        # For now, return job ID
        
        return jsonify({
            'job_id': job_id,
            'status': 'queued',
            'message': 'Job queued for processing',
            'estimated_time': 60  # seconds
        }), 202
    
    except Exception as e:
        logging.error(f"Error in generate_animation: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/v1/job/<job_id>', methods=['GET'])
@require_api_key
def get_job_status(job_id: str):
    """Get job status."""
    # Implement job status checking
    return jsonify({
        'job_id': job_id,
        'status': 'completed',
        'progress': 100,
        'output_path': f'/api/v1/download/{job_id}'
    })


@app.route('/api/v1/download/<job_id>', methods=['GET'])
@require_api_key
def download_result(job_id: str):
    """Download generated video."""
    # Implement file download
    return jsonify({'error': 'Not implemented'}), 501


@app.route('/api/v1/keys', methods=['POST'])
@limiter.limit("5 per hour")
def create_api_key():
    """
    Create a new API key.
    
    Request body:
    {
        "name": "My Application",
        "rate_limit": 100
    }
    """
    try:
        data = request.get_json() or {}
        name = data.get('name', 'Unnamed')
        rate_limit = data.get('rate_limit', 100)
        
        # Generate API key
        api_key = secrets.token_urlsafe(32)
        
        # Store API key
        api_keys[api_key] = {
            'name': name,
            'created_at': datetime.now().isoformat(),
            'active': True,
            'rate_limit': rate_limit
        }
        
        save_api_keys()
        
        return jsonify({
            'api_key': api_key,
            'name': name,
            'created_at': api_keys[api_key]['created_at'],
            'rate_limit': rate_limit
        }), 201
    
    except Exception as e:
        logging.error(f"Error creating API key: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/v1/keys/<api_key>', methods=['DELETE'])
@require_api_key
def revoke_api_key(api_key: str):
    """Revoke an API key."""
    if api_key in api_keys:
        api_keys[api_key]['active'] = False
        save_api_keys()
        return jsonify({'message': 'API key revoked'}), 200
    return jsonify({'error': 'API key not found'}), 404


@app.route('/api/v1/stats', methods=['GET'])
@require_api_key
def get_usage_stats():
    """Get usage statistics for current API key."""
    api_key = request.headers.get('X-API-Key') or request.args.get('api_key')
    
    if api_key in usage_stats:
        return jsonify(usage_stats[api_key]), 200
    
    return jsonify({
        'total_requests': 0,
        'endpoints': {},
        'last_used': None
    }), 200


@app.route('/api/v1/webhook', methods=['POST'])
@limiter.limit("20 per minute")
@require_api_key
def register_webhook():
    """
    Register a webhook for async processing.
    
    Request body:
    {
        "url": "https://example.com/webhook",
        "events": ["job.completed", "job.failed"]
    }
    """
    try:
        data = request.get_json()
        
        if not data or 'url' not in data:
            return jsonify({'error': 'Missing required field: url'}), 400
        
        webhook_id = secrets.token_hex(8)
        
        # Store webhook (implement webhook storage)
        
        return jsonify({
            'webhook_id': webhook_id,
            'url': data['url'],
            'events': data.get('events', ['job.completed']),
            'status': 'active'
        }), 201
    
    except Exception as e:
        logging.error(f"Error registering webhook: {e}")
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)

