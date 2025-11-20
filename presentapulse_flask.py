"""
Flask wrapper for PresentaPulse to integrate with unified platform
This allows PresentaPulse to be served alongside Chat-with-Ollama and MoA
"""
from flask import Flask, render_template_string, redirect, url_for
import subprocess
import os
from pathlib import Path

app = Flask(__name__)

# PresentaPulse configuration
PRESENTAPULSE_DIR = Path(__file__).parent
PRESENTAPULSE_APP = PRESENTAPULSE_DIR / 'app.py'
PRESENTAPULSE_PORT = 8080

@app.route('/')
def index():
    """Redirect to PresentaPulse Gradio app"""
    # Check if app.py exists
    if not PRESENTAPULSE_APP.exists():
        return render_template_string("""
        <!DOCTYPE html>
        <html>
        <head>
            <title>PresentaPulse - Setup Required</title>
            <style>
                body {
                    font-family: 'Inter', sans-serif;
                    background: #0d1117;
                    color: #e6edf3;
                    display: flex;
                    align-items: center;
                    justify-content: center;
                    min-height: 100vh;
                    margin: 0;
                }
                .container {
                    background: rgba(255, 255, 255, 0.03);
                    backdrop-filter: blur(20px);
                    border: 1px solid rgba(255, 255, 255, 0.08);
                    border-radius: 20px;
                    padding: 48px;
                    max-width: 600px;
                    text-align: center;
                }
                h1 { color: #667eea; margin-bottom: 16px; }
                p { color: rgba(230, 237, 243, 0.6); line-height: 1.6; }
                .code {
                    background: rgba(0, 0, 0, 0.3);
                    padding: 12px;
                    border-radius: 8px;
                    font-family: monospace;
                    margin: 16px 0;
                    text-align: left;
                }
            </style>
        </head>
        <body>
            <div class="container">
                <h1>🎬 PresentaPulse Setup Required</h1>
                <p>PresentaPulse requires LivePortrait integration. Please follow these steps:</p>
                <div class="code">
                    1. Clone LivePortrait:<br>
                    git clone https://github.com/KwaiVGI/LivePortrait.git<br><br>
                    2. Copy src/ directory to PresentaPulse folder<br>
                    3. Install requirements: pip install -r requirements.txt<br>
                    4. Run: python app.py
                </div>
                <p><a href="/hub.php" style="color: #58a6ff;">← Back to Platform Hub</a></p>
            </div>
        </body>
        </html>
        """)
    
    # Redirect to Gradio app (running on port 8080)
    return redirect(f'http://localhost:{PRESENTAPULSE_PORT}', code=302)

@app.route('/health')
def health():
    """Health check endpoint"""
    return {'status': 'ok', 'app': 'PresentaPulse'}

if __name__ == '__main__':
    # Check if Gradio app is running, if not, start it
    import socket
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    result = sock.connect_ex(('localhost', PRESENTAPULSE_PORT))
    sock.close()
    
    if result != 0:
        print(f"Starting PresentaPulse Gradio app on port {PRESENTAPULSE_PORT}...")
        # Start Gradio app in background
        subprocess.Popen(['python', str(PRESENTAPULSE_APP)], cwd=str(PRESENTAPULSE_DIR))
    
    app.run(debug=True, host='0.0.0.0', port=5001)

