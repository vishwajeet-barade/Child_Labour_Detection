"""
Flask backend API for Child Labour Detection
Handles video uploads and processing
"""

from flask import Flask, request, jsonify, send_from_directory, send_file
from flask_cors import CORS
import os
import uuid
from werkzeug.utils import secure_filename
from detect_actions import main as process_video
import threading

app = Flask(__name__)
# Enable CORS for all routes
CORS(app)

# Configuration
UPLOAD_FOLDER = 'uploads'
OUTPUT_FOLDER = 'outputs'
ALLOWED_EXTENSIONS = {'mp4', 'avi', 'mov', 'webm', 'mkv', 'flv'}
MAX_FILE_SIZE = 500 * 1024 * 1024  # 500MB

# Create necessary directories
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/api/upload', methods=['POST'])
def upload_video():
    """Handle video upload and processing"""
    if 'video' not in request.files:
        return jsonify({'error': 'No video file provided'}), 400
    
    file = request.files['video']
    
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    if not allowed_file(file.filename):
        return jsonify({'error': 'Invalid file type. Allowed: MP4, AVI, MOV, WebM, MKV, FLV'}), 400
    
    try:
        # Generate unique filename
        file_ext = secure_filename(file.filename).rsplit('.', 1)[1].lower()
        unique_id = str(uuid.uuid4())
        input_filename = f"{unique_id}.{file_ext}"
        input_path = os.path.join(UPLOAD_FOLDER, input_filename)
        
        # Save uploaded file
        file.save(input_path)
        
        # Generate output filename
        output_filename = f"{unique_id}_processed.mp4"
        output_path = os.path.join(OUTPUT_FOLDER, output_filename)
        
        # Process video in background thread
        processing_thread = threading.Thread(
            target=process_video_background,
            args=(input_path, output_path)
        )
        processing_thread.daemon = True
        processing_thread.start()
        
        return jsonify({
            'message': 'Video uploaded successfully',
            'input_filename': input_filename,
            'output_filename': output_filename,
            'status': 'processing'
        }), 200
        
    except Exception as e:
        return jsonify({'error': f'Error processing video: {str(e)}'}), 500

def process_video_background(input_path, output_path):
    """Process video in background thread"""
    try:
        # Process video using detect_actions.py main function
        process_video(
            video_source=input_path,
            model_dir='models',
            out_path=output_path,
            conf_threshold=0.6
        )
    except Exception as e:
        print(f"Error processing video: {str(e)}")

@app.route('/api/status/<filename>', methods=['GET'])
def check_status(filename):
    """Check if processed video is ready"""
    import time
    
    output_path = os.path.join(OUTPUT_FOLDER, filename)
    
    if os.path.exists(output_path):
        file_size = os.path.getsize(output_path)
        
        # Check if file size is reasonable (at least 1KB)
        if file_size > 1024:  # At least 1KB
            # Check file modification time multiple times to ensure it's stable
            # We need to ensure the file hasn't been modified recently
            mod_time = os.path.getmtime(output_path)
            time_since_mod = time.time() - mod_time
            
            # Require at least 3 seconds of stability to ensure processing is complete
            # This is important because video writing can be slow
            if time_since_mod >= 3.0:
                # Double-check: get file size again after a small delay
                # If file size is the same, it's likely complete
                time.sleep(0.1)
                new_file_size = os.path.getsize(output_path)
                
                if new_file_size == file_size:
                    return jsonify({
                        'status': 'ready',
                        'filename': filename,
                        'file_size': file_size
                    }), 200
        
        # File exists but might still be processing
        return jsonify({
            'status': 'processing',
            'filename': filename,
            'file_size': file_size if file_size > 0 else 0
        }), 200
    else:
        return jsonify({
            'status': 'processing',
            'filename': filename
        }), 200

@app.route('/outputs/<path:filename>')
def get_output_video(filename):
    """Serve processed video file as static file - browser handles playback"""
    # Simply serve the file from outputs directory
    # Flask automatically handles range requests for video streaming
    return send_from_directory(OUTPUT_FOLDER, filename)

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({'status': 'healthy'}), 200

if __name__ == '__main__':
    print("Starting Flask server...")
    print("Backend API available at http://localhost:5000")
    app.run(host='0.0.0.0', port=5000, debug=True)

