import cv2
import numpy as np
from flask import Flask, Response, jsonify, send_from_directory
from flask_cors import CORS
import json
import threading
import time
import os
import multiprocessing as mp
from multiprocessing import shared_memory

# Use multiprocessing's shared memory for cross-process communication
# Default dimensions - update as needed
IMAGE_HEIGHT = 15
IMAGE_WIDTH = 20

def create_shared_memory():
    """Create shared memory for the image array"""
    # Create a shared memory segment for the image
    array_size = IMAGE_HEIGHT * IMAGE_WIDTH
    try:
        # First try to access existing shared memory
        shm = shared_memory.SharedMemory(name="webcam_image")
    except:
        # If it doesn't exist, create it
        shm = shared_memory.SharedMemory(
            name="webcam_image", 
            create=True, 
            size=array_size
        )
        print("Created new shared memory")
    
    # Also create a small shared memory segment to indicate availability
    try:
        flag_shm = shared_memory.SharedMemory(name="image_available")
    except:
        flag_shm = shared_memory.SharedMemory(
            name="image_available", 
            create=True, 
            size=1
        )
        # Initialize flag to 0 (not available)
        flag_shm.buf[0] = 0
    
    return shm, flag_shm

# Create shared memory at module load time
shared_mem, flag_mem = create_shared_memory()

# Initialize Flask app
app = Flask(__name__, static_folder='../frontend/build')
CORS(app)

from process_image import process_image_simplified, process_image

# Function to capture from webcam and process images
def webcam_capture_and_process():
    # Initialize webcam
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return
    
    history_frames = None
    color_history = None

    # Create NumPy array from the shared memory
    img_array = np.ndarray(
        (IMAGE_HEIGHT, IMAGE_WIDTH),
        dtype=np.uint8,
        buffer=shared_mem.buf
    )

    try:
        while True:
            # Read frame from webcam
            ret, frame = cap.read()
            
            if not ret:
                print("Failed to get frame from webcam")
                break
                
            # Process the image 
            processed_frame, history_frames, color_history, color_matrix = process_image_simplified(
                frame,
                white_threshold=120,
                color_variance_threshold=15,
                history_frames=history_frames,
                history_size=10,
                stabilization_weight=0.7,
                color_history=color_history
            )
            
            # Copy the data to shared memory
            # First make sure dimensions match
            if color_matrix.shape == (IMAGE_HEIGHT, IMAGE_WIDTH):
                # Make data available to other processes
                np.copyto(img_array, color_matrix)
                # Set the flag to indicate data is available
                flag_mem.buf[0] = 1
                # print(f"Updated shared memory: shape={color_matrix.shape}, values={np.unique(color_matrix, return_counts=True)}")
            else:
                print(f"Warning: Matrix shape mismatch. Expected {(IMAGE_HEIGHT, IMAGE_WIDTH)}, got {color_matrix.shape}")
            
            # Get the original dimensions
            height, width = frame.shape[:2]
            
            # Resize the processed frame to match the original dimensions
            enlarged_processed = cv2.resize(processed_frame, (width, height), 
                                           interpolation=cv2.INTER_NEAREST)
            
            # Create a side-by-side display
            side_by_side = np.zeros((height, width * 2, 3), dtype=np.uint8)
            side_by_side[:, :width] = frame
            side_by_side[:, width:] = enlarged_processed
            side_by_side[:, width-1:width+1] = [255, 255, 255]  # White line
            
            # Add labels
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(side_by_side, 'Original', (10, 30), font, 1, (255, 255, 255), 2)
            cv2.putText(side_by_side, 'Processed', (width + 10, 30), font, 1, (255, 255, 255), 2)
            
            # Display the side-by-side view
            cv2.imshow('Original vs Processed', side_by_side)

            # Exit if 'q' is pressed
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
            # Brief pause to reduce CPU usage
            time.sleep(0.05)
    finally:
        # Release resources
        cap.release()
        cv2.destroyAllWindows()

# Route to serve the numpy array as JSON
@app.route('/api/image-data', methods=['GET'])
def get_image_data():
    # Create a NumPy array from the shared memory
    img_array = np.ndarray(
        (IMAGE_HEIGHT, IMAGE_WIDTH),
        dtype=np.uint8,
        buffer=shared_mem.buf
    )
    
    # Check if data is available
    is_available = flag_mem.buf[0] == 1
    
    if not is_available:
        return jsonify({"error": "No processed image available yet"}), 404
    
    # Make a copy of the array to avoid any potential race conditions
    current_array = img_array.copy()
    
    # values, counts = np.unique(current_array, return_counts=True)
    # print(f"Serving image data: shape={current_array.shape}, values={values}, counts={counts}")
    matrix_list = current_array.tolist()    

    return jsonify({
        "image": matrix_list,
        "shape": current_array.shape,
        "timestamp": time.time()
    })

# Serve React static files
@app.route('/', defaults={'path': ''})
@app.route('/<path:path>')
def serve(path):
    if path != "" and os.path.exists(app.static_folder + '/' + path):
        return send_from_directory(app.static_folder, path)
    else:
        return send_from_directory(app.static_folder, 'index.html')

def cleanup_shared_memory():
    """Clean up shared memory when the program exits"""
    try:
        shared_mem.close()
        shared_mem.unlink()
        flag_mem.close()
        flag_mem.unlink()
        print("Shared memory cleaned up")
    except Exception as e:
        print(f"Error cleaning up shared memory: {e}")

if __name__ == '__main__':
    # Register cleanup function to be called on exit
    import atexit
    atexit.register(cleanup_shared_memory)
    
    # Start webcam capture in a separate thread
    webcam_thread = threading.Thread(target=webcam_capture_and_process, daemon=True)
    webcam_thread.start()
    
    # Start Flask server
    # Note: When in production, set debug=False
    app.run(debug=True, host='0.0.0.0', port=5000, threaded=True)