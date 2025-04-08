import cv2
import numpy as np
from flask import Flask, Response, jsonify, send_from_directory
import json
import threading
import time
import os

from process_image import process_image_simplified, process_image

# Global variable to store the numpy array
global_image_array = np.zeros((480, 640, 3), dtype=np.uint8)
processed_image_available = False

# Initialize Flask app
app = Flask(__name__, static_folder='../frontend/build')

# Function to capture from webcam and process images
def webcam_capture_and_process():
    global global_image_array, processed_image_available
    
    # Initialize webcam
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return
    
    history_frames = None
    color_history = None

    try:
        while True:
            # Read frame from webcam
            ret, frame = cap.read()
            
            if not ret:
                print("Failed to get frame from webcam")
                break
                
            # Process the image using our stabilized function
            processed_frame, history_frames, color_history, color_matrix = process_image_simplified(
                frame,
                white_threshold=120,
                color_variance_threshold=15,
                history_frames=history_frames,
                history_size=10,
                stabilization_weight=0.7,
                color_history=color_history
            )
            # processed_frame, history_frames = process_image2(
            #     frame,
            #     white_threshold=11,
            #     history_frames=history_frames,
            #     history_size=5,
            #     stabilization_weight=0.1
            # )
            
            # Update the global array
            global_image_array = color_matrix
            processed_image_available = True
            
            # Get the original dimensions for display purposes
            display_width = frame.shape[1]  # Original width
            display_height = frame.shape[0]  # Original height

            # Get the original dimensions
            height, width = frame.shape[:2]

            # # Resize the processed image to original dimensions for display
            # display_frame = cv2.resize(processed_frame, (2*display_width, 2*display_height), 
            #               interpolation=cv2.INTER_NEAREST)  # Using NEAREST for pixelated effect

            # # Display the resized processed image
            # cv2.imshow('Processed Webcam Feed', display_frame)
            # Resize the processed frame to match the original frame's dimensions
            # The processed frame is 1/16 the size of the original, so we need to scale it up
            enlarged_processed = cv2.resize(processed_frame, (width, height), 
                                           interpolation=cv2.INTER_NEAREST)  # Using NEAREST for pixelated effect
            
            # Create a side-by-side display
            # Create a blank canvas with twice the width to hold both images
            side_by_side = np.zeros((height, width * 2, 3), dtype=np.uint8)
            
            # Place the original frame on the left side
            side_by_side[:, :width] = frame
            
            # Place the processed frame on the right side
            side_by_side[:, width:] = enlarged_processed
            
            # Add a vertical line between the frames
            side_by_side[:, width-1:width+1] = [255, 255, 255]  # White line
            
            # Add labels to each frame
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(side_by_side, 'Original', (10, 30), font, 1, (255, 255, 255), 2)
            cv2.putText(side_by_side, 'Processed', (width + 10, 30), font, 1, (255, 255, 255), 2)
            
            # Display the side-by-side view
            cv2.imshow('Original vs Processed', side_by_side)

            # Exit if 'q' is pressed
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
            # Brief pause to reduce CPU usage
            time.sleep(0.01)
    finally:
        # Release resources
        cap.release()
        cv2.destroyAllWindows()




# Route to serve the numpy array as JSON
@app.route('/api/image-data', methods=['GET'])
def get_image_data():
    global global_image_array, processed_image_available
    
    if not processed_image_available:
        return jsonify({"error": "No processed image available yet"}), 404
    
    # Convert the numpy array to a format suitable for JSON
    # We'll convert to a base64 string for efficient transfer
    import base64
    _, buffer = cv2.imencode('.jpg', global_image_array)
    img_str = base64.b64encode(buffer).decode('utf-8')
    
    return jsonify({
        "image": img_str,
        "shape": global_image_array.shape,
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

if __name__ == '__main__':
    # Start webcam capture in a separate thread
    webcam_thread = threading.Thread(target=webcam_capture_and_process, daemon=True)
    webcam_thread.start()
    
    # Start Flask server
    app.run(debug=True, host='0.0.0.0', port=5000, threaded=True)