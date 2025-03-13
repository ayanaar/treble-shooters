import cv2
import numpy as np
from flask import Flask, Response, jsonify, send_from_directory
import json
import threading
import time
import os

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
            processed_frame, history_frames, color_history = process_image(
                frame,
                white_threshold=120,
                color_variance_threshold=15,
                history_frames=history_frames,
                history_size=5,
                stabilization_weight=0.7,
                color_history=color_history,
                color_stabilization_weight=0.9
            )
            # processed_frame, history_frames = process_image2(
            #     frame,
            #     white_threshold=11,
            #     history_frames=history_frames,
            #     history_size=5,
            #     stabilization_weight=0.1
            # )
            
            # Update the global array
            global_image_array = processed_frame
            processed_image_available = True
            
            # Get the original dimensions for display purposes
            display_width = frame.shape[1]  # Original width
            display_height = frame.shape[0]  # Original height

            # Resize the processed image to original dimensions for display
            display_frame = cv2.resize(processed_frame, (display_width, display_height), 
                          interpolation=cv2.INTER_NEAREST)  # Using NEAREST for pixelated effect

            # Display the resized processed image
            cv2.imshow('Processed Webcam Feed', display_frame)
            
            # Exit if 'q' is pressed
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
            # Brief pause to reduce CPU usage
            time.sleep(0.01)
    finally:
        # Release resources
        cap.release()
        cv2.destroyAllWindows()

def process_image(image, white_threshold=120, color_variance_threshold=15, 
                  history_frames=None, history_size=5, stabilization_weight=0.7,
                  color_history=None, color_stabilization_weight=0.9):
    """
    Process an image to create a masked version with improved stabilization
    and assign remaining non-masked pixels to red, blue, or green categories.
    
    Args:
        image: Input image (BGR format)
        white_threshold: Brightness threshold for white detection
        color_variance_threshold: Maximum allowed variance between RGB channels
        history_frames: Previous mask history for temporal stabilization
        history_size: Number of frames to keep in history
        stabilization_weight: Weight given to historical data (higher = more stable)
        color_history: Previous color assignments for temporal stabilization
        color_stabilization_weight: Weight for color assignment stability
        
    Returns:
        processed: Processed image with mask and color assignments applied
        history_frames: Updated mask history
        color_history: Updated color assignments history
    """
    # Using pre-imported cv2 and numpy
    import numpy as np
    import cv2
    
    # Initialize history if None
    if history_frames is None:
        history_frames = []
    
    # Initialize color history if None
    if color_history is None:
        color_history = None
    
    # Downsample the image to 1/16 size for efficiency
    eighth_size = cv2.resize(image, (image.shape[1] // 16, image.shape[0] // 16), 
                          interpolation=cv2.INTER_AREA)
    
    # Ensure the image is uint8
    eighth_size_uint8 = np.uint8(eighth_size)
    
    # Normalize the image using histogram equalization to reduce exposure changes
    hsv = cv2.cvtColor(eighth_size_uint8, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    v_eq = clahe.apply(v)
    hsv_eq = cv2.merge([h, s, v_eq])
    normalized_img = cv2.cvtColor(hsv_eq, cv2.COLOR_HSV2BGR)
    
    # Create a mask that checks if all RGB channels are above threshold
    r, g, b = normalized_img[:,:,0], normalized_img[:,:,1], normalized_img[:,:,2]
    
    # Calculate brightness as a weighted sum (human perception)
    brightness = 0.299 * r + 0.587 * g + 0.114 * b
    
    # Create a soft brightness mask with a transition zone
    # This helps reduce flickering by creating a gradient near the threshold
    soft_threshold = 10  # Range for soft transition
    brightness_mask_soft = np.clip((brightness - (white_threshold - soft_threshold)) / 
                                  (2 * soft_threshold), 0, 1)
    
    # Calculate max difference between any two channels
    rg_diff = np.abs(r.astype(np.int16) - g.astype(np.int16))
    rb_diff = np.abs(r.astype(np.int16) - b.astype(np.int16))
    gb_diff = np.abs(g.astype(np.int16) - b.astype(np.int16))
    
    max_diff = np.maximum(np.maximum(rg_diff, rb_diff), gb_diff)
    
    # Create a soft color balance mask
    color_balance_soft = np.clip(1 - (max_diff - color_variance_threshold + 5) / 10, 0, 1)
    
    # Combined soft mask: must be bright AND have balanced RGB values
    current_mask_soft = brightness_mask_soft * color_balance_soft
    
    # Apply spatial filtering to reduce noise using OpenCV
    current_mask_filtered = cv2.GaussianBlur(current_mask_soft.astype(np.float32), (5, 5), 1.0)
    
    # Apply temporal stabilization using the history of masks
    if len(history_frames) > 0:
        # Calculate weighted average of historical masks with exponential decay
        # More recent frames have higher weight
        weights = np.array([np.exp(-i/2) for i in range(len(history_frames)-1, -1, -1)])
        weights = weights / np.sum(weights)
        
        history_mask = np.zeros_like(current_mask_filtered)
        for i, (mask, _) in enumerate(history_frames):
            history_mask += weights[i] * mask
        
        # Apply hysteresis to stabilize transitions
        # If a pixel was previously masked, it needs more evidence to become unmasked
        # If a pixel was previously not masked, it needs more evidence to become masked
        hysteresis_high = 0.55  # Threshold to transition from unmasked to masked
        hysteresis_low = 0.45   # Threshold to transition from masked to unmasked
        
        # Get previous frame's final binary mask
        _, prev_binary = history_frames[-1]
        
        # Apply hysteresis thresholding
        current_stabilized = np.copy(current_mask_filtered)
        # Pixels that were masked need to fall below low threshold to become unmasked
        current_stabilized[prev_binary] = np.where(
            current_mask_filtered[prev_binary] < hysteresis_low,
            current_mask_filtered[prev_binary], 
            1.0
        )
        # Pixels that were not masked need to rise above high threshold to become masked
        current_stabilized[~prev_binary] = np.where(
            current_mask_filtered[~prev_binary] > hysteresis_high,
            current_mask_filtered[~prev_binary], 
            0.0
        )
        
        # Weight the current mask with the history using adaptive weighting
        # Higher weight for stable areas, lower weight for changing areas
        confidence = 1.0 - np.abs(current_mask_filtered - history_mask)
        adaptive_weight = stabilization_weight * confidence + 0.5 * (1 - confidence)
        
        stabilized_mask = (adaptive_weight * history_mask + 
                          (1 - adaptive_weight) * current_stabilized)
        
        # Apply threshold to get binary mask
        final_mask_soft = stabilized_mask
        final_mask = stabilized_mask > 0.5
    else:
        final_mask_soft = current_mask_filtered
        final_mask = current_mask_filtered > 0.5
    
    # Update history - store both the soft mask and binary mask
    # Use a tuple to store both pieces of information
    history_frames.append((current_mask_filtered, final_mask))
    
    while len(history_frames) > history_size:
        history_frames.pop(0)
    
    # Create a copy of the image for processing
    processed = eighth_size_uint8.copy()
    
    # Apply the mask (make masked areas black)
    processed[final_mask] = [0, 0, 0]
    
    # Now assign colors to non-masked areas
    # -------------------------------------------------
    # Get non-masked pixels
    non_masked = ~final_mask
    
    # Create a color assignment map based on pixel characteristics
    # We'll use HSV color space for more stable clustering
    hsv_img = cv2.cvtColor(normalized_img, cv2.COLOR_BGR2HSV)
    h_channel = hsv_img[:,:,0]  # Hue
    s_channel = hsv_img[:,:,1]  # Saturation
    
    # Create initial color assignments based on hue and saturation
    # This will assign pixels to one of three categories (0, 1, 2)
    # corresponding to red, green, blue
    
    # For consistent clustering, we'll use pixel position (x, y) and color features
    height, width = non_masked.shape
    y_coords, x_coords = np.mgrid[0:height, 0:width]
    
    # Normalize spatial coordinates to give them appropriate weight
    x_norm = x_coords / width * 30  # Scale factor can be adjusted
    y_norm = y_coords / height * 30
    
    # Create feature vectors for each pixel
    # Features: x, y, hue, saturation
    # We'll use only these features for non-masked pixels
    features = np.stack([x_norm, y_norm, h_channel.astype(float), s_channel.astype(float)], axis=-1)
    
    # Create initial color assignments using thresholds
    color_assignment = np.zeros(non_masked.shape, dtype=np.uint8)
    
    # If we have a previous color assignment, use it for stability
    if color_history is not None:
        # Apply stabilization by blending previous assignments with new ones
        # First, generate new assignments based on features
        feature_assignment = np.zeros_like(color_assignment)
        
        # Simple feature-based assignment using hue ranges
        # Red: hue 0-30 or 150-180
        # Green: hue 30-90
        # Blue: hue 90-150
        hue_ranges = [
            ((h_channel < 30) | (h_channel >= 150)),  # Red
            ((h_channel >= 30) & (h_channel < 90)),   # Green
            ((h_channel >= 90) & (h_channel < 150))   # Blue
        ]
        
        feature_assignment[hue_ranges[0]] = 0  # Red
        feature_assignment[hue_ranges[1]] = 1  # Green
        feature_assignment[hue_ranges[2]] = 2  # Blue
        
        # For stability, we'll blend previous assignment with new one
        # Masked areas should be ignored in this step
        non_masked_indices = np.where(non_masked)
        
        # Create a confidence map - how confident we are in the new assignment
        # Higher saturation = more confident
        confidence = np.clip(s_channel / 255.0, 0.1, 0.5)
        
        # Blend previous and new assignments with weighted average
        temp_assignment = np.copy(color_history)
        
        # Only update non-masked areas
        # Extract confidence values for non-masked pixels
        confidence_non_masked = confidence[non_masked_indices]
        
        # Calculate adaptive weight for each non-masked pixel
        adaptive_weight = color_stabilization_weight * (1 - confidence_non_masked)
        
        # Calculate the probability of keeping the previous assignment
        keep_prev_mask = np.random.random(len(non_masked_indices[0])) < adaptive_weight
        change_indices = np.where(~keep_prev_mask)[0]
        
        # Update only pixels that should change
        if len(change_indices) > 0:
            update_y = non_masked_indices[0][change_indices]
            update_x = non_masked_indices[1][change_indices]
            temp_assignment[update_y, update_x] = feature_assignment[update_y, update_x]
        
        color_assignment = temp_assignment
        
    else:
        # For first frame, create assignment from scratch
        # Simple clustering using hue thresholds
        color_assignment[non_masked & (h_channel < 30)] = 0  # Red
        color_assignment[non_masked & (h_channel >= 30) & (h_channel < 90)] = 1  # Green
        color_assignment[non_masked & (h_channel >= 90)] = 2  # Blue
    
    # Store color assignment for next frame
    color_history = color_assignment.copy()
    
    # Apply color mapping to the processed image
    # Red pixels
    processed[non_masked & (color_assignment == 0)] = [0, 0, 255]  # BGR for Red
    # Green pixels
    processed[non_masked & (color_assignment == 1)] = [0, 255, 0]  # BGR for Green
    # Blue pixels
    processed[non_masked & (color_assignment == 2)] = [255, 0, 0]  # BGR for Blue
    
    return processed, history_frames, color_history

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