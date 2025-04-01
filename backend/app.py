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
            blurred = cv2.GaussianBlur(frame, (0, 0), 2.0)
            side_by_side[:, :width] = blurred
            
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

def process_image_simplified(image, white_threshold=120, color_variance_threshold=15, 
                  history_frames=None, history_size=10, stabilization_weight=0.7,
                  color_history=None):
    """
    Simplified version of the image processing function that creates a masked version
    and assigns remaining non-masked pixels to red, blue, or green categories
    with temporal stability in color assignments.
    
    Args:
        image: Input image (BGR format)
        white_threshold: Brightness threshold for white detection
        color_variance_threshold: Maximum allowed variance between RGB channels
        history_frames: Previous mask history for temporal stabilization
        history_size: Number of frames to keep in history (default 10)
        stabilization_weight: Weight given to historical data (higher = more stable)
        color_history: Previous color assignments for temporal stabilization
        
    Returns:
        processed: Processed image with mask and color assignments applied
        history_frames: Updated mask history
        color_history: Updated color assignment history
        color_matrix: Matrix with pixel values (0=Black, 1=Red, 2=Green, 3=Blue)
    """    
    # Initialize history if None
    if history_frames is None:
        history_frames = []
    
    # Initialize color history if None
    if color_history is None:
        color_history = None
    
    # Downsample the image to 1/16 size for efficiency
    # INTER_AREA already applies antialiasing - no need for separate blur
    eighth_size = cv2.resize(image, (image.shape[1] // 16, image.shape[0] // 16), 
                            interpolation=cv2.INTER_AREA)
    
    # Ensure the image is uint8
    eighth_size_uint8 = np.uint8(eighth_size)
    
    # Simple normalization instead of complex CLAHE
    normalized_img = cv2.normalize(eighth_size_uint8, None, 0, 255, cv2.NORM_MINMAX)
    
    # Extract channels
    b, g, r = cv2.split(normalized_img)
    
    # Calculate brightness as a weighted sum (human perception)
    brightness = 0.299 * r + 0.587 * g + 0.114 * b
    
    # Simplified masking process - combines brightness and color variance
    brightness_mask = brightness > white_threshold
    
    # Calculate max difference between any two channels
    rg_diff = np.abs(r.astype(np.int16) - g.astype(np.int16))
    rb_diff = np.abs(r.astype(np.int16) - b.astype(np.int16))
    gb_diff = np.abs(g.astype(np.int16) - b.astype(np.int16))
    
    max_diff = np.maximum(np.maximum(rg_diff, rb_diff), gb_diff)
    color_balance_mask = max_diff < color_variance_threshold
    
    # Simplified combined mask
    # Convert to HSV to check saturation
    hsv_for_mask = cv2.cvtColor(normalized_img, cv2.COLOR_BGR2HSV)
    s_for_mask = hsv_for_mask[:,:,1]
    low_saturation_mask = s_for_mask < 40
    
    # Combined mask: bright AND (balanced colors OR low saturation)
    current_mask = brightness_mask & (color_balance_mask | low_saturation_mask)
    
    # Apply spatial filtering to reduce noise
    current_mask_filtered = cv2.GaussianBlur(current_mask.astype(np.float32), (5, 5), 1.0)
    
    # Apply temporal stabilization
    if len(history_frames) > 0:
        # Use weighting of recent frames
        # Average the last 3 frames with equal weights for simplicity
        num_frames_to_use = min(3, len(history_frames))
        recent_frames = history_frames[-num_frames_to_use:]
        
        history_mask = np.zeros_like(current_mask_filtered)
        for mask, _ in recent_frames:
            history_mask += mask
        history_mask /= num_frames_to_use
        
        # Temporal smoothing with fixed weight
        stabilized_mask = stabilization_weight * history_mask + (1 - stabilization_weight) * current_mask_filtered
        
        final_mask = stabilized_mask > 0.5
    else:
        final_mask = current_mask_filtered > 0.5
    
    # Update history - store both the soft mask and binary mask
    history_frames.append((current_mask_filtered, final_mask))
    
    while len(history_frames) > history_size:
        history_frames.pop(0)
    
    # Create a copy of the image for processing
    processed = eighth_size_uint8.copy()
    
    # Apply the mask (make masked areas black)
    processed[final_mask] = [0, 0, 0]
    
    # Get non-masked pixels
    non_masked = ~final_mask
    
    # Create color assignments based on HSV values
    h_channel = hsv_for_mask[:,:,0]  # Hue
    
    # Color assignment
    # Red: hue 0-30 or 150-180
    # Green: hue 30-90
    # Blue: hue 90-150
    new_color_assignment = np.zeros(non_masked.shape, dtype=np.uint8)
    new_color_assignment[(h_channel < 30) | (h_channel >= 150)] = 0  # Red
    new_color_assignment[(h_channel >= 30) & (h_channel < 90)] = 1   # Green
    new_color_assignment[(h_channel >= 90) & (h_channel < 150)] = 2  # Blue
    
    # Color stabilization
    if color_history is not None:
        prev_non_masked, prev_colors = color_history
        
        # Only apply to pixels that were non-masked in both frames
        valid_history = non_masked & prev_non_masked
        
        # Fixed stability threshold 
        color_stability_mask = s_for_mask > 60  # Higher saturation = more confident in color
        
        # Final color assignment with stabilization
        color_assignment = np.copy(new_color_assignment)
        
        # Where we have valid history and low color confidence, keep the previous color
        should_keep_prev = valid_history & ~color_stability_mask
        color_assignment[should_keep_prev] = prev_colors[should_keep_prev]
    else:
        # For first frame, just use the new assignment
        color_assignment = new_color_assignment
    
    # Store color history for next frame
    color_history = (non_masked, color_assignment)
    
    # Combined step - apply color mapping and create color matrix simultaneously
    color_matrix = np.zeros(non_masked.shape, dtype=np.uint8)  # Default to 0 (black)
    
    # Apply color mapping to the processed image and color matrix in one step
    # Red pixels
    red_mask = non_masked & (color_assignment == 0)
    processed[red_mask] = [0, 0, 255]  # BGR for Red
    color_matrix[red_mask] = 1  # Red
    
    # Green pixels
    green_mask = non_masked & (color_assignment == 1)
    processed[green_mask] = [0, 255, 0]  # BGR for Green
    color_matrix[green_mask] = 2  # Green
    
    # Blue pixels
    blue_mask = non_masked & (color_assignment == 2)
    processed[blue_mask] = [255, 0, 0]  # BGR for Blue
    color_matrix[blue_mask] = 3  # Blue
    
    return processed, history_frames, color_history, color_matrix

def process_image(image, white_threshold=120, color_variance_threshold=15, 
                  history_frames=None, history_size=10, stabilization_weight=0.7,
                  color_history=None):
    """
    Process an image to create a masked version with improved stabilization
    and assign remaining non-masked pixels to red, blue, or green categories
    with temporal stability in color assignments.
    
    Args:
        image: Input image (BGR format)
        white_threshold: Brightness threshold for white detection
        color_variance_threshold: Maximum allowed variance between RGB channels
        history_frames: Previous mask history for temporal stabilization
        history_size: Number of frames to keep in history (default 10)
        stabilization_weight: Weight given to historical data (higher = more stable)
        color_history: Previous color assignments for temporal stabilization
        
    Returns:
        processed: Processed image with mask and color assignments applied
        history_frames: Updated mask history
        color_history: Updated color assignment history
        color_matrix: Matrix with pixel values (0=Black, 1=Red, 2=Green, 3=Blue)
    """    
    # Initialize history if None
    if history_frames is None:
        history_frames = []
    
    # Initialize color history if None
    if color_history is None:
        color_history = None
    
    # Apply Gaussian blur before downsampling
    sigma = 2.0  # Adjust based on your downsampling factor
    blurred = cv2.GaussianBlur(image, (0, 0), sigma)

    # Downsample the image to 1/16 size for efficiency
    eighth_size = cv2.resize(image, (image.shape[1] // 16, image.shape[0] // 16), 
                          interpolation=cv2.INTER_AREA)
    
    # Ensure the image is uint8
    eighth_size_uint8 = np.uint8(eighth_size)
    
    # Normalize the image using histogram equalization to reduce exposure changes
    # hsv = cv2.cvtColor(eighth_size_uint8, cv2.COLOR_BGR2HSV)
    # h, s, v = cv2.split(hsv)
    # clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    # v_eq = clahe.apply(v)
    # hsv_eq = cv2.merge([h, s, v_eq])
    # normalized_img = cv2.cvtColor(hsv_eq, cv2.COLOR_HSV2BGR)
    normalized_img = cv2.normalize(eighth_size_uint8, None, 0, 255, cv2.NORM_MINMAX)
    
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
    
    # Also consider low saturation for whites
    # Convert to HSV to check saturation
    hsv_for_mask = cv2.cvtColor(normalized_img, cv2.COLOR_BGR2HSV)
    s_for_mask = hsv_for_mask[:,:,1]
    
    # Low saturation contributes to the white mask
    # Lower saturation = more likely to be considered white
    low_saturation_soft = np.clip(1.0 - (s_for_mask.astype(float) / 40.0), 0.0, 1.0)
    
    # Combined soft mask: must be bright AND (have balanced RGB values OR low saturation)
    current_mask_soft = brightness_mask_soft * np.maximum(color_balance_soft, low_saturation_soft * 0.8)
    
    # Apply spatial filtering to reduce noise using OpenCV
    current_mask_filtered = cv2.GaussianBlur(current_mask_soft.astype(np.float32), (5, 5), 1.0)
    
    # Apply temporal stabilization using the history of masks
    if len(history_frames) > 0:
        # Calculate weighted average of historical masks with a slower exponential decay
        # More recent frames have higher weight, but with a slower falloff for better stability
        weights = np.array([np.exp(-i/3.5) for i in range(len(history_frames)-1, -1, -1)])
        weights = weights / np.sum(weights)
        
        history_mask = np.zeros_like(current_mask_filtered)
        for i, (mask, _) in enumerate(history_frames):
            history_mask += weights[i] * mask
        
        # Apply hysteresis to stabilize transitions with bias toward masking
        # If a pixel was previously masked, it needs more evidence to become unmasked
        # If a pixel was previously not masked, it needs less evidence to become masked
        hysteresis_high = 0.40  # Threshold to transition from unmasked to masked
        hysteresis_low = 0.30   # Threshold to transition from masked to unmasked
        
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
    
    # Now assign colors to non-masked areas with temporal stability
    # -------------------------------------------------
    # Get non-masked pixels
    non_masked = ~final_mask
    
    # Create color assignments based on HSV values
    hsv_img = cv2.cvtColor(normalized_img, cv2.COLOR_BGR2HSV)
    h_channel = hsv_img[:,:,0]  # Hue
    s_channel = hsv_img[:,:,1]  # Saturation
    
    # Create current frame's color assignment
    new_color_assignment = np.zeros(non_masked.shape, dtype=np.uint8)
    
    # Identify low saturation areas (near white/gray/black) and exclude them from color assignment
    # These areas get noisy hue values and shouldn't be strongly assigned to a color
    low_saturation_mask = s_channel < 40  # Low saturation threshold
    
    # Simple hue-based assignment for higher saturation pixels
    # Red: hue 0-30 or 150-180
    # Green: hue 30-90
    # Blue: hue 90-150
    new_color_assignment[(h_channel < 30) | (h_channel >= 150)] = 0  # Red
    new_color_assignment[(h_channel >= 30) & (h_channel < 90)] = 1   # Green
    new_color_assignment[(h_channel >= 90) & (h_channel < 150)] = 2  # Blue
    
    # Apply color hysteresis for stability if we have color history
    if color_history is not None:
        # Define confidence based on saturation and hue difference
        # Higher saturation = more confident in the color assignment
        # Very low saturation = very low confidence (these are near white/gray areas)
        s_confidence = np.clip(s_channel.astype(float) / 255.0, 0.2, 1.0)
        
        # Low saturation pixels get very low confidence in their hue-based assignment
        s_confidence[low_saturation_mask] *= 0.3  # Reduce confidence in low saturation areas
        
        # Calculate hue difference between current and previous frame
        # This helps detect actual color changes even when saturation is low
        hue_diff = np.zeros_like(s_confidence)
        
        if isinstance(color_history, tuple) and len(color_history) == 2:
            prev_non_masked, prev_colors = color_history
            
            # Calculate color assignment difference - different assignment = 1, same = 0
            color_diff = (new_color_assignment != prev_colors).astype(float)
            
            # Combine saturation confidence with color difference
            # If color assignment changed significantly, increase confidence in the new value
            # This makes the system more responsive to real color changes
            combined_confidence = s_confidence + (color_diff * 0.3)  # Boost confidence when color changes
            
            # Calculate the probability of keeping the previous assignment
            # Use a weighted confidence approach rather than a hard threshold
            # This creates a smoother transition when colors are changing
            keep_weight = 1.0 - combined_confidence  # Higher confidence = lower chance of keeping old color
            keep_weight = np.clip(keep_weight, 0.0, 0.8)  # Cap the maximum stickiness
            
            # Only keep colors with sufficient weight
            keep_prev_color = keep_weight > 0.6
        
        # Create stabilized color assignment
        color_assignment = np.copy(new_color_assignment)
        
        # Only keep previous colors where confidence is low and pixel was not masked
        # in both current and previous frame
        valid_history = non_masked & np.ones_like(non_masked)  # All non-masked pixels
        if isinstance(color_history, tuple) and len(color_history) == 2:
            prev_non_masked, _ = color_history
            valid_history = non_masked & prev_non_masked  # Pixels that were non-masked in both frames
        
        # Initialize prev_colors to ensure it exists
        prev_colors = np.zeros_like(color_assignment)
        if isinstance(color_history, tuple) and len(color_history) == 2:
            _, prev_colors = color_history
            
        # Apply hysteresis to color assignments
        # For each color channel, apply special handling
        for color_idx in range(3):
            # Pixels that were previously this color need stronger evidence to change
            prev_is_this_color = valid_history & (prev_colors == color_idx)
            curr_is_diff_color = new_color_assignment != color_idx
            
            # Where prev was this color, but current is different with LOW confidence,
            # keep as previous color
            should_keep_as_prev = prev_is_this_color & curr_is_diff_color & keep_prev_color
            color_assignment[should_keep_as_prev] = color_idx
        
    else:
        # For first frame, just use the new assignment
        color_assignment = new_color_assignment
    
    # Store color history for next frame (both non-masked region and color assignments)
    color_history = (non_masked, color_assignment)
    
    # Apply color mapping to the processed image - but only for non-masked areas
    # Red pixels
    processed[non_masked & (color_assignment == 0)] = [0, 0, 255]  # BGR for Red
    # Green pixels
    processed[non_masked & (color_assignment == 1)] = [0, 255, 0]  # BGR for Green
    # Blue pixels
    processed[non_masked & (color_assignment == 2)] = [255, 0, 0]  # BGR for Blue
    
    # Create color matrix with values (0=Black, 1=Red, 2=Green, 3=Blue)
    color_matrix = np.zeros(non_masked.shape, dtype=np.uint8)
    
    # Set values based on masks:
    # 0 = Black (masked areas)
    # 1 = Red (color_assignment == 0)
    # 2 = Green (color_assignment == 1)
    # 3 = Blue (color_assignment == 2)
    
    # By default, the matrix is initialized with zeros (Black)
    # Set the non-masked areas to their respective color codes
    color_matrix[non_masked & (color_assignment == 0)] = 1  # Red
    color_matrix[non_masked & (color_assignment == 1)] = 2  # Green
    color_matrix[non_masked & (color_assignment == 2)] = 3  # Blue
    
    return processed, history_frames, color_history, color_matrix


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