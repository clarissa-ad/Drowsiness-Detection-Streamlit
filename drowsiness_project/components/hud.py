import cv2
import numpy as np

def draw_hud(frame, probability):
    """
    Draws the 'Iron Man' style HUD on the video frame.
    
    Args:
        frame: The raw video frame (numpy array).
        probability: Float (0.0 to 1.0). 
                     1.0 means VERY DROWSY (Eyes Closed).
                     0.0 means AWAKE (Eyes Open).
    
    Returns:
        The annotated frame.
    """
    height, width, _ = frame.shape
    
    # --- CONFIGURATION ---
    # Threshold: If probability > 0.8, we trigger the alarm visuals
    IS_DROWSY = probability > 0.5 
    
    # Colors (BGR format for OpenCV)
    # Green for Safe, Red for Danger
    COLOR_SAFE = (0, 255, 0)      # Bright Green
    COLOR_DANGER = (0, 0, 255)    # Bright Red
    color = COLOR_DANGER if IS_DROWSY else COLOR_SAFE
    
    # --- 1. THE BORDER (Full Screen Alert) ---
    # Draw a thick border around the entire screen
    thickness = 10 if IS_DROWSY else 2
    cv2.rectangle(frame, (0, 0), (width, height), color, thickness)
    
    # --- 2. THE STATUS TEXT ---
    status_text = "WARNING: DROWSY" if IS_DROWSY else "SYSTEM: ACTIVE"
    
    # Add a background box for the text so it's readable
    cv2.rectangle(frame, (10, 10), (300, 60), (0, 0, 0), -1) # Black box
    cv2.putText(frame, status_text, (20, 45), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

    # --- 3. THE DROWSINESS BAR (Visualizing the Score) ---
    # We draw a vertical bar on the right side that fills up
    bar_width = 30
    bar_height = 200
    bar_x = width - 50
    bar_y = 50
    
    # Draw the outline of the bar
    cv2.rectangle(frame, (bar_x, bar_y), 
                  (bar_x + bar_width, bar_y + bar_height), 
                  (255, 255, 255), 2)
    
    # Calculate how high the fill should be based on probability
    fill_height = int(bar_height * probability)
    
    # Draw the fill (bottom-up)
    cv2.rectangle(frame, 
                  (bar_x, bar_y + bar_height - fill_height), 
                  (bar_x + bar_width, bar_y + bar_height), 
                  color, -1)
    
    # Label for the bar
    cv2.putText(frame, "DROWSINESS", (width - 130, 40), 
                cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255), 1)
    
    # --- 4. TARGET RETICLE (Iron Man Style) ---
    # Draws a small target in the center of the screen
    center_x, center_y = width // 2, height // 2
    cv2.circle(frame, (center_x, center_y), 20, (255, 255, 255), 1)
    cv2.line(frame, (center_x - 30, center_y), (center_x + 30, center_y), (255, 255, 255), 1)
    cv2.line(frame, (center_x, center_y - 30), (center_x, center_y + 30), (255, 255, 255), 1)

    return frame