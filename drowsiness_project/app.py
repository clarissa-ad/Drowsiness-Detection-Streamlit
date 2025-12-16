# app.py
import streamlit as st
import cv2
import numpy as np
import time

# --- IMPORT MODULES ---
# We import the functions from the files you just created
from components import hud, dashboard, playback
from utils import load_cnn_model, get_eye_image_for_cnn, draw_eye_boxes

# --- PAGE CONFIG ---
st.set_page_config(
    page_title="Drowsiness Detection System",
    layout="wide"
)

st.title("👁️ Drowsiness Detection System")

# --- GLOBAL MODEL LOADING ---
model = load_cnn_model()
if model is None:
    st.stop()

# --- SIDEBAR (Calling Member 3's Code) ---
threshold = dashboard.render_sidebar()

# --- MAIN SELECTION ---
mode = st.sidebar.radio("Select Mode", ["Live Webcam", "Video File"])

if mode == "Video File":
    # Call Member 4's Code
    playback.run_video_file()

elif mode == "Live Webcam":
    # --- THE LIVE LOOP (Your Responsibility) ---
    
    # 1. Create placeholders for the video and the metrics
    #    This allows us to update the image without reloading the whole page
    col1, col2 = st.columns([3, 1])
    with col1:
        st_frame = st.empty() # The video feed
    with col2:
        st_metric = st.empty() # The live graph/stats

    # 2. Button controls
    start_button = st.button("Start Camera")
    stop_button = st.button("Stop Camera")

    # 3. Camera Loop - runs continuously when Start is clicked
    if start_button:
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            st.error("Could not access the webcam. Check your permissions.")
        else:
            # Create a stop placeholder
            stop_placeholder = st.empty()
            
            # Continuous loop until manually stopped
            while True:
                ret, frame = cap.read()
                
                if not ret:
                    st.error("Failed to capture frame")
                    break
                
                # --- MODEL INFERENCE SECTION ---
                # Ekstrak area mata dan preprocess untuk CNN
                eye_img_processed, eye_bboxes = get_eye_image_for_cnn(frame)
                drowsiness_score = 0.0
                
                if eye_img_processed is not None:
                    # Prediksi menggunakan model CNN
                    prediction = model.predict(eye_img_processed, verbose=0)
                    raw_prediction = float(prediction[0][0])
                    
                    # INVERT: Model predicts "awake" (1=awake, 0=drowsy)
                    # We need drowsiness score (1=drowsy, 0=awake)
                    drowsiness_score = 1.0 - raw_prediction
                    
                    # Draw Bounding Boxes untuk kedua mata berdasarkan threshold
                    if drowsiness_score >= threshold:
                        # Merah jika drowsy
                        frame = draw_eye_boxes(frame, eye_bboxes, color=(0, 0, 255), thickness=2)
                    else:
                        # Hijau jika awake
                        frame = draw_eye_boxes(frame, eye_bboxes, color=(0, 255, 0), thickness=2)

                # --- VISUALIZATION SECTION (Calling Member 2) ---
                # We pass the frame with bounding box and the prediction to the HUD
                frame_with_hud = hud.draw_hud(frame, drowsiness_score)

                # --- DISPLAY SECTION ---
                # Convert BGR (OpenCV) to RGB (Streamlit)
                frame_rgb = cv2.cvtColor(frame_with_hud, cv2.COLOR_BGR2RGB)
                
                # Update the Streamlit Image
                st_frame.image(frame_rgb, channels="RGB", use_container_width=True)
                
                # Update Member 3's Dashboard (Optional live metrics)
                st_metric.metric("Drowsiness Score", f"{drowsiness_score:.2f}")
                
                # Check for stop button click
                if stop_placeholder.button("Stop Camera", key=f"stop_{time.time()}"):
                    break
                
                # Small delay for smoother performance
                time.sleep(0.03)
        
        # Release camera when done
        cap.release()
        cv2.destroyAllWindows()