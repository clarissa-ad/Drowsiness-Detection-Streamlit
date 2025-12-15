# app.py
import streamlit as st
import cv2
import numpy as np
import time

# --- IMPORT MODULES ---
# We import the functions from the files you just created
from components import hud, dashboard, playback
# from utils import load_model, preprocess (Uncomment this when you have utils ready)

# --- PAGE CONFIG ---
st.set_page_config(
    page_title="Drowsiness Detection System",
    layout="wide"
)

st.title("👁️ Drowsiness Detection System")

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

    # 2. Start the Webcam
    cap = cv2.VideoCapture(0) # 0 is usually the default camera

    if not cap.isOpened():
        st.error("Could not access the webcam. Check your permissions.")

    start_button = st.button("Start Camera")
    stop_button = st.button("Stop Camera")

    if start_button:
        while True: # The Infinite Loop
            ret, frame = cap.read()
            if not ret:
                st.error("Failed to capture frame")
                break
            
            # --- MODEL INFERENCE SECTION ---
            # (Later, you will put your actual model code here)
            # frame_processed = preprocess(frame)
            # prediction = model.predict(frame_processed)
            dummy_probability = np.random.uniform(0, 1) # Fake prediction for now

            # --- VISUALIZATION SECTION (Calling Member 2) ---
            # We pass the raw frame and the prediction to the HUD
            frame_with_hud = hud.draw_hud(frame, dummy_probability)

            # --- DISPLAY SECTION ---
            # Convert BGR (OpenCV) to RGB (Streamlit)
            frame_rgb = cv2.cvtColor(frame_with_hud, cv2.COLOR_BGR2RGB)
            
            # Update the Streamlit Image
            st_frame.image(frame_rgb, channels="RGB", use_container_width=True)
            
            # Update Member 3's Dashboard (Optional live metrics)
            st_metric.metric("Drowsiness Score", f"{dummy_probability:.2f}")

            # Stop logic
            # (Streamlit reruns the script on interaction, so 'stop' buttons 
            # inside loops are tricky. Usually we rely on the generic 'stop' button 
            # or use session state, but for a simple demo, this suffices).
            
    cap.release()