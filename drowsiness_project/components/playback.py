import streamlit as st
import cv2
import tempfile
import time
import os
from components import hud
from utils import process_frame

def run_video_file():
    """
    Allows the user to upload a video file and processes it frame-by-frame
    using the exact same logic as the live webcam.
    """
    st.markdown("### 📂 Video Playback Mode")
    st.info("Upload a video (MP4/AVI) to test the detection on pre-recorded footage.")

    # 1. Create the File Uploader
    uploaded_file = st.file_uploader("Choose a video...", type=["mp4", "avi", "mov"])

    if uploaded_file is not None:
        # --- TRICKY PART: OpenCV cannot read directly from Streamlit's memory ---
        # We must save the uploaded file to a temporary file on the disk first.
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(uploaded_file.read())
        
        # 2. Open the video using OpenCV
        cap = cv2.VideoCapture(tfile.name)
        
        # Create a placeholder for the video stream
        st_frame = st.empty()
        st_metrics = st.empty()
        
        stop_button = st.button("Stop Playback")

        # 3. The Loop (Processing the file)
        while cap.isOpened():
            ret, frame = cap.read()
            
            if not ret:
                st.write("End of video.")
                break
            
            if stop_button:
                break

            # --- A. LOGIC (Member 1's Math) ---
            # We call the shared utils function to get the EAR score
            ear_score = process_frame(frame)
            
            # Simple Logic: If EAR < 0.21, eyes are closed (Drowsy)
            # We convert this to a probability for the HUD (1.0 = Asleep, 0.0 = Awake)
            drowsiness_prob = 1.0 if ear_score < 0.21 else 0.0

            # --- B. VISUALS (Member 2's Design) ---
            # We pass the frame to the HUD designer to draw the futuristic overlay
            frame = hud.draw_hud(frame, drowsiness_prob)

            # --- C. DISPLAY ---
            # Convert BGR (OpenCV standard) to RGB (Streamlit standard)
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Update the image placeholder
            st_frame.image(frame_rgb, channels="RGB", use_container_width=True)
            
            # Optional: Show the raw EAR score below
            st_metrics.metric("Eye Aspect Ratio (EAR)", f"{ear_score:.2f}")

            # Slow down slightly to match normal video speed (approx 30 FPS)
            time.sleep(0.03)

        cap.release()
        # Clean up the temp file
        try:
            os.remove(tfile.name)
        except:
            pass