import streamlit as st
import cv2
import tempfile
import time
import os
from components import hud
from utils import load_cnn_model, get_eye_image_for_cnn, draw_eye_box

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

            # --- A. MODEL INFERENCE (CNN-based detection) ---
            # Load model (cached)
            model = load_cnn_model()
            if model is None:
                st.error("Model failed to load")
                break
                
            # Extract eye regions and predict drowsiness
            eye_img_processed, eye_bboxes = get_eye_image_for_cnn(frame)
            drowsiness_prob = 0.0
            
            if eye_img_processed is not None:
                prediction = model.predict(eye_img_processed, verbose=0)
                raw_prediction = float(prediction[0][0])
                # Invert: model predicts awake, we need drowsiness
                drowsiness_prob = 1.0 - raw_prediction
                
                # Draw bounding boxes on the frame
                color = (0, 0, 255) if drowsiness_prob >= 0.5 else (0, 255, 0)
                frame = draw_eye_box(frame, eye_bboxes, color=color, thickness=2)

            # --- B. VISUALS (Member 2's Design) ---
            # We pass the frame to the HUD designer to draw the futuristic overlay
            frame = hud.draw_hud(frame, drowsiness_prob)

            # --- C. DISPLAY ---
            # Convert BGR (OpenCV standard) to RGB (Streamlit standard)
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Mirror the frame horizontally (same as live camera)
            frame_rgb = cv2.flip(frame_rgb, 1)
            
            # Update the image placeholder
            st_frame.image(frame_rgb, channels="RGB", use_container_width=True)
            
            # Show the drowsiness score
            st_metrics.metric("Drowsiness Score", f"{drowsiness_prob:.2f}")

            # Slow down slightly to match normal video speed (approx 30 FPS)
            time.sleep(0.03)

        cap.release()
        # Clean up the temp file
        try:
            os.remove(tfile.name)
        except:
            pass