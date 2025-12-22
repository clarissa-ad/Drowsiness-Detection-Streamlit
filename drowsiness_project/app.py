# app.py
import streamlit as st
import cv2
import numpy as np
import time

# --- IMPORT MODULES ---
# We import the functions from the files you just created
from components import hud, dashboard, playback
from utils import load_cnn_model, get_eye_image_for_cnn, draw_eye_box

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
    # --- SESSION STATE INITIALIZATION ---
    # Inisialisasi variabel untuk Consecutive Frames Counter dan Bbox Stabilization
    if 'drowsy_counter' not in st.session_state:
        st.session_state.drowsy_counter = 0
    if 'prev_bboxes' not in st.session_state:
        st.session_state.prev_bboxes = None
    if 'alarm_active' not in st.session_state:
        st.session_state.alarm_active = False
    
    # Konstanta untuk Consecutive Frames Logic
    CONSECUTIVE_FRAMES_THRESHOLD = 25  # ~1 detik pada 30 FPS (bisa disesuaikan 20-30)
    
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
        # Reset state saat kamera dimulai
        st.session_state.drowsy_counter = 0
        st.session_state.prev_bboxes = None
        st.session_state.alarm_active = False
        
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
                
                # Mirror frame di AWAL agar semua processing bekerja pada frame yang sudah mirrored
                # Ini memastikan gerakan natural DAN text tidak terbalik
                frame = cv2.flip(frame, 1)
                
                # --- MODEL INFERENCE SECTION ---
                # Ekstrak area mata dan preprocess untuk CNN (DENGAN STABILISASI)
                eye_img_processed, eye_bboxes = get_eye_image_for_cnn(
                    frame, 
                    prev_bboxes=st.session_state.prev_bboxes
                )
                drowsiness_score = 0.0
                
                if eye_img_processed is not None:
                    # Update prev_bboxes untuk frame berikutnya
                    st.session_state.prev_bboxes = eye_bboxes
                    
                    # Prediksi menggunakan model CNN (VERBOSE=0 untuk performance)
                    prediction = model.predict(eye_img_processed, verbose=0)
                    raw_score = float(prediction[0][0])
                    
                    # Invert the score: Model outputs eye openness (high=open, low=closed)
                    # We want drowsiness (high=drowsy/closed, low=awake/open)
                    drowsiness_score = 1.0 - raw_score
                    
                    # --- CONSECUTIVE FRAMES LOGIC (ANTI-BLINK) ---
                    if drowsiness_score >= threshold:
                        # Increment counter jika drowsy
                        st.session_state.drowsy_counter += 1
                    else:
                        # Reset counter jika awake
                        st.session_state.drowsy_counter = 0
                        st.session_state.alarm_active = False
                    
                    # Aktifkan alarm hanya setelah consecutive frames threshold tercapai
                    if st.session_state.drowsy_counter >= CONSECUTIVE_FRAMES_THRESHOLD:
                        st.session_state.alarm_active = True
                    
                    # Draw Bounding Box berdasarkan ALARM STATUS (bukan threshold langsung)
                    if st.session_state.alarm_active:
                        # Merah jika alarm aktif (mata tertutup berturut-turut)
                        frame = draw_eye_box(frame, eye_bboxes, color=(0, 0, 255), thickness=2)
                    else:
                        # Hijau jika awake atau belum mencapai threshold consecutive
                        frame = draw_eye_box(frame, eye_bboxes, color=(0, 255, 0), thickness=2)
                else:
                    # Jika wajah tidak terdeteksi, reset state
                    st.session_state.drowsy_counter = 0
                    st.session_state.prev_bboxes = None
                    st.session_state.alarm_active = False

                # --- VISUALIZATION SECTION (Calling Member 2 dengan alarm_active) ---
                # Pass alarm_active ke HUD untuk tampilkan peringatan visual
                frame_with_hud = hud.draw_hud(
                    frame, 
                    drowsiness_score,
                    alarm_active=st.session_state.alarm_active
                )

                # --- DISPLAY SECTION ---
                # Convert BGR (OpenCV) to RGB (Streamlit)
                frame_rgb = cv2.cvtColor(frame_with_hud, cv2.COLOR_BGR2RGB)
                
                # Frame sudah di-flip di awal, tidak perlu flip lagi
                
                # Update the Streamlit Image
                st_frame.image(frame_rgb, channels="RGB", use_container_width=True)
                
                # Update Member 3's Dashboard (Optional live metrics)
                # Tambahkan info counter untuk debugging
                st_metric.metric(
                    "Drowsiness Score", 
                    f"{drowsiness_score:.2f}",
                    delta=f"Counter: {st.session_state.drowsy_counter}/{CONSECUTIVE_FRAMES_THRESHOLD}"
                )
                
                # Check for stop button click
                if stop_placeholder.button("Stop Camera", key=f"stop_{time.time()}"):
                    break
                
                # Small delay for smoother performance (~30 FPS)
                time.sleep(0.03)
        
        # Release camera when done
        cap.release()
        cv2.destroyAllWindows()
        
        # Reset state setelah kamera berhenti
        st.session_state.drowsy_counter = 0
        st.session_state.prev_bboxes = None
        st.session_state.alarm_active = False