# utils.py
import cv2
import numpy as np
import mediapipe as mp
import streamlit as st

# Use Keras directly instead of tensorflow.keras for better compatibility
try:
    from keras.models import load_model
except ImportError:
    # Fallback to tensorflow.keras if needed
    from tensorflow.keras.models import load_model

# --- KONSTANTA ---
MODEL_PATH = 'models/best_model.h5'
IMAGE_SIZE = (64, 64)

# MediaPipe Landmarks untuk mata
LEFT_EYE = [33, 133, 160, 159, 158, 157, 173, 144, 145, 153, 154, 155]
RIGHT_EYE = [362, 263, 387, 386, 385, 384, 398, 373, 374, 380, 381, 382]

# --- MODEL LOADER ---
@st.cache_resource
def load_cnn_model():
    """
    Memuat model CNN dari file .h5
    Menggunakan st.cache_resource agar model hanya dimuat sekali
    """
    try:
        model = load_model(MODEL_PATH)
        return model
    except Exception as e:
        st.error(f"Gagal memuat model dari {MODEL_PATH}: {e}")
        return None

# --- FACE MESH INITIALIZATION ---
@st.cache_resource
def get_face_mesh():
    """
    Inisialisasi MediaPipe Face Mesh
    Menggunakan st.cache_resource agar hanya diinisialisasi sekali
    """
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )
    return face_mesh

# --- PREPROCESSING UTAMA ---
def get_eye_image_for_cnn(frame):
    """
    Mendeteksi wajah dan mengekstrak area mata untuk input CNN
    
    Args:
        frame: Frame video dari webcam (BGR format)
    
    Returns:
        tuple: (eye_img_processed, bboxes) atau (None, None) jika wajah tidak ditemukan
        - eye_img_processed: Array numpy (1, 64, 64, 3) siap untuk prediksi
        - bboxes: Tuple (left_eye_bbox, right_eye_bbox) untuk kedua mata
    """
    face_mesh = get_face_mesh()
    
    # Konversi BGR ke RGB untuk MediaPipe
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(frame_rgb)
    
    if not results.multi_face_landmarks:
        return None, None
    
    # Ambil landmarks wajah pertama
    face_landmarks = results.multi_face_landmarks[0]
    h, w, _ = frame.shape
    
    # Ekstrak koordinat mata kiri dan kanan secara terpisah
    left_eye_points = []
    right_eye_points = []
    
    for idx in LEFT_EYE:
        landmark = face_landmarks.landmark[idx]
        x = int(landmark.x * w)
        y = int(landmark.y * h)
        left_eye_points.append((x, y))
    
    for idx in RIGHT_EYE:
        landmark = face_landmarks.landmark[idx]
        x = int(landmark.x * w)
        y = int(landmark.y * h)
        right_eye_points.append((x, y))
    
    # Hitung bounding box untuk mata kiri
    left_eye_points = np.array(left_eye_points)
    left_x_min, left_y_min = left_eye_points.min(axis=0)
    left_x_max, left_y_max = left_eye_points.max(axis=0)
    
    # Hitung bounding box untuk mata kanan
    right_eye_points = np.array(right_eye_points)
    right_x_min, right_y_min = right_eye_points.min(axis=0)
    right_x_max, right_y_max = right_eye_points.max(axis=0)
    
    # Tambahkan padding lebih besar untuk area mata yang lebih luas
    padding = 30  # Increased from 20 to 30
    
    # Left eye bbox dengan padding
    left_x_min = max(0, left_x_min - padding)
    left_y_min = max(0, left_y_min - padding)
    left_x_max = min(w, left_x_max + padding)
    left_y_max = min(h, left_y_max + padding)
    
    # Right eye bbox dengan padding
    right_x_min = max(0, right_x_min - padding)
    right_y_min = max(0, right_y_min - padding)
    right_x_max = min(w, right_x_max + padding)
    right_y_max = min(h, right_y_max + padding)
    
    # Crop area kedua mata untuk model (gabungan)
    all_x_min = min(left_x_min, right_x_min)
    all_y_min = min(left_y_min, right_y_min)
    all_x_max = max(left_x_max, right_x_max)
    all_y_max = max(left_y_max, right_y_max)
    
    eye_region = frame[all_y_min:all_y_max, all_x_min:all_x_max]
    
    if eye_region.size == 0:
        return None, None
    
    # Resize ke ukuran input model
    eye_region_resized = cv2.resize(eye_region, IMAGE_SIZE)
    
    # Normalisasi
    eye_img_processed = eye_region_resized.astype('float32') / 255.0
    
    # Tambahkan dimensi batch
    eye_img_processed = np.expand_dims(eye_img_processed, axis=0)
    
    # Return bounding boxes untuk kedua mata
    left_eye_bbox = (left_x_min, left_y_min, left_x_max - left_x_min, left_y_max - left_y_min)
    right_eye_bbox = (right_x_min, right_y_min, right_x_max - right_x_min, right_y_max - right_y_min)
    
    return eye_img_processed, (left_eye_bbox, right_eye_bbox)

# --- VISUALIZATION HELPER ---
def draw_eye_boxes(frame, bboxes, color=(0, 255, 0), thickness=2):
    """
    Menggambar kotak pembatas di sekitar area mata kiri dan kanan
    
    Args:
        frame: Frame video
        bboxes: Tuple (left_eye_bbox, right_eye_bbox) atau None
        color: Warna kotak (BGR format)
        thickness: Ketebalan garis
    
    Returns:
        frame: Frame dengan kotak pembatas
    """
    if bboxes is None:
        return frame
    
    left_eye_bbox, right_eye_bbox = bboxes
    
    # Draw left eye box
    if left_eye_bbox is not None:
        x, y, w, h = left_eye_bbox
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, thickness)
    
    # Draw right eye box
    if right_eye_bbox is not None:
        x, y, w, h = right_eye_bbox
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, thickness)
    
    return frame