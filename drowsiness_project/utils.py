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

# --- BOUNDING BOX STABILIZATION ---
def stabilize_bbox(new_bbox, prev_bbox, alpha=0.7):
    """
    Stabilkan bounding box dengan Exponential Moving Average (EMA)
    untuk mengurangi jitter/flicker pada deteksi real-time
    
    Args:
        new_bbox: Tuple (x, y, w, h) - koordinat bbox terbaru
        prev_bbox: Tuple (x, y, w, h) - koordinat bbox sebelumnya (None jika pertama kali)
        alpha: Float (0.0-1.0) - smoothing factor (lebih tinggi = lebih responsif)
    
    Returns:
        Tuple (x, y, w, h) - koordinat bbox yang sudah distabilkan
    """
    if prev_bbox is None:
        return new_bbox
    
    # EMA: stabilized = alpha * new + (1-alpha) * prev
    stabilized = tuple(
        int(alpha * n + (1 - alpha) * p) 
        for n, p in zip(new_bbox, prev_bbox)
    )
    return stabilized

# --- PREPROCESSING UTAMA ---
def get_eye_image_for_cnn(frame, prev_bboxes=None):
    """
    Mendeteksi wajah dan mengekstrak area mata untuk input CNN
    DENGAN stabilisasi bounding box untuk mengurangi jitter
    
    Args:
        frame: Frame video dari webcam (BGR format)
        prev_bboxes: List dari bbox sebelumnya [(left_bbox, right_bbox)] atau None
    
    Returns:
        tuple: (eye_img_processed, eye_bboxes) atau (None, None) jika wajah tidak ditemukan
        - eye_img_processed: Array numpy (1, 64, 64, 3) siap untuk prediksi
        - eye_bboxes: List berisi dua tuple [(left_x, left_y, left_w, left_h), (right_x, right_y, right_w, right_h)]
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
    all_eye_points = []
    
    for idx in LEFT_EYE:
        landmark = face_landmarks.landmark[idx]
        x = int(landmark.x * w)
        y = int(landmark.y * h)
        left_eye_points.append((x, y))
        all_eye_points.append((x, y))
    
    for idx in RIGHT_EYE:
        landmark = face_landmarks.landmark[idx]
        x = int(landmark.x * w)
        y = int(landmark.y * h)
        right_eye_points.append((x, y))
        all_eye_points.append((x, y))
    
    # Hitung bounding box untuk SETIAP mata (untuk visualisasi)
    left_eye_points = np.array(left_eye_points)
    right_eye_points = np.array(right_eye_points)
    
    padding_individual = 10
    
    # Left eye bbox (RAW)
    left_x_min, left_y_min = left_eye_points.min(axis=0)
    left_x_max, left_y_max = left_eye_points.max(axis=0)
    left_x_min = max(0, left_x_min - padding_individual)
    left_y_min = max(0, left_y_min - padding_individual)
    left_x_max = min(w, left_x_max + padding_individual)
    left_y_max = min(h, left_y_max + padding_individual)
    left_bbox_raw = (left_x_min, left_y_min, left_x_max - left_x_min, left_y_max - left_y_min)
    
    # Right eye bbox (RAW)
    right_x_min, right_y_min = right_eye_points.min(axis=0)
    right_x_max, right_y_max = right_eye_points.max(axis=0)
    right_x_min = max(0, right_x_min - padding_individual)
    right_y_min = max(0, right_y_min - padding_individual)
    right_x_max = min(w, right_x_max + padding_individual)
    right_y_max = min(h, right_y_max + padding_individual)
    right_bbox_raw = (right_x_min, right_y_min, right_x_max - right_x_min, right_y_max - right_y_min)
    
    # STABILISASI BBOX dengan EMA
    if prev_bboxes is not None and len(prev_bboxes) == 2:
        left_bbox = stabilize_bbox(left_bbox_raw, prev_bboxes[0], alpha=0.7)
        right_bbox = stabilize_bbox(right_bbox_raw, prev_bboxes[1], alpha=0.7)
    else:
        left_bbox = left_bbox_raw
        right_bbox = right_bbox_raw
    
    # Hitung bounding box GABUNGAN untuk CNN input
    all_eye_points = np.array(all_eye_points)
    x_min, y_min = all_eye_points.min(axis=0)
    x_max, y_max = all_eye_points.max(axis=0)
    
    # Tambahkan padding untuk combined box
    padding = 20
    x_min = max(0, x_min - padding)
    y_min = max(0, y_min - padding)
    x_max = min(w, x_max + padding)
    y_max = min(h, y_max + padding)
    
    # Crop area mata (TETAP menggunakan combined region untuk CNN)
    eye_region = frame[y_min:y_max, x_min:x_max]
    
    if eye_region.size == 0:
        return None, None
    
    # Resize ke ukuran input model (64x64)
    eye_region_resized = cv2.resize(eye_region, IMAGE_SIZE)
    
    # NORMALISASI KETAT: Identik dengan training phase
    # 1. Convert ke float32
    # 2. Bagi dengan 255.0 untuk range [0.0, 1.0]
    eye_img_processed = eye_region_resized.astype('float32') / 255.0
    
    # Tambahkan dimensi batch: (1, 64, 64, 3)
    eye_img_processed = np.expand_dims(eye_img_processed, axis=0)
    
    # Return processed image dan DUA bounding boxes terpisah (STABILIZED)
    eye_bboxes = [left_bbox, right_bbox]
    
    return eye_img_processed, eye_bboxes

# --- VISUALIZATION HELPER ---
def draw_eye_box(frame, bboxes, color=(0, 255, 0), thickness=2):
    """
    Menggambar kotak pembatas di sekitar area mata
    
    Args:
        frame: Frame video
        bboxes: List of tuples [(x, y, w, h), ...] atau single tuple (x, y, w, h)
        color: Warna kotak (BGR format)
        thickness: Ketebalan garis
    
    Returns:
        frame: Frame dengan kotak pembatas
    """
    if bboxes is None:
        return frame
    
    # Handle both single bbox and list of bboxes
    if isinstance(bboxes, list):
        for bbox in bboxes:
            if bbox is not None:
                x, y, w, h = bbox
                cv2.rectangle(frame, (x, y), (x + w, y + h), color, thickness)
    else:
        # Single bbox (backward compatibility)
        x, y, w, h = bboxes
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, thickness)
    
    return frame