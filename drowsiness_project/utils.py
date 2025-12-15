# utils.py
import cv2
import mediapipe as mp
import numpy as np
from scipy.spatial import distance as dist

# Initialize MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
    refine_landmarks=True  # Important for iris tracking if needed later
)

# Constants for Eye Landmarks (MediaPipe indexes)
# These specific numbers map to the eyelids
LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]

def calculate_ear(eye_points, landmarks):
    """
    Calculates the Eye Aspect Ratio (EAR).
    Input: List of 6 mesh points for one eye.
    Output: A float representing 'openness'.
    """
    # Vertical distances
    A = dist.euclidean(landmarks[eye_points[1]], landmarks[eye_points[5]])
    B = dist.euclidean(landmarks[eye_points[2]], landmarks[eye_points[4]])
    # Horizontal distance
    C = dist.euclidean(landmarks[eye_points[0]], landmarks[eye_points[3]])

    # EAR Formula
    ear = (A + B) / (2.0 * C)
    return ear

def process_frame(frame):
    """
    Takes a raw frame, finds the face, and returns the EAR scores.
    """
    h, w, _ = frame.shape
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb_frame)

    # Defaults (if no face found)
    left_ear = 0.0
    right_ear = 0.0
    
    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            # Convert normalized landmarks to pixel coordinates
            mesh_points = np.array([
                np.multiply([p.x, p.y], [w, h]).astype(int) 
                for p in face_landmarks.landmark
            ])

            # Calculate EAR for both eyes
            left_ear = calculate_ear(LEFT_EYE, mesh_points)
            right_ear = calculate_ear(RIGHT_EYE, mesh_points)

    # Average the two eyes
    avg_ear = (left_ear + right_ear) / 2.0
    return avg_ear