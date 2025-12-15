# components/hud.py
import cv2

def draw_hud(frame, probability):
    # DUMMY FUNCTION: Just puts text so we know it works
    cv2.putText(frame, "HUD LOADED", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    return frame