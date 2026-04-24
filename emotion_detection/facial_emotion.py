import cv2
import time
import numpy as np

FACE_CASCADE_PATH = "emotion_detection/cascades/haarcascade_frontalface_default.xml"

_face_cascade = None

def get_cascade():
    global _face_cascade
    if _face_cascade is None:
        _face_cascade = cv2.CascadeClassifier(FACE_CASCADE_PATH)
    return _face_cascade


def _estimate_emotion_from_face(frame, x, y, w, h) -> dict:
    """
    Heuristic emotion estimation from face region geometry and brightness.
    No ML model needed — works on Pi 3 with minimal CPU.
    """
    face_roi = frame[y:y+h, x:x+w]
    gray_roi = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)

    # Resize to fixed size for consistent analysis
    face_resized = cv2.resize(gray_roi, (64, 64))

    # --- Feature 1: overall brightness (dark face = sad/tired) ---
    brightness = np.mean(face_resized)

    # --- Feature 2: upper vs lower half contrast ---
    upper = face_resized[:32, :]
    lower = face_resized[32:, :]
    upper_mean = np.mean(upper)
    lower_mean = np.mean(lower)
    upper_lower_diff = float(upper_mean - lower_mean)

    # --- Feature 3: edge density (tense face = more edges) ---
    edges = cv2.Canny(face_resized, 50, 150)
    edge_density = np.sum(edges > 0) / (64 * 64) * 100

    # --- Feature 4: mouth region activity ---
    mouth_region = face_resized[40:60, 15:50]
    mouth_edges = cv2.Canny(mouth_region, 30, 100)
    mouth_activity = np.sum(mouth_edges > 0) / mouth_region.size * 100

    # --- Feature 5: brow region contrast (furrowed = stress) ---
    brow_region = face_resized[8:20, 10:54]
    brow_std = np.std(brow_region)

    # --- Map features to emotion scores (0–100) ---
    happy_score  = min(100, mouth_activity * 2.5 + max(0, lower_mean - upper_mean) * 1.5)
    stress_score = min(100, edge_density * 1.8 + brow_std * 1.2)
    sad_score    = min(100, max(0, 128 - brightness) * 0.6 + max(0, upper_lower_diff) * 0.8)
    calm_score   = max(0, 100 - stress_score * 0.6 - sad_score * 0.3)

    return {
        "happy":   round(happy_score, 1),
        "stress":  round(stress_score, 1),
        "sad":     round(sad_score, 1),
        "calm":    round(calm_score, 1),
        # kept for UI display compatibility with your app.py
        "angry":   0.0,
        "fear":    0.0,
        "disgust": 0.0,
        "surprise":0.0,
        "neutral": round(calm_score, 1),
    }


def map_emotion_scores_to_therapeutic(emotions: dict) -> str:
    stress = emotions.get("stress", 0)
    sad    = emotions.get("sad", 0)
    happy  = emotions.get("happy", 0)
    calm   = emotions.get("calm", 0)

    if stress >= 35: return "stress"
    if sad    >= 35: return "sad"
    if happy  >= 35: return "happy"
    if calm   >= 25: return "calm"

    scores = {"stress": stress, "sad": sad, "happy": happy, "calm": calm}
    return max(scores, key=scores.get)


def detect_face_emotion(debug: bool = False):
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        return ("calm", {}) if debug else "calm"

    # Warm up camera
    for _ in range(5):
        cap.read()
        time.sleep(0.03)

    ret, frame = cap.read()
    cap.release()

    if not ret:
        return ("calm", {}) if debug else "calm"

    frame = cv2.resize(frame, (320, 240))  # smaller = faster on Pi 3

    try:
        cascade = get_cascade()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        faces = cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(60, 60)
        )

        if len(faces) == 0:
            return ("calm", {}) if debug else "calm"

        # Use the largest detected face
        x, y, w, h = max(faces, key=lambda f: f[2] * f[3])

        emotions = _estimate_emotion_from_face(frame, x, y, w, h)
        final = map_emotion_scores_to_therapeutic(emotions)

        return (final, emotions) if debug else final

    except Exception:
        return ("calm", {}) if debug else "calm"