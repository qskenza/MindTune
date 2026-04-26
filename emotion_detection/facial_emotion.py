from deepface import DeepFace
import cv2
import time

THERAPEUTIC_EMOTIONS = ["stress", "calm", "sad", "happy"]


def map_emotion_scores_to_therapeutic(emotions: dict) -> str:
    angry   = float(emotions.get("angry", 0))
    fear    = float(emotions.get("fear", 0))
    disgust = float(emotions.get("disgust", 0))
    sad     = float(emotions.get("sad", 0))
    happy   = float(emotions.get("happy", 0))
    surprise= float(emotions.get("surprise", 0))
    neutral = float(emotions.get("neutral", 0))

    stress_score = angry + fear + disgust
    happy_score  = happy + surprise

    if stress_score >= 30: return "stress"
    if sad >= 35:          return "sad"
    if happy_score >= 35:  return "happy"
    if neutral >= 25:      return "calm"

    scores = {
        "stress": stress_score,
        "sad": sad,
        "happy": happy_score,
        "calm": neutral
    }
    return max(scores, key=scores.get)


def detect_face_emotion(debug: bool = False):
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        return ("calm", {}) if debug else "calm"

    for _ in range(5):
        cap.read()
        time.sleep(0.03)

    ret, frame = cap.read()
    cap.release()

    if not ret:
        return ("calm", {}) if debug else "calm"

    frame = cv2.resize(frame, (640, 480))

    try:
        result = DeepFace.analyze(
            img_path=frame,
            actions=["emotion"],
            enforce_detection=False,
            detector_backend="opencv",
            align=True,
            silent=True
        )

        if isinstance(result, list):
            result = result[0]

        emotions = result.get("emotion", {})
        if not emotions:
            return ("calm", {}) if debug else "calm"

        final_emotion = map_emotion_scores_to_therapeutic(emotions)

        return (final_emotion, emotions) if debug else final_emotion

    except Exception:
        return ("calm", {}) if debug else "calm"