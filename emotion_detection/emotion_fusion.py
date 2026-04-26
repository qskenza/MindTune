def normalize_face_emotion(face_emotion: str) -> str:
    face_emotion = str(face_emotion).lower()

    mapping = {
        "happy": "happy",
        "sad": "sad",
        "neutral": "calm",
        "calm": "calm",
        "angry": "angry",
        "fear": "stress",
        "disgust": "stress",
        "stress": "stress",
        "surprise": "happy",
        "active": "active",
    }

    return mapping.get(face_emotion, "calm")


def normalize_sensor_state(sensor_state: str) -> str:
    sensor_state = str(sensor_state).lower()

    mapping = {
        "calm": "calm",
        "neutral": "neutral",
        "stress": "stress",
        "stressed": "stress",
        "active": "active",
        "angry": "angry",
        "happy": "happy",
        "sad": "sad",
    }

    return mapping.get(sensor_state, "calm")


def fuse_emotions(face_emotion: str, sensor_state: str) -> str:
    face_label = normalize_face_emotion(face_emotion)
    sensor_label = normalize_sensor_state(sensor_state)

    # Strong physiological stress overrides neutral/calm face
    if sensor_label == "stress" and face_label in ["calm", "neutral", "stress", "angry"]:
        return "stress"

    # Facial emotions that sensors cannot directly detect
    if face_label == "happy":
        return "happy"

    if face_label == "sad":
        return "sad"

    if face_label == "angry":
        if sensor_label == "stress":
            return "stress"
        return "angry"

    # Motion-based state
    if sensor_label == "active":
        return "active"

    # Both calm or face calm
    if face_label == "calm" and sensor_label in ["calm", "neutral"]:
        return "calm"

    return face_label