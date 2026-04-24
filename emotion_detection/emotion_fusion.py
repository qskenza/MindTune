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
    }

    return mapping.get(face_emotion, "calm")


def normalize_sensor_state(sensor_state: str) -> str:
    sensor_state = str(sensor_state).lower()

    mapping = {
        "calm": "calm",
        "stress": "stress",
        "stressed": "stress",
        "active": "active",
        "neutral": "calm",
    }

    return mapping.get(sensor_state, "calm")


def fuse_emotions(face_emotion: str, sensor_state: str) -> str:
    face_label = normalize_face_emotion(face_emotion)
    sensor_label = normalize_sensor_state(sensor_state)

    if sensor_label == "stress" and face_label in ["angry", "fear", "stress", "calm"]:
        return "stress"

    if face_label == "angry":
        return "angry"

    if face_label == "sad":
        return "sad"

    if face_label == "happy":
        return "happy"

    if sensor_label == "active":
        return "active"

    return face_label