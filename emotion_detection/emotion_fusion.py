def normalize_face_emotion(face_emotion: str) -> str:
    face_emotion = face_emotion.lower()

    mapping = {
        "happy": "happy",
        "sad": "sad",
        "neutral": "calm",
        "calm": "calm",
        "angry": "stress",
        "fear": "stress",
        "disgust": "stress",
        "stress": "stress",
        "surprise": "happy"
    }

    return mapping.get(face_emotion, "calm")


def normalize_watch_emotion(watch_emotion: str) -> str:
    watch_emotion = watch_emotion.lower()

    mapping = {
        "happy": "happy",
        "sad": "sad",
        "calm": "calm",
        "neutral": "calm",
        "stress": "stress",
        "angry": "stress",
        "fear": "stress",
        "disgust": "stress"
    }

    return mapping.get(watch_emotion, "calm")


def fuse_emotions(face_emotion: str, watch_emotion: str) -> str:
    face_label = normalize_face_emotion(face_emotion)
    watch_label = normalize_watch_emotion(watch_emotion)

    # if both agree
    if face_label == watch_label:
        return face_label

    # during prototype testing, trust strong facial sadness/happiness more
    if face_label in ["sad", "happy"]:
        return face_label

    # stress only wins if face also leans stress OR face is calm/uncertain
    if watch_label == "stress" and face_label in ["calm", "stress"]:
        return "stress"

    if face_label == "stress":
        return "stress"

    return face_label