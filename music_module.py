def recommend_music(emotion, mode="Instrumental"):
    mode = mode.lower()

    if mode == "instrumental":
        if emotion == "stress":
            return "Slow ambient instrumental"
        elif emotion == "calm":
            return "Soft therapeutic pads"
        elif emotion == "happy":
            return "Bright uplifting instrumental"
        else:
            return "Neutral ambient instrumental"

    if mode == "ambient / noise":
        if emotion == "stress":
            return "Soft calming noise texture"
        elif emotion == "calm":
            return "Gentle ambient airflow"
        elif emotion == "happy":
            return "Light airy ambience"
        else:
            return "Neutral background ambience"

    return "Therapeutic audio"
