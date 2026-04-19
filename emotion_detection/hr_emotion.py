def hr_to_emotion(hr: int | None, baseline_hr: int = 75) -> str:
    if hr is None:
        return "calm"

    if hr >= baseline_hr + 25:
        return "stress"
    elif hr >= baseline_hr + 10:
        return "calm"
    else:
        return "calm"