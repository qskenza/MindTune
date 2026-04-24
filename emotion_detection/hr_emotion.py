def classify_physiological_state(
    hr: int | None,
    movement_score: float | None,
    temperature: float | None,
    baseline_hr: int = 75,
) -> str:
    if hr is None:
        return "calm"

    movement = movement_score or 0
    temp = temperature or 0

    if hr >= baseline_hr + 20 and movement >= 4:
        return "stress"

    if hr >= baseline_hr + 25:
        return "stress"

    if movement >= 7:
        return "active"

    if hr <= baseline_hr + 10 and movement < 3:
        return "calm"

    return "neutral"