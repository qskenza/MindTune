import sys
from pathlib import Path
import time

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

import streamlit as st
import joblib

from emotion_detection.facial_emotion import detect_face_emotion
from emotion_detection.emotion_fusion import fuse_emotions, normalize_face_emotion, normalize_watch_emotion
from music_module import recommend_music
from audio.audio_generator import generate_audio
from wearable.watch_data import get_watch_data
from logs.experiment_logger import log_experiment

st.set_page_config(page_title="MindTune", page_icon="🎵", layout="wide")

MODEL_PATH = ROOT / "models" / "classical" / "rf_model.pkl"
model = joblib.load(MODEL_PATH)

st.markdown("""
    <style>
    .main-title {
        font-size: 2.5rem;
        font-weight: 700;
        margin-bottom: 0.2rem;
    }
    .subtitle {
        font-size: 1rem;
        color: #6b7280;
        margin-bottom: 1.5rem;
    }
    .card {
        padding: 1rem 1.2rem;
        border-radius: 16px;
        background-color: #f8f9fc;
        border: 1px solid #e6e8ef;
        margin-bottom: 1rem;
    }
    .emotion-pill {
        font-size: 1.2rem;
        font-weight: 700;
        padding: 0.8rem 1rem;
        border-radius: 14px;
        text-align: center;
        margin-top: 0.5rem;
        margin-bottom: 0.8rem;
    }
    </style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-title">🎵 MindTune</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="subtitle">Emotion-adaptive therapeutic music generation for stress, anxiety, and mood regulation</div>',
    unsafe_allow_html=True
)

colA, colB = st.columns([2, 1])

with colA:
    st.markdown("### Session Setup")
    mode = st.radio(
        "Select audio mode",
        ["Instrumental", "Ambient / Noise"],
        horizontal=True
    )

with colB:
    st.markdown("### System Status")
    st.success("Model loaded")
    st.info("Physiological data currently simulated")
    st.info("MusicGen enabled for Instrumental mode")

st.markdown("---")
st.write("Analyze the user state and generate a therapeutic audio response.")

def emotion_indicator(emotion: str):
    emotion = emotion.lower()
    if emotion == "stress":
        return "🔴 STRESS", "#ffe5e5"
    if emotion == "calm":
        return "🟢 CALM", "#e8f7e8"
    if emotion == "happy":
        return "🟡 HAPPY", "#fff8d9"
    if emotion == "sad":
        return "🔵 SAD", "#e8f0ff"
    return "⚪ UNKNOWN", "#f3f4f6"

def therapeutic_explanation(emotion: str, mode: str):
    emotion = emotion.lower()

    if emotion == "stress":
        return "Therapeutic Response: calming audio is generated to reduce stress and help stabilize the user’s emotional state."
    if emotion == "calm":
        return "Therapeutic Response: soothing audio is generated to maintain relaxation and emotional balance."
    if emotion == "happy":
        return "Therapeutic Response: uplifting audio is generated to reinforce positive emotional well-being."
    if emotion == "sad":
        return "Therapeutic Response: comforting audio is generated to support emotional soothing and regulation."
    return "Therapeutic Response: adaptive audio is generated based on the detected emotional state."

if st.button("Analyze Emotion", use_container_width=True):
    with st.spinner("Analyzing emotional state and generating therapeutic audio..."):
        ecg, eda, temp, resp = get_watch_data(simulated=True)

        watch_raw = model.predict([[ecg, eda, temp, resp]])[0]
        watch_emotion = normalize_watch_emotion(str(watch_raw))

        face_emotion, face_scores = detect_face_emotion(debug=True)
        face_emotion = normalize_face_emotion(face_emotion)

        final_emotion = fuse_emotions(face_emotion, watch_emotion)

        recommendation = recommend_music(final_emotion, mode)

        start_time = time.time()
        audio_file = generate_audio(final_emotion, mode)
        generation_time = time.time() - start_time

        log_experiment(
            face_emotion=face_emotion,
            watch_emotion=watch_emotion,
            final_emotion=final_emotion,
            mode=mode,
            generation_time=generation_time
        )

    label, color = emotion_indicator(final_emotion)

    st.markdown("## Analysis Results")
    st.markdown(
        f'<div class="emotion-pill" style="background-color:{color};">{label}</div>',
        unsafe_allow_html=True
    )

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("Physiological")
        st.write(f"**Predicted emotion:** {watch_emotion}")
        st.write(f"**ECG:** {ecg:.3f}")
        st.write(f"**EDA:** {eda:.3f}")
        st.write(f"**Temperature:** {temp:.2f}")
        st.write(f"**Respiration:** {resp:.3f}")
        st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("Facial")
        st.write(f"**Detected emotion:** {face_emotion}")

        if face_scores:
            st.write("**Emotion scores:**")
            for key in ["angry", "fear", "disgust", "sad", "happy", "surprise", "neutral"]:
                if key in face_scores:
                    st.write(f"- {key}: {face_scores[key]:.1f}%")

        st.markdown('</div>', unsafe_allow_html=True)

    with col3:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("Final Output")
        st.write(f"**Final therapeutic emotion:** {final_emotion}")
        st.write(f"**Audio mode:** {mode}")
        st.write(f"**Generation time:** {generation_time:.2f} s")
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("## Therapeutic Recommendation")
    st.success(recommendation)

    st.info(therapeutic_explanation(final_emotion, mode))

    st.markdown("## Generated Audio")
    st.audio(audio_file, format="audio/wav")

    st.caption(
        "The current prototype uses webcam-based facial emotion detection and simulated physiological values. "
        "The architecture is prepared for future wearable data integration."
    )