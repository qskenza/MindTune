import sys
from pathlib import Path
import time
import asyncio
import threading

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

import streamlit as st

from emotion_detection.facial_emotion import detect_face_emotion
from emotion_detection.emotion_fusion import (
    fuse_emotions,
    normalize_face_emotion,
    normalize_sensor_state,
)
from emotion_detection.realtime_sensors import (
    start_hrm,
    arduino_reader,
    get_sensor_state,
)
from music_module import recommend_music
from audio.audio_generator import generate_audio
from logs.experiment_logger import log_experiment


st.set_page_config(page_title="MindTune", page_icon="🎵", layout="wide")


def start_background_sensors():
    arduino_thread = threading.Thread(target=arduino_reader, daemon=True)
    arduino_thread.start()

    def run_hrm_loop():
        asyncio.run(start_hrm())

    hrm_thread = threading.Thread(target=run_hrm_loop, daemon=True)
    hrm_thread.start()


if "sensors_started" not in st.session_state:
    start_background_sensors()
    st.session_state.sensors_started = True


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
    '<div class="subtitle">Emotion-adaptive therapeutic music generation using facial emotion, HRM belt, temperature, and motion sensors</div>',
    unsafe_allow_html=True
)


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
    if emotion == "angry":
        return "🟠 ANGRY", "#fff0df"
    if emotion == "active":
        return "🟣 ACTIVE", "#f1e8ff"

    return "⚪ NEUTRAL", "#f3f4f6"


def therapeutic_explanation(emotion: str, mode: str):
    emotion = emotion.lower()

    if emotion == "stress":
        return "Therapeutic Response: calming audio is generated to reduce stress and stabilize the user’s emotional state."
    if emotion == "calm":
        return "Therapeutic Response: soothing audio is generated to maintain relaxation and emotional balance."
    if emotion == "happy":
        return "Therapeutic Response: uplifting audio is generated to reinforce positive emotional well-being."
    if emotion == "sad":
        return "Therapeutic Response: comforting audio is generated to support emotional soothing and regulation."
    if emotion == "angry":
        return "Therapeutic Response: grounding audio is generated to reduce emotional intensity."
    if emotion == "active":
        return "Therapeutic Response: balanced audio is generated to support regulation after physical activation."

    return "Therapeutic Response: adaptive audio is generated based on the detected emotional state."


colA, colB = st.columns([2, 1])

with colA:
    st.markdown("### Session Setup")
    mode = st.radio(
        "Select audio mode",
        ["Instrumental", "Ambient / Noise"],
        horizontal=True,
    )

with colB:
    st.markdown("### System Status")

    sensor_data = get_sensor_state()

    if sensor_data["hr_connected"]:
        st.success("HRM belt connected")
    else:
        st.warning("HRM belt not connected yet")

    if sensor_data["arduino_connected"]:
        st.success("Arduino sensors connected")
    else:
        st.warning("Arduino sensors not connected yet")

    st.info("Facial emotion detection enabled")


st.markdown("---")
st.write("Analyze the user state and generate a therapeutic audio response.")


if st.button("Analyze Emotion", use_container_width=True):
    with st.spinner("Analyzing emotional state and generating therapeutic audio..."):
        sensor_data = get_sensor_state()

        sensor_emotion = normalize_sensor_state(sensor_data["sensor_state"])

        face_emotion, face_scores = detect_face_emotion(debug=True)
        face_emotion = normalize_face_emotion(face_emotion)

        final_emotion = fuse_emotions(face_emotion, sensor_emotion)

        recommendation = recommend_music(final_emotion, mode)

        start_time = time.time()
        audio_file = generate_audio(final_emotion, mode)
        generation_time = time.time() - start_time

        log_experiment(
            face_emotion=face_emotion,
            watch_emotion=sensor_emotion,
            final_emotion=final_emotion,
            mode=mode,
            generation_time=generation_time,
        )

    label, color = emotion_indicator(final_emotion)

    st.markdown("## Analysis Results")
    st.markdown(
        f'<div class="emotion-pill" style="background-color:{color};">{label}</div>',
        unsafe_allow_html=True,
    )

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("Physiological Sensors")
        st.write(f"**Predicted sensor state:** {sensor_emotion}")
        st.write(f"**Heart Rate:** {sensor_data['heart_rate'] if sensor_data['heart_rate'] is not None else 'No data'} bpm")
        st.write(f"**RR Mean:** {sensor_data['rr_mean']}")
        #st.write(f"**RMSSD:** {sensor_data['rmssd']}")
        st.write(f"**Temperature:** {sensor_data['temperature']} °C")
        st.write(f"**Movement Score:** {sensor_data['movement_score']}")
        st.write(f"**HRM Connected:** {sensor_data['hr_connected']}")
        st.write(f"**Arduino Connected:** {sensor_data['arduino_connected']}")
        st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("Facial Emotion")
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
        "This prototype combines webcam-based facial emotion detection with real-time physiological sensing "
        "from an HRM belt, Arduino Nano, MPU6050 motion sensor, and DS18B20 temperature sensor."
    )