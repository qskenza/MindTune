from pathlib import Path
import numpy as np
from scipy.io.wavfile import write
from ai_music.generate_musicgen import generate_musicgen

SAMPLE_RATE = 44100
DURATION = 12


def normalize(signal):
    peak = np.max(np.abs(signal))
    if peak == 0:
        return signal
    return signal / peak


def fade_in_out(signal, fade_in=1.5, fade_out=2.0, sample_rate=SAMPLE_RATE):
    n = len(signal)
    env = np.ones(n)

    fade_in_samples = min(int(fade_in * sample_rate), n)
    fade_out_samples = min(int(fade_out * sample_rate), n)

    if fade_in_samples > 0:
        env[:fade_in_samples] = np.linspace(0, 1, fade_in_samples)

    if fade_out_samples > 0:
        env[-fade_out_samples:] = np.linspace(1, 0, fade_out_samples)

    return signal * env


def white_noise(duration, amplitude=0.05):
    n = int(duration * SAMPLE_RATE)
    return np.random.normal(0, amplitude, n)


def pink_noise(duration, amplitude=0.05):
    n = int(duration * SAMPLE_RATE)
    white = np.random.normal(0, 1, n)
    pink = np.cumsum(white)
    return normalize(pink) * amplitude


def brown_noise(duration, amplitude=0.05):
    n = int(duration * SAMPLE_RATE)
    white = np.random.normal(0, 1, n)
    brown = np.cumsum(white)
    return normalize(brown) * amplitude


def generate_noise_audio(emotion, output_path="audio/generated_audio.wav"):
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    if emotion == "stress":
        signal = brown_noise(DURATION, amplitude=0.07)
    elif emotion == "happy":
        signal = white_noise(DURATION, amplitude=0.035)
    elif emotion == "sad":
        signal = pink_noise(DURATION, amplitude=0.045)
    else:
        signal = pink_noise(DURATION, amplitude=0.04)

    signal = fade_in_out(normalize(signal), fade_in=1.5, fade_out=2.0)
    audio = np.int16(signal * 32767 * 0.8)
    write(output_path, SAMPLE_RATE, audio)

    return output_path


def generate_audio(emotion, mode="Instrumental"):
    mode = mode.lower()

    if mode == "instrumental":
        return generate_musicgen(emotion, duration_seconds=12)
    else:
        return generate_noise_audio(emotion, output_path="audio/generated_audio.wav")