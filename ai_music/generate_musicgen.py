from pathlib import Path
import torch
import soundfile as sf
from transformers import AutoProcessor, MusicgenForConditionalGeneration

MODEL_NAME = "facebook/musicgen-small"
OUTPUT_DIR = Path("ai_music/outputs")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

EMOTION_PROMPTS = {
    "stress": (
        "very soothing instrumental ambient therapy music, soft piano, warm pads, "
        "slow tempo, relaxing, peaceful, no drums"
    ),
    "calm": (
        "peaceful therapeutic instrumental music, soft piano, airy pads, meditative, "
        "gentle flowing melody, no percussion"
    ),
    "happy": (
        "light uplifting instrumental wellness music, warm piano, soft synths, bright gentle mood, "
        "positive melody, no aggressive drums"
    ),
    "sad": (
        "comforting emotional instrumental therapy music, soft piano, warm ambient texture, "
        "gentle healing melody, no percussion"
    ),
    "neutral": (
        "soft ambient instrumental background music, calm and smooth, gentle therapeutic mood, "
        "light piano and pads, no percussion"
    ),
}

_processor = None
_model = None


def get_musicgen():
    global _processor, _model

    if _processor is None or _model is None:
        _processor = AutoProcessor.from_pretrained(MODEL_NAME)
        _model = MusicgenForConditionalGeneration.from_pretrained(MODEL_NAME).to(DEVICE)

    return _processor, _model


def build_prompt(emotion: str) -> str:
    return EMOTION_PROMPTS.get(emotion.lower(), EMOTION_PROMPTS["neutral"])


def generate_musicgen(emotion: str = "calm", duration_seconds: int = 12) -> str:
    processor, model = get_musicgen()
    prompt = build_prompt(emotion)

    inputs = processor(
        text=[prompt],
        padding=True,
        return_tensors="pt"
    )

    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}

    sampling_rate = model.config.audio_encoder.sampling_rate

    # Much smaller generation size for speed
    max_new_tokens = max(128, int(duration_seconds * 50))

    with torch.no_grad():
        audio_values = model.generate(
            **inputs,
            do_sample=True,
            guidance_scale=3,
            max_new_tokens=max_new_tokens
        )

    audio = audio_values[0, 0].detach().cpu().numpy()

    out_path = OUTPUT_DIR / f"{emotion.lower()}_musicgen.wav"
    sf.write(out_path, audio, sampling_rate)

    return str(out_path)


if __name__ == "__main__":
    path = generate_musicgen("calm", duration_seconds=12)
    print("Generated:", path)