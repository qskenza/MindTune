from pathlib import Path
import subprocess
import time
import shutil


ROOT = Path(__file__).resolve().parent
BUNDLES_DIR = ROOT / "bundles"
OUTPUTS_DIR = ROOT / "outputs"

BUNDLE_FILE = BUNDLES_DIR / "chord_pitches_improv.mag"


EMOTION_CONFIG = {
    "stress": {
        "qpm": 68,
        "chords": ["Am", "F", "C", "G"],
        "repeats": 12,
        "primer": "[57, -2, 60, -2, 64, -2, 69, -2]",
    },
    "calm": {
        "qpm": 72,
        "chords": ["C", "G", "Am", "F"],
        "repeats": 12,
        "primer": "[60, -2, 64, -2, 67, -2, 72, -2]",
    },
    "happy": {
        "qpm": 88,
        "chords": ["C", "Em", "F", "G"],
        "repeats": 12,
        "primer": "[60, -2, 64, -2, 67, -2, 72, -2]",
    },
    "sad": {
        "qpm": 66,
        "chords": ["Am", "Dm", "F", "E"],
        "repeats": 12,
        "primer": "[57, -2, 60, -2, 62, -2, 64, -2]",
    },
    "neutral": {
        "qpm": 74,
        "chords": ["C", "Am", "F", "G"],
        "repeats": 12,
        "primer": "[60, -2, 64, -2, 67, -2, 72, -2]",
    },
}


def generate_midi(emotion: str = "calm") -> str:
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

    if not BUNDLE_FILE.exists():
        raise FileNotFoundError(
            f"Bundle not found: {BUNDLE_FILE}\n"
            "Put chord_pitches_improv.mag inside ai_music/bundles/"
        )

    emotion = emotion.lower()
    config = EMOTION_CONFIG.get(emotion, EMOTION_CONFIG["neutral"])

    backing_chords = " ".join(config["chords"] * config["repeats"])
    timestamp = int(time.time())

    before = set(OUTPUTS_DIR.glob("*.mid"))

    cmd = [
        "improv_rnn_generate",
        "--config=chord_pitches_improv",
        f"--bundle_file={BUNDLE_FILE}",
        f"--output_dir={OUTPUTS_DIR}",
        "--num_outputs=1",
        f"--primer_melody={config['primer']}",
        f"--backing_chords={backing_chords}",
        "--render_chords",
        f"--qpm={config['qpm']}",
    ]

    subprocess.run(cmd, check=True)

    after = set(OUTPUTS_DIR.glob("*.mid"))
    new_files = sorted(after - before, key=lambda p: p.stat().st_mtime)

    if not new_files:
        raise RuntimeError("No MIDI file was generated.")

    generated = new_files[-1]
    final_name = OUTPUTS_DIR / f"{emotion}_{timestamp}.mid"
    shutil.move(str(generated), str(final_name))

    return str(final_name)


if __name__ == "__main__":
    midi_path = generate_midi("calm")
    print("Generated MIDI:", midi_path)