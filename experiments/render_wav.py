from pathlib import Path
import subprocess


ROOT = Path(__file__).resolve().parent
SOUNDFONTS_DIR = ROOT / "soundfonts"
OUTPUTS_DIR = ROOT / "outputs"


def find_soundfont() -> Path:
    sf2_files = list(SOUNDFONTS_DIR.glob("*.sf2"))
    if not sf2_files:
        raise FileNotFoundError(
            "No .sf2 file found in ai_music/soundfonts/"
        )
    return sf2_files[0]


def render_wav(midi_path: str) -> str:
    midi_file = Path(midi_path)
    if not midi_file.exists():
        raise FileNotFoundError(f"MIDI file not found: {midi_file}")

    soundfont = find_soundfont()
    wav_path = OUTPUTS_DIR / f"{midi_file.stem}.wav"

    cmd = [
        "fluidsynth",
        "-ni",
        str(soundfont),
        str(midi_file),
        "-F",
        str(wav_path),
        "-r",
        "44100",
    ]

    subprocess.run(cmd, check=True)

    return str(wav_path)


if __name__ == "__main__":
    example_midis = sorted(OUTPUTS_DIR.glob("*.mid"), key=lambda p: p.stat().st_mtime)
    if not example_midis:
        raise RuntimeError("No MIDI files found in ai_music/outputs/")
    wav = render_wav(str(example_midis[-1]))
    print("Rendered WAV:", wav)