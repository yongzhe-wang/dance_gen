import os
import subprocess
from pathlib import Path
import uuid

EDGE_DIR = Path(__file__).resolve().parent.parent / "edge"  # points to /edge

def download_audio(youtube_url: str, output_dir: Path) -> Path:
    """Downloads audio from YouTube to a given output directory"""
    output_dir.mkdir(parents=True, exist_ok=True)
    audio_out = output_dir / "%(id)s.%(ext)s"

    subprocess.run([
        "yt-dlp",
        "--extract-audio",
        "--audio-format", "wav",
        "--audio-quality", "0",
        "--output", str(audio_out),
        youtube_url
    ], check=True)

    for file in output_dir.iterdir():
        if file.suffix == ".wav":
            return file
    raise FileNotFoundError("Audio download failed.")

def run_edge_inference(music_dir: Path, motion_dir: Path) -> Path:
    """Runs EDGE test.py and returns the generated .pkl path"""
    motion_dir.mkdir(parents=True, exist_ok=True)

    subprocess.run([
        "python", str(EDGE_DIR / "test.py"),  # FIXED HERE
        "--music_dir", str(music_dir),
        "--save_motions",
        "--motion_save_dir", str(motion_dir)
    ], check=True)

    pkl_files = list(motion_dir.glob("*.pkl"))
    if not pkl_files:
        raise FileNotFoundError("No .pkl output found")
    return pkl_files[0]


def generate_dance_from_youtube(youtube_url: str) -> str:
    """
    Main pipeline:
    - Download YouTube audio
    - Generate motion using EDGE
    - Return path to .pkl
    """
    run_id = str(uuid.uuid4())[:8]
    base_dir = EDGE_DIR / "tmp" / run_id
    music_dir = base_dir / "music"
    motion_dir = base_dir / "motion"

    audio_path = download_audio(youtube_url, music_dir)
    print(f"[INFO] Downloaded: {audio_path}")

    pkl_path = run_edge_inference(music_dir, motion_dir)
    print(f"[INFO] Generated motion: {pkl_path}")

    return str(pkl_path)
