import os
import sys
from pathlib import Path

os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
from faster_whisper import WhisperModel

def format_timestamp(seconds: float) -> str:
    ms = int((seconds - int(seconds)) * 1000)
    s = int(seconds)
    h, s = divmod(s, 3600)
    m, s = divmod(s, 60)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"

def generate_subtitles(video_dir: str = "output/test_video", max_words: int = 4):
    base_path = Path(video_dir)
    audio_path = base_path / "audio" / "full_audio.mp3"
    srt_path = base_path / "subtitles.srt"

    if not audio_path.exists():
        raise FileNotFoundError(f"No se encontró {audio_path}. Ejecuta primero 02_tts_gen.py")

    print("Cargando faster-whisper en CPU (int8)...")
    model = WhisperModel("small", device="cpu", compute_type="int8")

    print("Transcribiendo audio palabra por palabra...")
    segments, _ = model.transcribe(str(audio_path), word_timestamps=True, language="en")

    words = []
    for seg in segments:
        if seg.words:
            for w in seg.words:
                words.append({"word": w.word.strip(), "start": w.start, "end": w.end})

    srt_entries = []
    idx = 1
    for i in range(0, len(words), max_words):
        chunk = words[i:i + max_words]
        start_ts = format_timestamp(chunk[0]["start"])
        end_ts = format_timestamp(chunk[-1]["end"])
        text = " ".join(w["word"] for w in chunk)
        srt_entries.append(f"{idx}\n{start_ts} --> {end_ts}\n{text}\n")
        idx += 1

    with open(srt_path, "w", encoding="utf-8") as f:
        f.write("\n".join(srt_entries))

    print(f"Subtítulos generados con éxito en: {srt_path} ({idx - 1} bloques)")

if __name__ == "__main__":
    video_dir = sys.argv[1] if len(sys.argv) > 1 else "output/test_video"
    generate_subtitles(video_dir)
