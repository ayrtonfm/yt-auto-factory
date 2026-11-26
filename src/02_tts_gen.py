import asyncio
import json
import subprocess
import sys
import yaml
from pathlib import Path
import edge_tts

def load_config(config_path="config/channel_en.yaml"):
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)["channel"]

def get_audio_duration(file_path: Path) -> float:
    cmd = [
        "ffprobe", "-v", "error",
        "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        str(file_path)
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    return float(result.stdout.strip())

async def generate_scene_audio(text: str, voice: str, output_file: Path):
    communicate = edge_tts.Communicate(text, voice, rate="-5%")
    await communicate.save(str(output_file))

async def run_tts(video_dir: str = "output/test_video"):
    cfg = load_config()
    voice = cfg["voice"]
    base_path = Path(video_dir)
    script_file = base_path / "script.json"
    audio_dir = base_path / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)

    with open(script_file, "r", encoding="utf-8") as f:
        script_data = json.load(f)

    concat_list_path = audio_dir / "concat_list.txt"
    concat_lines = []

    print(f"Generando narración con la voz: {voice}...")
    for scene in script_data["scenes"]:
        sid = scene["scene_id"]
        audio_file = audio_dir / f"scene_{sid:02d}.mp3"
        await generate_scene_audio(scene["narration"], voice, audio_file)

        duration = get_audio_duration(audio_file)
        scene["audio_path"] = str(audio_file)
        scene["duration"] = round(duration, 3)
        concat_lines.append(f"file '{audio_file.name}'")
        print(f"  Escena {sid:02d} lista ({scene['duration']}s)")

    # Guardar duraciones actualizadas en script.json
    with open(script_file, "w", encoding="utf-8") as f:
        json.dump(script_data, f, indent=2, ensure_ascii=False)

    # Unir todos los audios en una pista maestra con FFmpeg
    with open(concat_list_path, "w", encoding="utf-8") as f:
        f.write("\n".join(concat_lines))

    full_audio = audio_dir / "full_audio.mp3"
    subprocess.run([
        "ffmpeg", "-y", "-f", "concat", "-safe", "0",
        "-i", str(concat_list_path), "-c", "copy", str(full_audio)
    ], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    total_dur = get_audio_duration(full_audio)
    print(f"\nAudio maestro generado en: {full_audio} (Duración total: {total_dur:.2f}s)")

if __name__ == "__main__":
    video_dir = sys.argv[1] if len(sys.argv) > 1 else "output/test_video"
    asyncio.run(run_tts(video_dir))
