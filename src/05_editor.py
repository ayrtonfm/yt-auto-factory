import json
import subprocess
import sys
from pathlib import Path

def get_video_encoder():
    """Prueba si h264_nvenc realmente funciona en esta GPU; si no, usa libx264."""
    test_cmd = [
        "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
        "-f", "lavfi", "-i", "color=c=black:s=128x128:d=0.1",
        "-c:v", "h264_nvenc", "-f", "null", "-"
    ]
    res = subprocess.run(test_cmd, capture_output=True, text=True)
    if res.returncode == 0:
        return "h264_nvenc", ["-preset", "p4"]
    return "libx264", ["-preset", "fast", "-crf", "23"]

def run_ffmpeg(cmd, cwd=None):
    res = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True)
    if res.returncode != 0:
        print("\n--- ERROR DE FFMPEG ---")
        print(res.stderr)
        raise RuntimeError(f"FFmpeg falló con código {res.returncode}")

def render_video(video_dir: str = "output/test_video", fps: int = 30):
    base_path = Path(video_dir).resolve()
    script_file = base_path / "script.json"
    audio_file = base_path / "audio" / "full_audio.mp3"
    srt_file = base_path / "subtitles.srt"
    clips_dir = base_path / "clips"
    clips_dir.mkdir(parents=True, exist_ok=True)

    if not script_file.exists() or not audio_file.exists() or not srt_file.exists():
        raise FileNotFoundError("Faltan archivos base (script.json, full_audio.mp3 o subtitles.srt).")

    with open(script_file, "r", encoding="utf-8") as f:
        script_data = json.load(f)

    encoder, enc_args = get_video_encoder()
    print(f"Codificador verificado: {encoder}")

    concat_lines = []
    for scene in script_data["scenes"]:
        sid = scene["scene_id"]
        img_path = Path(scene["image_path"]).resolve()
        duration = float(scene["duration"])
        frames = max(1, int(round(duration * fps)))
        clip_path = clips_dir / f"scene_{sid:02d}.mp4"

        print(f"  Renderizando clip escena {sid:02d} ({duration:.2f}s)...")

        vf_filter = (
            f"scale=2560:1440,"
            f"zoompan=z='min(zoom+0.0005,1.15)':d={frames}:"
            f"x='iw/2-(iw/zoom/2)':y='ih/2-(ih/zoom/2)':s=1280x720:fps={fps},"
            f"format=yuv420p"
        )

        cmd = [
            "ffmpeg", "-y",
            "-i", str(img_path),
            "-vf", vf_filter,
            "-t", f"{duration:.3f}",
            "-c:v", encoder, *enc_args,
            "-r", str(fps),
            str(clip_path)
        ]
        run_ffmpeg(cmd)
        concat_lines.append(f"file '{clip_path.name}'")

    concat_list = clips_dir / "concat_clips.txt"
    with open(concat_list, "w", encoding="utf-8") as f:
        f.write("\n".join(concat_lines))

    merged_video = clips_dir / "merged_video.mp4"
    print("Uniendo todas las escenas...")
    run_ffmpeg([
        "ffmpeg", "-y", "-f", "concat", "-safe", "0",
        "-i", str(concat_list), "-c", "copy", str(merged_video)
    ])

    final_video = base_path / "final_video.mp4"
    sub_style = (
        "FontName=Arial,FontSize=20,Bold=1,"
        "PrimaryColour=&H00FFFFFF,OutlineColour=&H00000000,"
        "BorderStyle=1,Outline=2,Shadow=1,Alignment=2,MarginV=35"
    )

    print("Agregando narración y quemando subtítulos...")
    run_ffmpeg([
        "ffmpeg", "-y",
        "-i", "clips/merged_video.mp4",
        "-i", "audio/full_audio.mp3",
        "-vf", f"subtitles=subtitles.srt:force_style='{sub_style}'",
        "-c:v", encoder, *enc_args,
        "-c:a", "aac", "-b:a", "192k",
        "-shortest",
        "final_video.mp4"
    ], cwd=str(base_path))

    print(f"\n¡Video final listo en: {final_video}!")

if __name__ == "__main__":
    video_dir = sys.argv[1] if len(sys.argv) > 1 else "output/test_video"
    render_video(video_dir)
