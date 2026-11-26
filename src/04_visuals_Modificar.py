import json
import os
import sys
import time
from pathlib import Path
from dotenv import load_dotenv
from google import genai
from google.genai import types
from google.genai.errors import ServerError, ClientError

load_dotenv()

IMAGE_MODELS_FALLBACK = [
    "gemini-3.1-flash-image",
    "gemini-3.1-flash-lite-image",
    "gemini-2.5-flash-image",
]

def generate_image_bytes(client, prompt: str) -> bytes:
    last_error = None
    for model_name in IMAGE_MODELS_FALLBACK:
        try:
            response = client.models.generate_content(
                model=model_name,
                contents=f"Generate a 16:9 widescreen image: {prompt}",
                config=types.GenerateContentConfig(
                    response_modalities=["IMAGE"],
                    image_config=types.ImageConfig(aspect_ratio="16:9"),
                ),
            )
            for part in response.candidates[0].content.parts:
                if part.inline_data and part.inline_data.data:
                    return part.inline_data.data
            raise RuntimeError(f"{model_name} no devolvió datos de imagen.")
        except (ServerError, ClientError, Exception) as e:
            print(f"    Falló {model_name}: {e}. Probando siguiente modelo...")
            last_error = e
            time.sleep(2)
    raise RuntimeError(f"Todos los modelos de imagen fallaron. Último error: {last_error}")

def run_visuals(video_dir: str = "output/test_video"):
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key or "pega_aqui" in api_key:
        raise ValueError("Configura tu GEMINI_API_KEY real en el archivo .env")

    client = genai.Client(api_key=api_key)
    base_path = Path(video_dir)
    script_file = base_path / "script.json"
    img_dir = base_path / "images"
    img_dir.mkdir(parents=True, exist_ok=True)

    with open(script_file, "r", encoding="utf-8") as f:
        script_data = json.load(f)

    print(f"Generando {len(script_data['scenes'])} imágenes 16:9...")
    for scene in script_data["scenes"]:
        sid = scene["scene_id"]
        img_file = img_dir / f"scene_{sid:02d}.png"

        if img_file.exists():
            print(f"  Escena {sid:02d} ya existe, saltando...")
            scene["image_path"] = str(img_file)
            continue

        print(f"  Generando escena {sid:02d}...")
        img_bytes = generate_image_bytes(client, scene["image_prompt"])
        with open(img_file, "wb") as f:
            f.write(img_bytes)

        scene["image_path"] = str(img_file)
        print(f"  Escena {sid:02d} guardada en {img_file}")
        time.sleep(2)  # Pausa breve para respetar límites por minuto

    with open(script_file, "w", encoding="utf-8") as f:
        json.dump(script_data, f, indent=2, ensure_ascii=False)

    print(f"\nTodas las imágenes generadas en: {img_dir}")

if __name__ == "__main__":
    video_dir = sys.argv[1] if len(sys.argv) > 1 else "output/test_video"
    run_visuals(video_dir)
