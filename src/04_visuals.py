import json
import os
import sys
import time
from pathlib import Path
from dotenv import load_dotenv
from huggingface_hub import InferenceClient

load_dotenv()

HF_MODELS_FALLBACK = [
    "black-forest-labs/FLUX.1-schnell",
    "stabilityai/stable-diffusion-xl-base-1.0",
    "black-forest-labs/FLUX.1-dev",
]

def generate_image_hf(client: InferenceClient, prompt: str):
    last_error = None
    for model_name in HF_MODELS_FALLBACK:
        try:
            return client.text_to_image(
                prompt=prompt,
                model=model_name,
                width=1280,
                height=720,
            )
        except Exception as e:
            print(f"    Falló {model_name}: {e}. Probando siguiente modelo...")
            last_error = e
            time.sleep(2)
    raise RuntimeError(f"Todos los modelos fallaron. Último error: {last_error}")

def run_visuals(video_dir: str = "output/test_video"):
    hf_token = os.getenv("HF_TOKEN")
    if not hf_token or "tu_token" in hf_token:
        raise ValueError("Configura tu HF_TOKEN real en el archivo .env")

    client = InferenceClient(provider="auto", api_key=hf_token)

    base_path = Path(video_dir)
    script_file = base_path / "script.json"
    img_dir = base_path / "images"
    img_dir.mkdir(parents=True, exist_ok=True)

    with open(script_file, "r", encoding="utf-8") as f:
        script_data = json.load(f)

    print("Generando imágenes 16:9 con Hugging Face Router (provider=auto)...")
    for scene in script_data["scenes"]:
        sid = scene["scene_id"]
        img_file = img_dir / f"scene_{sid:02d}.png"

        if img_file.exists():
            print(f"  Escena {sid:02d} ya existe, saltando...")
            scene["image_path"] = str(img_file)
            continue

        print(f"  Generando escena {sid:02d}...")
        image = generate_image_hf(client, scene["image_prompt"])
        image.save(img_file)

        scene["image_path"] = str(img_file)
        print(f"  Escena {sid:02d} guardada en {img_file}")
        time.sleep(1)

    with open(script_file, "w", encoding="utf-8") as f:
        json.dump(script_data, f, indent=2, ensure_ascii=False)

    print(f"\nTodas las imágenes generadas en: {img_dir}")

if __name__ == "__main__":
    video_dir = sys.argv[1] if len(sys.argv) > 1 else "output/test_video"
    run_visuals(video_dir)
