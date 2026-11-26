import json
import os
import shutil
import sys
from pathlib import Path
import torch
from diffusers import DiffusionPipeline

def run_colab_visuals(script_path: str = "script.json", out_dir: str = "images"):
    if not torch.cuda.is_available():
        raise RuntimeError("No se detectó GPU. En Colab ve a: Entorno de ejecución -> Cambiar tipo -> GPU T4 o L4.")

    script_file = Path(script_path)
    if not script_file.exists():
        raise FileNotFoundError(f"No se encontró {script_file}. Súbelo al panel izquierdo de Colab.")

    img_dir = Path(out_dir)
    img_dir.mkdir(parents=True, exist_ok=True)

    print("Cargando SDXL en GPU (fp16)...")
    pipe = DiffusionPipeline.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0",
        torch_dtype=torch.float16,
        use_safetensors=True,
        variant="fp16",
    ).to("cuda")

    with open(script_file, "r", encoding="utf-8") as f:
        script_data = json.load(f)

    negative_prompt = "text, watermark, signature, blurry, low quality, cartoon, anime, modern, bright neon colors"

    for scene in script_data["scenes"]:
        sid = scene["scene_id"]
        img_file = img_dir / f"scene_{sid:02d}.png"

        if img_file.exists():
            print(f"  Escena {sid:02d} ya existe, saltando...")
            continue

        print(f"  Generando escena {sid:02d} (1024x576)...")
        generator = torch.Generator("cuda").manual_seed(42 + sid)
        image = pipe(
            prompt=scene["image_prompt"],
            negative_prompt=negative_prompt,
            width=1024,
            height=576,
            num_inference_steps=25,
            guidance_scale=7.5,
            generator=generator,
        ).images[0]

        image.save(img_file)
        print(f"  Escena {sid:02d} guardada en {img_file}")

    shutil.make_archive("images", "zip", img_dir)
    print("\nArchivo images.zip creado con éxito.")

    try:
        from google.colab import files
        files.download("images.zip")
    except ImportError:
        print("Ejecutado fuera de Colab. Descarga manualmente images.zip.")

if __name__ == "__main__":
    script_arg = sys.argv[1] if len(sys.argv) > 1 else "script.json"
    run_colab_visuals(script_arg)
