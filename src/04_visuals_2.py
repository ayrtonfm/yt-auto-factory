import json
import sys
from pathlib import Path
import torch
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler

def run_local_visuals(video_dir: str = "output/test_video"):
    if not torch.cuda.is_available():
        raise RuntimeError(
            "PyTorch no detecta tu RTX 3050. Instala PyTorch con CUDA antes de ejecutar este script."
        )

    base_path = Path(video_dir)
    script_file = base_path / "script.json"
    img_dir = base_path / "images"
    img_dir.mkdir(parents=True, exist_ok=True)

    print("Cargando DreamShaper 8 optimizado para 4 GB VRAM...")
    pipe = StableDiffusionPipeline.from_pretrained(
        "Lykon/dreamshaper-8",
        torch_dtype=torch.float16,
        safety_checker=None,
    )
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config, algorithm_type="dpmsolver++", use_karras_sigmas=True)
    pipe = pipe.to("cuda")

    # Optimizaciones clave para no exceder 4 GB de VRAM
    pipe.enable_attention_slicing()
    pipe.vae.enable_slicing()

    with open(script_file, "r", encoding="utf-8") as f:
        script_data = json.load(f)

    negative_prompt = (
        "text, watermark, signature, blurry, low quality, deformed, ugly, "
        "cartoon, anime, modern objects, bright colors"
    )

    for scene in script_data["scenes"]:
        sid = scene["scene_id"]
        img_file = img_dir / f"scene_{sid:02d}.png"

        if img_file.exists():
            print(f"  Escena {sid:02d} ya existe, saltando...")
            scene["image_path"] = str(img_file)
            continue

        print(f"  Generando escena {sid:02d} en RTX 3050 (768x432)...")
        generator = torch.Generator("cuda").manual_seed(42 + sid)
        image = pipe(
            prompt=scene["image_prompt"],
            negative_prompt=negative_prompt,
            width=768,
            height=432,  # 16:9 seguro para 4 GB VRAM (FFmpeg escalará a 1080p/720p)
            num_inference_steps=25,
            guidance_scale=7.5,
            generator=generator,
        ).images[0]

        image.save(img_file)
        scene["image_path"] = str(img_file)
        print(f"  Escena {sid:02d} guardada en {img_file}")
        torch.cuda.empty_cache()

    with open(script_file, "w", encoding="utf-8") as f:
        json.dump(script_data, f, indent=2, ensure_ascii=False)

    print(f"\nTodas las imágenes generadas en: {img_dir}")

if __name__ == "__main__":
    video_dir = sys.argv[1] if len(sys.argv) > 1 else "output/test_video"
    run_local_visuals(video_dir)
