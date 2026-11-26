import json
import os
import sys
import time
import yaml
from pathlib import Path
from dotenv import load_dotenv
from google import genai
from google.genai import types
from google.genai.errors import ServerError, ClientError

load_dotenv()

MODELS_FALLBACK = [
    "gemini-3.5-flash",
    "gemini-3.5-flash-lite",
    "gemini-3.1-flash-lite",
    "gemini-2.5-flash-lite",
    "gemini-2.5-flash",
    "gemini-2.5-pro",
]

def load_config(config_path="config/channel_en.yaml"):
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)["channel"]

def call_gemini_with_fallback(client, prompt: str):
    last_error = None
    for model_name in MODELS_FALLBACK:
        try:
            print(f"Intentando generar guion con: {model_name}...")
            response = client.models.generate_content(
                model=model_name,
                contents=prompt,
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    temperature=0.7,
                ),
            )
            print(f"Éxito usando: {model_name}")
            return response.text
        except (ServerError, ClientError) as e:
            print(f"Falló {model_name} ({e.code}). Probando siguiente modelo...")
            last_error = e
            time.sleep(2)
    raise RuntimeError(f"Todos los modelos fallaron. Último error: {last_error}")

def generate_script(topic: str, output_dir: str = "output/test_video"):
    cfg = load_config()
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key or "pega_aqui" in api_key:
        raise ValueError("Configura tu GEMINI_API_KEY real en el archivo .env")

    client = genai.Client(api_key=api_key)

    prompt = f"""
    You are an elite YouTube documentary scriptwriter specializing in {cfg['niche']}.
    Write a gripping, high-retention video script in {cfg['language']} about: "{topic}".

    RULES:
    - Scene 1 MUST be an intense 10-second hook that creates immediate curiosity (NEVER say "Hello everyone" or "Welcome back").
    - Keep sentences punchy, atmospheric, and written for spoken narration.
    - Break the script into exactly {cfg['target_scenes']} scenes.
    - For each scene, write a detailed image generation prompt that ends with this exact style: "{cfg['visual_style']}".

    Return ONLY valid JSON with this exact structure:
    {{
      "title": "Click-worthy YouTube title (max 65 chars)",
      "description": "SEO YouTube description with 3 relevant hashtags",
      "scenes": [
        {{
          "scene_id": 1,
          "narration": "Spoken text for this scene...",
          "image_prompt": "Detailed visual description of the scene, {cfg['visual_style']}"
        }}
      ]
    }}
    """

    raw_json = call_gemini_with_fallback(client, prompt)
    script_data = json.loads(raw_json)

    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    json_file = out_path / "script.json"

    with open(json_file, "w", encoding="utf-8") as f:
        json.dump(script_data, f, indent=2, ensure_ascii=False)

    print(f"Guion generado con éxito en: {json_file}")
    print(f"Título: {script_data['title']}")
    print(f"Total de escenas: {len(script_data['scenes'])}")

if __name__ == "__main__":
    topic = sys.argv[1] if len(sys.argv) > 1 else "The Dancing Plague of 1518: When an entire city danced to death"
    generate_script(topic)
