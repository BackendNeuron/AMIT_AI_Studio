from diffusers import StableDiffusionPipeline
import torch
import sys  # ✅ For immediate flush of progress printing

STYLE_PRESETS = {
    "photo": { "positive": "ultra realistic, high detail, soft lighting, sharp focus, cinematic, masterpiece, 8k", "negative": "blurry, low quality, distorted, artifacts, bad anatomy" },
    "anime": { "positive": "high quality anime art style, clean lines, vibrant colors, masterpiece", "negative": "blurry, low detail, distorted face, off-model" },
    "art": { "positive": "fantasy painting, highly detailed, concept art, dramatic lighting, masterpiece", "negative": "flat colors, low quality, bad composition" },
    "cyberpunk": { "positive": "cyberpunk style, neon lights, futuristic, glowing atmosphere, masterpiece", "negative": "washed out, dull colors, bad lighting" }
}

from diffusers import StableDiffusionPipeline, EulerDiscreteScheduler

print("✅ Loading Stable Diffusion Model... (only once)")

pipe = StableDiffusionPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    torch_dtype=torch.float32
)

# ✅ Replace scheduler to enable callbacks
pipe.scheduler = EulerDiscreteScheduler.from_config(pipe.scheduler.config)

device = "cuda" if torch.cuda.is_available() else "cpu"
pipe = pipe.to(device)
print(f"🚀 Model ready on {device.upper()}")

def generate_txt2img(style_choice: str = None, custom_prompt: str = None):
    positive_base = "high detail, masterpiece"
    negative_base = "blurry, low quality"

    if style_choice and style_choice.lower() in STYLE_PRESETS:
        preset = STYLE_PRESETS[style_choice.lower()]
        positive_prompt = f"{preset['positive']}, {custom_prompt}"
        negative_prompt = preset["negative"]
    else:
        positive_prompt = f"{positive_base}, {custom_prompt}"
        negative_prompt = negative_base

    # ✅ THIS IS CORRECT — Callback goes HERE inside the function
    def progress_callback(step: int, timestep: int, latents):
        print(f"[SD] Step {step}/35 | Timestep: {timestep}")
        sys.stdout.flush()  # 👈 Forces immediate terminal output

    # ✅ Pass callback here — ALSO CORRECT
    result = pipe(
        positive_prompt,
        height=512,
        width=512,
        num_inference_steps=35,
        guidance_scale=7.5,
        negative_prompt=negative_prompt,
        callback=progress_callback,
        callback_steps=1
    )

    return result.images[0]  # ✅ All good
