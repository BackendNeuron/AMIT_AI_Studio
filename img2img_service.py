# img2img_service.py
import torch
from diffusers import StableDiffusionImg2ImgPipeline
from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image

from diffusers import StableDiffusionImg2ImgPipeline
from PIL import Image
import torch

def generate_img2img(input_image_path: str, user_prompt: str, strength=0.7):
    device = "cpu"
    image_size = (512, 512)
    num_inference_steps = 50
    guidance_scale = 8.0

    # Load Stable Diffusion img2img pipeline
    sd_pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5", torch_dtype=torch.float32
    ).to(device)

    # Load input image
    raw_image = Image.open(input_image_path).convert("RGB")
    init_image = raw_image.resize(image_size)

    # Use only user-provided prompt
    final_prompt = user_prompt
    print(f"[PROMPT] Sending to Stable Diffusion:\n{final_prompt}\n")

    # Generate image
    images = sd_pipe(
        prompt=final_prompt,
        image=init_image,
        strength=strength,
        guidance_scale=guidance_scale,
        num_inference_steps=num_inference_steps
    ).images

    return images[0]
