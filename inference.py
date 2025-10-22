import torch
from diffusers import StableDiffusionXLPipeline
from PIL import Image
import numpy as np
from realesrgan import RealESRGANer
from basicsr.archs.rrdbnet_arch import RRDBNet
import base64
from io import BytesIO

model_path = "/workspace/models/juggernautXL_v10.safetensors"
pipe = StableDiffusionXLPipeline.from_single_file(
    model_path,
    torch_dtype=torch.float16,
    use_safetensors=True,
    variant="fp16"
).to("cuda")

pipe.enable_xformers_memory_efficient_attention()

model_esrgan = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)
upsampler = RealESRGANer(
    scale=4,
    model_path='https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth',
    model=model_esrgan,
    tile=0,
    tile_pad=10,
    pre_pad=0,
    half=True,
    device="cuda"
)

def handler(job):
    job_input = job["input"]
    
    prompt = job_input.get("prompt", "a beautiful food photo")
    negative_prompt = job_input.get("negative_prompt", "blurry, cartoon, text, signature")
    seed = job_input.get("seed", None)
    width = job_input.get("width", 1280)
    height = job_input.get("height", 720)
    steps = job_input.get("steps", 25)
    guidance_scale = job_input.get("guidance_scale", 7.5)

    generator = torch.Generator(device="cuda")
    if seed is not None:
        generator = generator.manual_seed(seed)
    else:
        seed = generator.seed()

    image = pipe(
        prompt=prompt,
        negative_prompt=negative_prompt,
        width=width,
        height=height,
        guidance_scale=guidance_scale,
        num_inference_steps=steps,
        generator=generator
    ).images[0]

    np_image = np.array(image)
    upscaled, _ = upsampler.enhance(np_image, outscale=4)

    pil_img = Image.fromarray(upscaled)
    buffer = BytesIO()
    pil_img.save(buffer, format="JPEG", quality=95)
    img_str = base64.b64encode(buffer.getvalue()).decode('utf-8')

    return {
        "image": img_str,
        "seed_used": seed
    }
