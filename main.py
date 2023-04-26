import torch
from PIL import Image
from diffusers import StableDiffusionImg2ImgPipeline


device = "cuda"
model = "prompthero/openjourney-v4"
prompt = (
    "an vintage impressionist style painting, painted by Claude Monet, oil on canvas"
)

pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
    model, torch_dtype=torch.float16
).to(device)

image = Image.open(
    "/home/fastdh/server/fast-painter/test_images/0II7ERB4P6FNXDF5.jpg"
).convert("RGB")
image.thumbnail((512, 512))

generator = torch.Generator(device=device).manual_seed(1024)
image = pipe(
    prompt=prompt,
    image=image,
    strength=0.6,
    guidance_scale=7.5,
    generator=generator,
).images[0]
