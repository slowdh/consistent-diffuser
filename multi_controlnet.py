from diffusers import (
    ControlNetModel,
    StableDiffusionControlNetPipeline,
    UniPCMultistepScheduler,
)
import torch

controlnet_canny = ControlNetModel.from_pretrained("lllyasviel/sd-controlnet-canny", torch_dtype=torch.float16)
controlnet_normal = ControlNetModel.from_pretrained("lllyasviel/sd-controlnet-normal", torch_dtype=torch.float16)
pipe = StableDiffusionControlNetPipeline.from_pretrained(
    "prompthero/openjourney-v4", controlnet=[controlnet_canny, controlnet_normal], torch_dtype=torch.float16
)

pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)

# xformers
pipe.enable_xformers_memory_efficient_attention()
pipe.to("cuda")


prompt = (
    "an vintage impressionist style painting, painted by Claude Monet, oil on canvas"
)

image_path = "/home/fastdh/server/fast-painter/test_images/0HT6VEPCP8SDYW65.jpg"

# output = pipe(
#     prompt=prompt,
#     image=[pose_image, canny_image],
#     generator=generator,
#     num_images_per_prompt=2,
#     num_inference_steps=20,
# )


# pipe.to("cpu")
# print(pipe.device)
