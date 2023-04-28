import torch
from PIL import Image
from diffusers import StableDiffusionImg2ImgPipeline
from diffusers import LMSDiscreteScheduler


class ImageToImagePipeline:
    def __init__(self, model, scheduler, seed, device):
        self.pipe = self.get_pipeline(model, device)
        self.set_scheduler(scheduler)
        self.generator = torch.Generator(device=device).manual_seed(seed)

    @staticmethod
    def get_pipeline(model, device):
        pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
            model, torch_dtype=torch.float16
        ).to(device)
        return pipe

    def set_scheduler(self, scheduler):
        if scheduler == "lms":
            scheduler = LMSDiscreteScheduler.from_config(self.pipe.scheduler.config)
            self.pipe.scheduler = scheduler
        elif scheduler is None:
            return
        else:
            raise NotImplementedError("Scheduler not supported yet")

    def run(self, prompt, image, strength, guidance, num_steps):
        image = self.pipe(
            prompt=prompt,
            image=image,
            strength=strength,
            guidance_scale=guidance,
            generator=self.generator,
            num_inference_steps=int(num_steps / strength),
        ).images[0]

        return image

    def preprocess_image(self, image):
        image = Image.fromarray(image, "RGB")
        image.thumbnail((512, 512))

        return image
