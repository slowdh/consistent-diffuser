import torch
from PIL import Image
from diffusers import (
    ControlNetModel,
    StableDiffusionImg2ImgPipeline,
    StableDiffusionControlNetPipeline,
    UniPCMultistepScheduler,
    LMSDiscreteScheduler,
)

from controlnet import ControlnetProcessor


class ImageToImagePipeline:
    def __init__(self, model, scheduler, seed, device):
        self.pipe = self.get_pipeline(model, device)
        self.set_scheduler(scheduler)
        self.generator = torch.Generator(device=device).manual_seed(seed)

    @staticmethod
    def get_pipeline(model, device):
        pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
            model, torch_dtype=torch.float16
        )
        pipe.enable_xformers_memory_efficient_attention()
        pipe.to(device)

        return pipe

    def set_scheduler(self, scheduler):
        if scheduler is None:
            return
        elif scheduler == "lms":
            scheduler = LMSDiscreteScheduler.from_config(self.pipe.scheduler.config)
            self.pipe.scheduler = scheduler
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


class MultiControlnetPipeline:
    def __init__(self, model, scheduler, controlnets, seed, device):
        self.controlnet_modules = self.get_controlnet_modules(controlnets)
        self.pipe = self.get_pipeline(model, self.controlnet_modules, device)
        self.set_scheduler(scheduler)
        self.generator = torch.Generator(device=device).manual_seed(seed)
        self.controlnet_processor = ControlnetProcessor(controlnets)

    @staticmethod
    def get_controlnet_modules(controlnets):
        controlnet_modules = []
        for controlnet in controlnets:
            if controlnet == "canny":
                controlnet = "lllyasviel/sd-controlnet-canny"
            elif controlnet == "normal":
                controlnet = "lllyasviel/sd-controlnet-normal"
            else:
                raise NotImplementedError(f"Controlnet {controlnet} not supported yet")
            
            controlnet_modules.append(
                ControlNetModel.from_pretrained(controlnet, torch_dtype=torch.float16)
            )

        return controlnet_modules

    @staticmethod
    def get_pipeline(model, controlnet_modules, device):
        pipe = StableDiffusionControlNetPipeline.from_pretrained(
            model,
            controlnet=controlnet_modules,
            torch_dtype=torch.float16,
        )
        pipe.enable_xformers_memory_efficient_attention()
        pipe.to(device)

        return pipe

    def set_scheduler(self, scheduler):
        if scheduler is None:
            return
        elif scheduler == "lms":
            scheduler = LMSDiscreteScheduler.from_config(self.pipe.scheduler.config)
            self.pipe.scheduler = scheduler
        elif scheduler == "UniPCMultistep":
            self.pipe.scheduler = UniPCMultistepScheduler.from_config(
                self.pipe.scheduler.config
            )
        else:
            raise NotImplementedError("Scheduler not supported yet")

    def preprocess_image(self, image):
        image = Image.fromarray(image, "RGB")
        image.thumbnail((512, 512))
        return image

    def run(self, prompt, image, strength, guidance, num_steps):
        image = self.pipe(
            prompt=prompt,
            image=image,
            generator=self.generator,
            num_images_per_prompt=1,
            num_inference_steps=int(num_steps / strength),
            guidance_scale=guidance
        ).images[0]

        return image
    