import cv2
import torch
import numpy as np
from PIL import Image
from diffusers import StableDiffusionImg2ImgPipeline
from diffusers import LMSDiscreteScheduler


device = "cuda"
model = "prompthero/openjourney-v4"
prompt = (
    "an vintage impressionist style painting, painted by Claude Monet, oil on canvas"
)
video_path = "/home/fastdh/server/fast-painter/fp_test_2.mp4"
STRENGTH = 0.2
GUIDANCE = 7
NUM_STEPS = 25


pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
    model, torch_dtype=torch.float16
).to(device)
lms = LMSDiscreteScheduler.from_config(pipe.scheduler.config)
pipe.scheduler = lms
generator = torch.Generator(device=device).manual_seed(1024)

cap = cv2.VideoCapture(video_path)
out = cv2.VideoWriter(
    filename="output.mp4",
    fourcc=cv2.VideoWriter_fourcc(*"mp4v"),
    fps=24,
    frameSize=(512, 512),
)

while True:
    ret, frame = cap.read()

    if ret:
        image = Image.fromarray(frame, 'RGB')
        image.thumbnail((512, 512))

        image = pipe(
            prompt=prompt,
            image=image,
            strength=STRENGTH,
            guidance_scale=GUIDANCE,
            generator=generator,
            num_inference_steps=int(NUM_STEPS / STRENGTH),
        ).images[0]
        image = np.array(image)

        out.write(image)

        # debugging
        input("for debugging: press any key")
        cv2.imshow('frame',image)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

cap.release()
out.release()
cv2.destroyAllWindows()
