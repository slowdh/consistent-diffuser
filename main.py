import cv2
import torch
import numpy as np
from PIL import Image
from diffusers import StableDiffusionImg2ImgPipeline


device = "cuda"
model = "prompthero/openjourney-v4"
prompt = (
    "an vintage impressionist style painting, painted by Claude Monet, oil on canvas"
)
video_path = "/home/fastdh/server/fast-painter/fast-painter.mp4"


pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
    model, torch_dtype=torch.float16
).to(device)
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
            strength=0.5,
            guidance_scale=6,
            generator=generator,
        ).images[0]
        image = np.array(image)

        out.write(image)
        cv2.imshow('frame',image)
        input("for debugging")

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

cap.release()
out.release()
cv2.destroyAllWindows()
