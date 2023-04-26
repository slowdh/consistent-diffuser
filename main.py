import cv2
import torch
from PIL import Image
from diffusers import StableDiffusionImg2ImgPipeline


device = "cuda"
model = "prompthero/openjourney-v4"
prompt = (
    "an vintage impressionist style painting, painted by Claude Monet, oil on canvas"
)
video_path = "/Users/leo/Desktop/fun/programming/fast-painter/fast-painter.mp4"


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


cap = cv2.VideoCapture(video_path)

frame_width = int(cap.get(3))
frame_height = int(cap.get(4))
assert frame_width == frame_height and frame_width == 512

out = cv2.VideoWriter(
    filename="output.avi",
    fourcc=cv2.VideoWriter_fourcc(*"X264"),
    fps=24,
    frameSize=(frame_width, frame_height),
)


while True:
    ret, frame = cap.read()
    out.write(frame)

    input("test")
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
out.release()
cv2.destroyAllWindows()
