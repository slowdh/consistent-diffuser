import cv2
import numpy as np

from sd_pipeline import ImageToImagePipeline


device = "cuda"
scheduler = "lms"
model = "prompthero/openjourney-v4"
prompt = (
    "an vintage impressionist style painting, painted by Claude Monet, oil on canvas"
)
video_path = "/home/fastdh/server/fast-painter/fp_test_2.mp4"
STRENGTH = 0.2
GUIDANCE = 7
NUM_STEPS = 25
SEED = 1024


pipe = ImageToImagePipeline(
    model=model,
    scheduler=scheduler,
    seed=SEED,
    device=device,
)

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
        image = pipe.preprocess_image(frame)
        image = pipe.run(
            prompt=prompt,
            image=image,
            strength=STRENGTH,
            guidance=GUIDANCE,
            num_steps=NUM_STEPS,
        )
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
