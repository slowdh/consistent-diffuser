import os
import argparse

import cv2
import numpy as np
import torch

from model.sd_pipeline import MultiControlnetPipeline
import os


def parse_args():
    parser = argparse.ArgumentParser()

    # SD diffusers
    parser.add_argument(
        "--model",
        type=str,
        default="prompthero/openjourney-v4",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default= "an vintage impressionist style painting, painted by Claude Monet, oil on canvas",
    )
    parser.add_argument(
        "--strength",
        type=float,
        default=0.2,
    )    
    parser.add_argument(
        "--guidance",
        type=float,
        default=7.0,
    )
    parser.add_argument(
        "--num_steps",
        type=int,
        default=25,
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=1024,
    )
    parser.add_argument(
        "--scheduler",
        type=str,
        default="UniPCMultistep",
        choices=["UniPCMultistep", "lms", "None"]
    )
    parser.add_argument(
        "--controlnets",
        type=str,
        default="canny",
        choices=["normal, canny, depth, hed, mlsd, seg, scribble, openpose"],
        help="comma seperated values would be parsed into array"
    )
    parser.add_argument(
        "--conditioning_scales",
        type=str,
        default="0.2",
        help="shoud have same length as number of controlnets"
    )

    # video settings
    parser.add_argument(
        "--fps",
        type=int,
        default=8,
    )

    # path
    parser.add_argument(
        "--video_load_path",
        type=str,
        default= "/home/fastdh/server/consistent-diffuser/data/test_videos/fp_test_3.mp4",
    )
    parser.add_argument(
        "--video_save_dir",
        type=str,
        default= "/home/fastdh/server/consistent-diffuser/data/output",
    )

    args = parser.parse_args()

    # process args
    if args.scheduler == "None":
        args.scheduler = None

    args.controlnets = [item.strip() for item in args.controlnets.split(",")]
    args.conditioning_scales = [float(item.strip()) for item in args.conditioning_scales.split(",")]

    return args


args = parse_args()
device = "cuda" if torch.cuda.is_available() else "cpu"
video_file_name = args.video_load_path.split("/")[-1]


pipe = MultiControlnetPipeline(
    model=args.model,
    scheduler=args.scheduler,
    controlnets=args.controlnets,
    seed=args.seed,
    device=device,
)

# set video capture and video writer
cap = cv2.VideoCapture(args.video_load_path)
out = cv2.VideoWriter(
    filename=os.path.join(args.video_save_dir, video_file_name),
    fourcc=cv2.VideoWriter_fourcc(*"mp4v"),
    fps=8,
    frameSize=(512, 512),
)

while True:
    ret, frame = cap.read()

    if ret:
        image = pipe.preprocess_image(frame)
        image = pipe.run(
            prompt=args.prompt,
            image=image,
            guidance=args.guidance,
            num_steps=args.num_steps,
            controlnet_conditioning_scale=args.conditioning_scales,
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
