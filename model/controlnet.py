import cv2
import numpy as np
from PIL import Image


class ControlnetProcessor:
    def __init__(self, modules):
        self.modules = modules

    @staticmethod
    def process_canny(image,low_threshold=100, high_threshold=200):
        image = np.array(image)
        image = cv2.Canny(image, low_threshold, high_threshold)
        image = image[:, :, None]
        image = np.concatenate([image, image, image], axis=2)
        return Image.fromarray(image)

    def process(self, image):
        processed_image = []
        for module in self.modules:
            if module == "canny":
                processed = self.process_canny(image)
                processed_image.append(processed)
            else:
                print(f"{module} is not implemented yet.")          

        return processed_image
    