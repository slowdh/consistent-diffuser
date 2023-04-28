import cv2
import numpy as np
from PIL import Image
from transformers import pipeline


class ControlnetProcessor:
    def __init__(self, modules):
        self.modules = modules
        self.depth_estimator = None
        
        if 'normal' in modules:
            self.depth_estimator = pipeline("depth-estimation", model ="Intel/dpt-hybrid-midas" )

    @staticmethod
    def process_canny(image,low_threshold=100, high_threshold=200):
        image = np.array(image)
        image = cv2.Canny(image, low_threshold, high_threshold)
        image = image[:, :, None]
        image = np.concatenate([image, image, image], axis=2)
        return Image.fromarray(image)

    def process_normal(self, image):
        image = self.depth_estimator(image)['predicted_depth'][0].numpy()
        image_depth = image.copy()
        image_depth -= np.min(image_depth)
        image_depth /= np.max(image_depth)

        bg_threhold = 0.4

        x = cv2.Sobel(image, cv2.CV_32F, 1, 0, ksize=3)
        x[image_depth < bg_threhold] = 0

        y = cv2.Sobel(image, cv2.CV_32F, 0, 1, ksize=3)
        y[image_depth < bg_threhold] = 0

        z = np.ones_like(x) * np.pi * 2.0

        image = np.stack([x, y, z], axis=2)
        image /= np.sum(image ** 2.0, axis=2, keepdims=True) ** 0.5
        image = (image * 127.5 + 127.5).clip(0, 255).astype(np.uint8)
        return Image.fromarray(image)
    
    def process(self, image):
        processed_image = []
        for module in self.modules:
            if module == "canny":
                processed = self.process_canny(image)
                processed_image.append(processed)
            elif module == "normal":
                processed = self.process_normal(image)
                processed_image.append(processed)
            else:
                print(f"{module} is not implemented yet in ControlnetProcessor.")          

        return processed_image
    