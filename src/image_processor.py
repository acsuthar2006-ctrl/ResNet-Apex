import cv2
import numpy as np
import base64
from PIL import Image

class ImageProcessor:
    @staticmethod
    def load_image(image_source):
        """Loads an image from filepath, PIL, or returns the numpy array."""
        if isinstance(image_source, str):
            img = cv2.imread(image_source)
            if img is None:
                raise ValueError(f"Could not read image at {image_source}.")
            return img
        elif isinstance(image_source, np.ndarray):
            return image_source
        elif isinstance(image_source, Image.Image):
            return cv2.cvtColor(np.array(image_source), cv2.COLOR_RGB2BGR)
        else:
            raise TypeError("Unsupported image source type.")
            
    @staticmethod
    def resize_if_needed(img_array, max_dim=1024):
        """Resizes the image if its largest dimension exceeds max_dim."""
        h, w = img_array.shape[:2]
        if max(h, w) > max_dim:
            scale = max_dim / max(h, w)
            new_w, new_h = int(w * scale), int(h * scale)
            img_array = cv2.resize(img_array, (new_w, new_h), interpolation=cv2.INTER_AREA)
        return img_array

    @staticmethod
    def enhance_for_vlm(img_array):
        """Applies CLAHE and Sharpening to make clothing folds/textures pop."""
        # Convert to LAB color space to apply CLAHE to the Lightness channel
        lab = cv2.cvtColor(img_array, cv2.COLOR_BGR2LAB)
        l_channel, a, b = cv2.split(lab)
        
        # Apply CLAHE
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        cl = clahe.apply(l_channel)
        
        # Merge back
        merged_lab = cv2.merge((cl, a, b))
        enhanced_img = cv2.cvtColor(merged_lab, cv2.COLOR_LAB2BGR)
        
        # Apply Sharpening (Unsharp Mask)
        gaussian_blur = cv2.GaussianBlur(enhanced_img, (0, 0), 2.0)
        sharpened = cv2.addWeighted(enhanced_img, 1.5, gaussian_blur, -0.5, 0)
        
        return sharpened

    @staticmethod
    def encode_to_base64(img_array, quality=80):
        """Encodes numpy array to base64 jpeg."""
        success, buffer = cv2.imencode('.jpg', img_array, [cv2.IMWRITE_JPEG_QUALITY, quality])
        if not success:
            return None
        return base64.b64encode(buffer).decode('utf-8')
