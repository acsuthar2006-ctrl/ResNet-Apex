import cv2
import numpy as np
import base64
import json
import re
import requests
import os
from dotenv import load_dotenv
from PIL import Image
from ultralytics import YOLO

# Load environment variables from .env file
load_dotenv()

# ==========================================
# CONFIGURATION (Loaded from .env)
# ==========================================
API_URL = os.environ.get("VLM_API_URL", "http://localhost:1234/v1/chat/completions")
MODEL_NAME = os.environ.get("VLM_MODEL_NAME", "Qwen3.5 9B")
# ==========================================

class ClothingPipeline:
    def __init__(self):
        print(f"Initializing VLM Pipeline pointing to {API_URL}...")
    #     print("Loading YOLOv8 Auto-Cropper...")
    #     self.yolo = YOLO("yolov8n.pt")
        
    # def _crop_to_person(self, img_array):
    #     # Run extremely fast YOLO inference
    #     results = self.yolo(img_array, verbose=False)
        
    #     # Look for the 'person' class (class 0 in COCO)
    #     best_box = None
    #     highest_conf = 0.0
        
    #     # pyrefly: ignore [bad-index, missing-attribute]
    #     for box in results[0].boxes:
    #         cls_id = int(box.cls[0].item())
    #         conf = box.conf[0].item()
    #         if cls_id == 0 and conf > highest_conf:
    #             highest_conf = conf
    #             best_box = box.xyxy[0].cpu().numpy()
                
    #     if best_box is not None:
    #         # Add 5% padding so we don't cut off shoes or hats
    #         h, w = img_array.shape[:2]
    #         x1, y1, x2, y2 = [int(v) for v in best_box]
            
    #         pad_x = int((x2 - x1) * 0.05)
    #         pad_y = int((y2 - y1) * 0.05)
            
    #         x1 = max(0, x1 - pad_x)
    #         y1 = max(0, y1 - pad_y)
    #         x2 = min(w, x2 + pad_x)
    #         y2 = min(h, y2 + pad_y)
            
    #         print(f"Person detected! Cropping background (Conf: {highest_conf:.2f})")
    #         return img_array[y1:y2, x1:x2]
            
    #     print("No person detected. Using original full image.")
    #     return img_array
        
    def _encode_image_to_base64(self, img_array):
        # Convert BGR numpy array to base64 jpeg with 80% quality compression
        # This drastically reduces the HTTP payload size sent to the LLM
        success, buffer = cv2.imencode('.jpg', img_array, [cv2.IMWRITE_JPEG_QUALITY, 80])
        if not success:
            return None
        # pyrefly: ignore [bad-argument-type]
        return base64.b64encode(buffer).decode('utf-8')

    def process_image(self, image_source):
        print(f"\nProcessing image via Local API...")
        
        if isinstance(image_source, str):
            img = cv2.imread(image_source)
            if img is None:
                print(f"Error: Could not read image at {image_source}.")
                return {}
        elif isinstance(image_source, np.ndarray):
            img = image_source
        elif isinstance(image_source, Image.Image):
            img = cv2.cvtColor(np.array(image_source), cv2.COLOR_RGB2BGR)
        else:
            print("Error: Unsupported image source type.")
            return {}
            
        # Step 1: Auto-Crop the person (removes background to save LLM computation)
        # img = self._crop_to_person(img)
            
        # --- SPEED OPTIMIZATION ---
        # Resize image so the longest edge is max 1024px to prevent the LLM from choking on huge files
        max_dim = 1024
        h, w = img.shape[:2]
        if max(h, w) > max_dim:
            scale = max_dim / max(h, w)
            new_w, new_h = int(w * scale), int(h * scale)
            img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
        # --------------------------
            
        base64_img = self._encode_image_to_base64(img)
        if not base64_img:
            return {}

        prompt = (
            "Analyze this image carefully. Identify ONLY the garments and clothing pieces the person is wearing "
            "(such as shirts, pants, jackets, dresses, shoes, skirts). "
            "CRITICAL RULES: \n"
            "1. DO NOT include body parts (e.g. face, hands, person).\n"
            "2. DO NOT include accessories, props, phones, bags, or jewelry.\n"
            "3. DO NOT include furniture, backgrounds, or any non-clothing items.\n"
            "4. Return ONLY a valid JSON dictionary where keys are descriptive clothing names (e.g., 'red flannel shirt') and values are counts (e.g., 1).\n"
            "Do not include any other text, markdown blocks, or explanation outside the JSON."
        )

        payload = {
            "model": MODEL_NAME,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_img}"}}
                    ]
                }
            ],
            "temperature": 0.1 # low temp for deterministic JSON
        }

        try:
            response = requests.post(API_URL, json=payload)
            response.raise_for_status()
            
            data = response.json()
            content = data['choices'][0]['message']['content'].strip()
            
            # Print exactly what Qwen generated to the terminal so we can debug it
            print("\n--- RAW QWEN RESPONSE ---")
            print(content)
            print("-------------------------\n")
            
            # Clean up markdown formatting and extract ONLY the JSON using Regex
            json_match = re.search(r'\{.*\}', content, re.DOTALL)
            if json_match:
                content = json_match.group(0)
            else:
                print("Error: Could not find valid JSON block in Qwen's response.")
                return {}
                
            clothing_counts = json.loads(content)
            
            print(f"VLM identified: {clothing_counts}")
            return clothing_counts
            
        except Exception as e:
            print(f"API Error: Make sure {MODEL_NAME} is running at {API_URL}")
            print(f"Exception details: {e}")
            return {}

if __name__ == "__main__":
    pipeline = ClothingPipeline()
    for test_img in ["sample7.png", "sample.png"]:
        final_results = pipeline.process_image(test_img)
        print("\n===============================")
        print(f"👗 FINAL CLOTHING TALLY ({test_img}) 👕")
        print("===============================")
        if isinstance(final_results, dict):
            for item, count in final_results.items():
                print(f" {item}: {count}")
        print("===============================\n")
