import cv2
import torch
import numpy as np
from PIL import Image
from ultralytics import YOLO
from transformers import CLIPProcessor, CLIPModel

class ClothingPipeline:
    def __init__(self, yolo_path="yolov8m.pt"):
        print("Initializing YOLO...")
        self.yolo_model = YOLO(yolo_path)
        
        print("Downloading/Loading OpenAI CLIP...")
        self.clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        self.clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        
        # Load the massive vocabulary from categories.py
        # pyrefly: ignore [missing-import]
        from src.categories import CLOTHING_CATEGORIES
        self.clothing_categories = CLOTHING_CATEGORIES
        
        print("Precomputing 850+ text categories (this takes a few seconds)...")
        text_inputs = [f"a photo of a person wearing a {c}" for c in self.clothing_categories]
        # pyrefly: ignore [unexpected-keyword]
        text_tokens = self.clip_processor(text=text_inputs, return_tensors="pt", padding=True, truncation=True)
        with torch.no_grad():
            # Foolproof method to get projected embeddings across all transformers versions
            dummy_image = torch.zeros(1, 3, 224, 224)
            outputs = self.clip_model(**text_tokens, pixel_values=dummy_image)
            self.text_features = outputs.text_embeds
            self.text_features = self.text_features / self.text_features.norm(p=2, dim=-1, keepdim=True)
        print("Precomputation complete!")
        
    def process_image(self, image_source):
        print(f"\nProcessing image...")
        
        if isinstance(image_source, str):
            img = cv2.imread(image_source)
            if img is None:
                print(f"Error: Could not read image at {image_source}.")
                return {}
        elif isinstance(image_source, np.ndarray):
            img = image_source  # Assume it's already a BGR numpy array if passed directly
        elif isinstance(image_source, Image.Image):
            img = cv2.cvtColor(np.array(image_source), cv2.COLOR_RGB2BGR)
        else:
            print("Unsupported image source type.")
            return {}
            
        results = self.yolo_model(img, verbose=False)
        # pyrefly: ignore [bad-index]
        result = results[0]
        
        clothing_counts = {}
        
        # pyrefly: ignore [missing-attribute]
        for box in result.boxes:
            class_id = int(box.cls[0].item())
            
            # YOLO class 0 is 'person'
            if class_id == 0:
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                
                # Crop the person out
                crop_img = img[y1:y2, x1:x2]
                
                if crop_img.size == 0:
                    continue
                    
                # Convert OpenCV BGR to PIL RGB
                crop_rgb = cv2.cvtColor(crop_img, cv2.COLOR_BGR2RGB)
                pil_img = Image.fromarray(crop_rgb)
                
                # Only process the image (Text is already precomputed!)
                # pyrefly: ignore [unexpected-keyword]
                image_inputs = self.clip_processor(images=pil_img, return_tensors="pt")
                with torch.no_grad():
                    # Foolproof method to get projected image embeddings
                    # pyrefly: ignore [unexpected-keyword]
                    dummy_text = self.clip_processor(text=["dummy"], return_tensors="pt")
                    outputs = self.clip_model(**dummy_text, **image_inputs)
                    image_features = outputs.image_embeds
                    image_features = image_features / image_features.norm(p=2, dim=-1, keepdim=True)
                
                # Calculate probabilities instantly
                logit_scale = self.clip_model.logit_scale.exp()
                logits_per_image = logit_scale * image_features @ self.text_features.t()
                probs = logits_per_image.softmax(dim=1)[0]
                top_probs, top_idxs = probs.topk(3)
                
                found_items = []
                seen_base_types = set()
                
                for prob, idx in zip(top_probs, top_idxs):
                    prob_val = prob.item()
                    
                    # Must be at least 15% confident to even be considered
                    if prob_val > 0.15:
                        item_name = self.clothing_categories[idx.item()]
                        
                        # Advanced Body Slot Logic
                        base = item_name.split()[-1]
                        
                        slot = "unknown"
                        if "shirt" in base or base == "t-shirt": slot = "top"
                        elif base in ["jeans", "pants", "shorts", "skirt", "sweatpants"]: slot = "bottom"
                        elif base in ["sweater", "hoodie", "jacket", "coat", "windbreaker", "poncho", "blazer"]: slot = "outerwear"
                        elif base in ["dress", "suit"]: slot = "full_body"
                        elif base in ["shoes", "sneakers", "boots"]: slot = "shoes"
                        
                        # Rule: If wearing a full-body outfit (dress/suit), block separate tops and bottoms!
                        if slot == "full_body":
                            seen_base_types.add("top")
                            seen_base_types.add("bottom")
                        # Rule: If wearing a top or bottom, block full-body outfits!
                        if slot in ["top", "bottom"]:
                            seen_base_types.add("full_body")
                        
                        # Only accept if we haven't already filled this body slot for this person!
                        if slot not in seen_base_types:
                            found_items.append((item_name, prob_val))
                            seen_base_types.add(slot)
                
                # Tally them all up
                for item_name, prob_val in found_items:
                    print(f"CLIP identified: {item_name} ({(prob_val * 100):.1f}%)")
                    if item_name in clothing_counts:
                        clothing_counts[item_name] += 1
                    else:

                        
                        clothing_counts[item_name] = 1
                    
        return clothing_counts

if __name__ == "__main__":
    pipeline = ClothingPipeline()
    
    for test_img in ["sample10.png"]:
        final_results = pipeline.process_image(test_img)
        
        print("\n===============================")
        print(f"👗 FINAL CLOTHING TALLY ({test_img}) 👕")
        print("===============================")
        for item, count in final_results.items():
            print(f" {item}: {count}")
        print("===============================\n")
