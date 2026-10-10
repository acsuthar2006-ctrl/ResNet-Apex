from ultralytics import YOLO

class PersonDetector:
    def __init__(self, model_path="yolov8n.pt"):
        print(f"Loading YOLO model from {model_path}...")
        self.yolo = YOLO(model_path)
        
    def get_person_crops(self, img_array, conf_threshold=0.40):
        """Runs inference and returns cropped images of detected people."""
        results = self.yolo(img_array, verbose=False)
        
        crops = []
        h, w = img_array.shape[:2]
        
        for box in results[0].boxes:
            cls_id = int(box.cls[0].item())
            conf = box.conf[0].item()
            
            # If it's a person (class 0) with decent confidence
            if cls_id == 0 and conf > conf_threshold:
                best_box = box.xyxy[0].cpu().numpy()
                x1, y1, x2, y2 = [int(v) for v in best_box]
                
                # Add 5% padding
                pad_x = int((x2 - x1) * 0.05)
                pad_y = int((y2 - y1) * 0.05)
                
                x1 = max(0, x1 - pad_x)
                y1 = max(0, y1 - pad_y)
                x2 = min(w, x2 + pad_x)
                y2 = min(h, y2 + pad_y)
                
                print(f"Person detected! Cropping (Conf: {conf:.2f})")
                crops.append({
                    "crop": img_array[y1:y2, x1:x2],
                    "box": [x1, y1, x2, y2],
                    "is_person": True
                })
                
        if len(crops) > 0:
            return crops
            
        print("No person detected. Using original full image.")
        return [{
            "crop": img_array,
            "box": [0, 0, w, h],
            "is_person": False
        }]
