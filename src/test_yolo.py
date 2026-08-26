import cv2
from ultralytics import YOLO
import os

def test_yolo_detection(image_path, output_dir="runs/crops"):
    # Create the output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # 1. Load the pre-trained YOLOv8 model (this will download a small model on first run)
    print("Loading YOLO model...")
    model = YOLO("yolov8n.pt") 

    # 2. Run inference (detection) on the image
    print(f"Running detection on {image_path}...")
    results = model(image_path)

    # 3. Read the image using OpenCV so we can crop it
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Could not read image at {image_path}")
        return

    # 4. Process the results
    # YOLO returns a list of Results objects (one for each image, we only passed one)
    # pyrefly: ignore [bad-index]
    result = results[0]
    
    crop_count = 0
    
    # Iterate through all the detected bounding boxes
    # pyrefly: ignore [missing-attribute]
    for box in result.boxes:
        # Get the class ID of the detected object
        class_id = int(box.cls[0].item())
        
        # YOLO's COCO dataset uses class_id 0 for 'person'
        if class_id == 0:
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            
            crop_img = img[y1:y2, x1:x2]
            
            crop_filename = os.path.join(output_dir, f"person_{crop_count}.jpg")
            cv2.imwrite(crop_filename, crop_img)
            print(f"Saved cropped person to {crop_filename}")
            
            crop_count += 1
            
    print(f"Total people found and cropped: {crop_count}")

if __name__ == "__main__":
    # TODO: Replace with the path to a test image you want to use!
    sample_image = "sample.jpg" 
    
    if not os.path.exists(sample_image):
        print(f"Please place an image named '{sample_image}' in the root directory!")
    else:
        test_yolo_detection(sample_image)
