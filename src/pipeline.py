import collections
from image_processor import ImageProcessor
from detector import PersonDetector
from vlm_client import VLMClient

class ClothingPipeline:
    def __init__(self):
        self.detector = PersonDetector()
        self.vlm_client = VLMClient()
        self.processor = ImageProcessor()
        
    def process_image(self, image_source):
        print(f"\nProcessing image via Modular Pipeline...")
        
        try:
            img_array = self.processor.load_image(image_source)
        except Exception as e:
            print(f"Error loading image: {e}")
            return {}
            
        # Step 1: Detect and crop people
        person_crops = self.detector.get_person_crops(img_array)
        
        master_tallies = collections.defaultdict(int)
        people_results = []
        
        for crop_idx, person_data in enumerate(person_crops):
            print(f"Processing crop #{crop_idx + 1} / {len(person_crops)}")
            crop_img = person_data["crop"]
            box = person_data["box"]
            is_person = person_data["is_person"]
            
            # Step 2: Smart Resize
            crop_img = self.processor.resize_if_needed(crop_img)
            
            # Step 3: Enhance Image for VLM (CLAHE + Sharpening)
            enhanced_img = self.processor.enhance_for_vlm(crop_img)
            
            # Step 4: Base64 Encode
            base64_img = self.processor.encode_to_base64(enhanced_img)
            if not base64_img:
                continue
                
            # Step 5: Ask LLM to count clothes
            clothing_counts = self.vlm_client.analyze_clothing(base64_img)
            
            people_results.append({
                "box": box,
                "clothes": clothing_counts,
                "is_person": is_person
            })
            
            for item, count in clothing_counts.items():
                master_tallies[item] += count
                
        final_result = {
            "tallies": dict(master_tallies),
            "people": people_results
        }
        print(f"VLM total identified across all crops: {dict(master_tallies)}")
        return final_result

if __name__ == "__main__":
    pipeline = ClothingPipeline()
    import os
    # Test on a sample image if it exists
    test_img = "sample.png"
    if os.path.exists(test_img):
        final_results = pipeline.process_image(test_img)
        print("\n===============================")
        print(f"👗 FINAL CLOTHING TALLY ({test_img}) 👕")
        print("===============================")
        if isinstance(final_results, dict) and "tallies" in final_results:
            for item, count in final_results["tallies"].items():
                print(f" {item}: {count}")
        print("===============================\n")
