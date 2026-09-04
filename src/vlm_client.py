import requests
import json
import re
import os
from dotenv import load_dotenv

class VLMClient:
    def __init__(self):
        load_dotenv()
        self.api_url = os.environ.get("VLM_API_URL", "http://localhost:1234/v1/chat/completions")
        self.model_name = os.environ.get("VLM_MODEL_NAME", "qwen/qwen3-vl-8b")
        print(f"Initializing VLM Client pointing to {self.api_url}...")
        
    def analyze_clothing(self, base64_img):
        prompt = (
            "Analyze this image carefully. Identify ALL garments and clothing pieces present in the image. "
            "If there is a person, identify what they are wearing. "
            "If it is a flat-lay photo or clothes folded/lying around, identify those items exactly as they appear. "
            "CRITICAL RULES: \n"
            "1. DO NOT include body parts (e.g. face, hands, person).\n"
            "2. DO NOT include accessories, props, phones, bags, or jewelry.\n"
            "3. DO NOT include furniture, backgrounds, or any non-clothing items like books or any kind of stationries.\n"
            "4. Count each distinct item carefully to get a highly accurate total count of clothes with their type.\n"
            "5. Return ONLY a valid JSON dictionary where keys are descriptive clothing names (e.g., 'red flannel shirt') and values are counts (e.g., 2).\n"
            "Do not include any other text, markdown blocks, or explanation outside the JSON."
        )

        payload = {
            "model": self.model_name,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_img}"}}
                    ]
                }
            ],
            "temperature": 0.1
        }

        try:
            response = requests.post(self.api_url, json=payload)
            response.raise_for_status()
            
            data = response.json()
            content = data['choices'][0]['message']['content'].strip()
            
            print(f"\n--- RAW QWEN RESPONSE ---")
            print(content)
            print("-------------------------\n")
            
            json_match = re.search(r'\{.*\}', content, re.DOTALL)
            if json_match:
                return json.loads(json_match.group(0))
            else:
                print("Error: Could not find valid JSON block.")
                return {}
                
        except Exception as e:
            print(f"API Error: Make sure {self.model_name} is running at {self.api_url}")
            print(f"Exception details: {e}")
            return {}
