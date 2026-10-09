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
            "Analyze this image and identify all clothing pieces and garments.\n"
            "CRITICAL INSTRUCTIONS:\n"
            "- Identify ALL garments and clothing pieces present in the image.\n"
            "- If there is a person, identify what they are wearing.\n"
            "- If it is flat-lay or folded clothes, identify each garment accurately.\n"
            "- DO NOT include body parts (e.g. face, hands, person).\n"
            "- DO NOT include accessories, props, phones, bags, or jewelry.\n"
            "- DO NOT include furniture, backgrounds, or non-clothing items.\n"
            "- Return ONLY a single valid JSON dictionary where keys are descriptive clothing names (e.g., 'red flannel shirt') and values are counts (e.g., 2).\n"
            "- No markdown, no commentary, no preamble."
        )

        payload = {
            "model": self.model_name,
            "messages": [
                {
                    "role": "system",
                    "content": "You are a direct vision classification engine. You must NEVER think, reason out loud, or use <think> tags. Output ONLY a valid JSON object starting immediately with '{'."
                },
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
            
            # Remove any residual <think>...</think> tags if generated
            clean_content = re.sub(r'<think>.*?</think>', '', content, flags=re.DOTALL).strip()
            json_match = re.search(r'\{.*\}', clean_content, re.DOTALL)
            if json_match:
                return json.loads(json_match.group(0))
            else:
                print("Error: Could not find valid JSON block.")
                return {}
                
        except Exception as e:
            print(f"API Error: Make sure {self.model_name} is running at {self.api_url}")
            print(f"Exception details: {e}")
            return {}
