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
        system_prompt = (
            "You are an expert Vision AI and Textile Classification Engine specialized in laundry auditing, "
            "wardrobe analysis, and garment recognition.\n"
            "Your objective is to inspect the provided image with extreme visual precision and identify every "
            "wearable clothing item, especially in challenging, folded, layered, or awkward arrangements.\n"
            "You must output ONLY a valid JSON dictionary starting with '{' and ending with '}'. "
            "Do not include any thought process, markdown code fences, commentary, or text outside the JSON object."
        )

        user_prompt = (
            "Analyze this image and identify all wearable clothing items and garments present.\n\n"
            "DETECTION GUIDELINES FOR AWKWARD & REAL-WORLD IMAGES:\n"
            "1. FOLDED & STACKED GARMENTS:\n"
            "   - Clothes are frequently folded into rectangles or squares. Look for hallmark structural clues:\n"
            "     * Collared Shirts / Polo Shirts: Look for collars, neckbands, buttons, plackets, cuffs, or chest patches.\n"
            "     * T-Shirts: Look for ribbed crewneck/V-neck collars, folded short/long sleeves, and soft knit fabric.\n"
            "     * Trousers / Jeans / Pants: Look for waistbands, belt loops, fly zippers, back/side pockets with buttons, denim rivets, and folded trouser legs.\n"
            "     * Sweaters / Hoodies / Jackets: Look for heavy knit textures, hoods, drawstrings, zippers, and ribbed cuffs/hems.\n"
            "2. OVERLAPPING & UNDERLYING ITEMS:\n"
            "   - When clothes are stacked or resting on top of one another, distinguish their fabric and color boundaries.\n"
            "   - Crucial: If one garment is laid underneath another (e.g., a shirt partially covered by folded pants or another shirt), "
            "you MUST detect and count the underlying garment too.\n"
            "3. CRUMPLED, AWKWARD, OR ROTATED VIEWS:\n"
            "   - Garments may be upside down, sideways, bunched up, or partially wrinkled. Identify each distinct garment by its unique fabric texture, seams, and color.\n"
            "4. WORN ON A PERSON:\n"
            "   - If a person is in the frame, identify their tops, bottoms, outerwear, and inner layers.\n\n"
            "STRICT NEGATIVE EXCLUSIONS (NEVER COUNT THESE):\n"
            "- BACKGROUND FABRICS: Bedsheets, blankets, quilts, duvet covers, mattress protectors, tablecloths, towels, curtains, and carpets/rugs are BACKGROUND SURFACES, NOT clothing—even if they have floral, striped, polka dot, or colorful patterns.\n"
            "- NON-CLOTHING OBJECTS: Do not include bags, backpacks, shoes, belts, jewelry, watches, phones, hangers, floor tiles, or furniture.\n"
            "- HUMAN BODY: Do not include the person, face, skin, or hands.\n\n"
            "NAMING & OUTPUT FORMAT:\n"
            "- Keys: Descriptive names formatted as '<color/pattern> <garment_type>' (e.g., 'beige collared polo shirt', 'blue denim button-down shirt', 'grey plaid formal trousers').\n"
            "- Values: Integer count of that specific garment (e.g., 1, 2).\n\n"
            "Example valid response:\n"
            "{\n"
            "  \"beige collared polo shirt\": 1,\n"
            "  \"blue denim button-down shirt\": 1,\n"
            "  \"grey plaid formal trousers\": 1\n"
            "}\n\n"
            "Return ONLY the JSON dictionary now:"
        )

        payload = {
            "model": self.model_name,
            "messages": [
                {
                    "role": "system",
                    "content": system_prompt
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": user_prompt},
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
            
            # 1. Remove any residual <think>...</think> tags if generated
            clean_content = re.sub(r'<think>.*?</think>', '', content, flags=re.DOTALL).strip()
            
            # 2. Strip markdown fences if present
            clean_content = re.sub(r'^```(?:json)?\s*', '', clean_content, flags=re.MULTILINE)
            clean_content = re.sub(r'\s*```$', '', clean_content, flags=re.MULTILINE).strip()
            
            # 3. Extract JSON object
            json_match = re.search(r'\{.*\}', clean_content, re.DOTALL)
            if json_match:
                json_str = json_match.group(0)
                # Fix any trailing commas before closing braces/brackets
                json_str = re.sub(r',\s*([\}\]])', r'\1', json_str)
                raw_dict = json.loads(json_str)
                
                # Sanitize: ensure keys are clean strings and counts are integers
                sanitized_dict = {}
                for k, v in raw_dict.items():
                    if isinstance(k, str) and k.strip():
                        try:
                            sanitized_dict[k.strip().lower()] = int(v) if int(v) > 0 else 1
                        except (ValueError, TypeError):
                            sanitized_dict[k.strip().lower()] = 1
                return sanitized_dict
            else:
                print("Error: Could not find valid JSON block.")
                return {}
                
        except Exception as e:
            print(f"API Error: Make sure {self.model_name} is running at {self.api_url}")
            print(f"Exception details: {e}")
            return {}
