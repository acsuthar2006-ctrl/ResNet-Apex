import io
import cv2
import numpy as np
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from pipeline import ClothingPipeline

app = FastAPI(title="Clothing Detection API")

# Add CORS so the frontend can communicate with the backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

print("Initializing AI Pipeline (This will precompute 850 categories)...")
pipeline = ClothingPipeline()

@app.post("/predict")
async def predict(image: UploadFile = File(...)):
    contents = await image.read()
    # Convert uploaded bytes to an OpenCV BGR numpy array
    nparr = np.frombuffer(contents, np.uint8)
    img_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    if img_bgr is None:
        return {"error": "Could not decode image"}
        
    # Process the image in-memory
    results = pipeline.process_image(img_bgr)
    
    return results

# Serve the frontend directory statically at the root URL (/)
import os
frontend_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "frontend")
# Create the frontend directory if it doesn't exist so mounting doesn't crash on startup
os.makedirs(frontend_dir, exist_ok=True)
app.mount("/", StaticFiles(directory=frontend_dir, html=True), name="frontend")

if __name__ == "__main__":
    import uvicorn
    # When running 'python src/api.py', the module is just 'api'
    uvicorn.run("api:app", host="127.0.0.1", port=8000, reload=True)
