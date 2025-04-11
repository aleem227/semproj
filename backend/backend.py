from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import torch
import base64
import io
from PIL import Image
import torchvision.transforms as transforms
import re
from typing import Optional

from modelling_resnet34 import UTKFaceModel

app = FastAPI()

# Enable CORS for all routes
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# Load the model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = UTKFaceModel()
model.load_state_dict(torch.load('utk_face_model_resnet34.pt', map_location=device)['model_state_dict'])
model.to(device)
model.eval()

# Define transformation for images
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

class ImageRequest(BaseModel):
    image: str

@app.get('/health')
def health():
    return {"status": "ok"}

@app.post('/predict/resnet34')
async def predict(request: ImageRequest):
    try:
        # Get the base64 image data
        image_data = request.image
        # Remove the data URL prefix if present
        if 'data:image' in image_data:
            image_data = re.sub('^data:image/.+;base64,', '', image_data)
        
        # Decode base64 to image
        image_bytes = base64.b64decode(image_data)
        image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        
        # Transform image and prepare for model
        image_tensor = transform(image).unsqueeze(0).to(device)
        
        # Make prediction
        with torch.no_grad():
            age_output, gender_output = model(image_tensor)
            
        # Return results
        return {
            'age': float(age_output.item()),
            'gender': float(gender_output.item())
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=5000)