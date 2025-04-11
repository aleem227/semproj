from flask import Flask, request, jsonify
from flask_cors import CORS
import torch
import base64
import io
from PIL import Image
import torchvision.transforms as transforms
import re

from modelling_resnet34 import UTKFaceModel

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

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

@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'ok'})

@app.route('/predict/resnet34', methods=['POST'])
def predict():
    if not request.json or 'image' not in request.json:
        return jsonify({'error': 'No image provided'}), 400
    
    try:
        # Get the base64 image data
        image_data = request.json['image']
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
        return jsonify({
            'age': float(age_output.item()),
            'gender': float(gender_output.item())
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500



if __name__ == '__main__':
    app.run(debug=True)