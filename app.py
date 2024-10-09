from flask import Flask, request, jsonify
import pickle
import numpy as np
from PIL import Image
import torch
import torchvision.transforms as transforms
from torchvision import models

app = Flask(__name__)

# Load all three models using .pkcls files (or .pa files)
with open('klasifikacija_zdravja_logistic_regression.pkcls', 'rb') as f:
    health_model = pickle.load(f)

with open('klasifikacija_vrste_logistic_regression.pkcls', 'rb') as f:
    species_model = pickle.load(f)

with open('klasifikacija_bolezni_logistic_regression.pkcls', 'rb') as f:
    sickness_model = pickle.load(f)

# Load pre-trained SqueezeNet for image embedding
squeezenet = models.squeezenet1_1(pretrained=True)
squeezenet.eval()  # Set to evaluation mode (no training)

# Define image transformation pipeline (resize, normalize, convert to tensor)
transform = transforms.Compose([
    transforms.Resize((256, 256)),  # Resizing to match the input size for SqueezeNet
    transforms.ToTensor(),  # Convert to PyTorch tensor
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # Normalization used by SqueezeNet
])

def extract_image_embedding(image):
    """
    Convert image to a feature vector using the SqueezeNet model.
    Args:
    - image: A PIL image object.

    Returns:
    - A NumPy array representing the image embedding (feature vector).
    """
    # Apply the image transformations (resize, tensor conversion, normalization)
    image_tensor = transform(image).unsqueeze(0)  # Add batch dimension (1, 3, 256, 256)

    # Get the image embeddings (forward pass through SqueezeNet)
    with torch.no_grad():
        embedding = squeezenet(image_tensor)

    # Convert the embedding tensor to a NumPy array and flatten
    embedding_np = embedding.cpu().numpy().flatten()

    return embedding_np

# Root route (for testing if the API is up)
@app.route('/')
def index():
    return "Dobrodošli v API za klasifikacijo rastlin. Uporabi /health, /species, ali /sickness endpointe za pridobivanje predikcij."

# Health classification route (accepting image)
@app.route('/health', methods=['POST'])
def classify_health():
    print("Received health classification request")  # Debug print
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400
    
    file = request.files['file']
    
    # Open the image with PIL
    image = Image.open(file.stream)
    
    # Extract image embeddings using SqueezeNet
    input_data = extract_image_embedding(image)
    
    # Predict the health status (returns a string label)
    health_prediction = health_model.predict([input_data])
    
    return jsonify({'health_status': health_prediction[0]})


# Species classification route (accepting image)
@app.route('/species', methods=['POST'])
def classify_species():
    print("Received species classification request")  # Debug print
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400
    
    file = request.files['file']
    
    # Open the image with PIL
    image = Image.open(file.stream)
    
    # Extract image embeddings using SqueezeNet
    input_data = extract_image_embedding(image)
    
    # Predict the species (returns a string label)
    species_prediction = species_model.predict([input_data])
    
    # Get confidence score (returns probabilities for each class)
    species_confidence = species_model.predict_proba([input_data])
    
    # Get the confidence for the predicted class and convert to percentage
    confidence_percentage = float(species_confidence.max()) * 100
    
    return jsonify({
        'species_name': species_prediction[0],        # This is the species label
        'confidence': confidence_percentage           # Confidence as a percentage
    })


# Sickness classification route (accepting image)
@app.route('/sickness', methods=['POST'])
def classify_sickness():
    print("Received disease classification request")  # Debug print
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400
    
    file = request.files['file']
    
    # Open the image with PIL
    image = Image.open(file.stream)
    
    # Extract image embeddings using SqueezeNet
    input_data = extract_image_embedding(image)
    
    # Predict the disease (returns a string label)
    sickness_prediction = sickness_model.predict([input_data])
    
    return jsonify({'disease': sickness_prediction[0]})


if __name__ == '__main__':
    app.run(debug=True)
