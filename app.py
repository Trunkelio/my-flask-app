from flask import Flask, request, jsonify
import pickle
import numpy as np
from PIL import Image
import torch
import torchvision.transforms as transforms
from torchvision import models

app = Flask(__name__)

# Load all three models using .pkcls files
with open('klasifikacija_zdravja_logistic_regression.pkcls', 'rb') as f:
    health_model = pickle.load(f)

with open('klasifikacija_vrste_logistic_regression.pkcls', 'rb') as f:
    species_model = pickle.load(f)

with open('klasifikacija_bolezni_logistic_regression.pkcls', 'rb') as f:
    sickness_model = pickle.load(f)

# Load pre-trained SqueezeNet for image embedding (force to CPU)
squeezenet = models.squeezenet1_1(pretrained=True)
squeezenet.eval()  # Set to evaluation mode (no training)

# Force the model to CPU
device = torch.device("cpu")
squeezenet = squeezenet.to(device)

# Define image transformation pipeline (resize, normalize, convert to tensor)
transform = transforms.Compose([
    transforms.Resize((256, 256)),  # Resizing to match the input size for SqueezeNet
    transforms.ToTensor(),  # Convert to PyTorch tensor
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # Normalization used by SqueezeNet
])

# Label mappings for health, species, and sickness classifications
health_labels = {
    0: "Healthy",
    1: "Sick",
    2: "Healthy_Sick"
}

species_labels = {
    0: "Grozdje",
    1: "Jabolko",
    2: "Jagoda",
    3: "Koruza",
    4: "Mango",
    5: "Paradajz"
}

sickness_labels = {
    0: "Grozdje___Black_Measles",
    1: "Grozdje___Black_rot",
    2: "Grozdje___Isariopsis_Leaf_Spot",
    3: "Jabolko__Cedar_apple_rust",
    4: "Jabolko___Apple_scab",
    5: "Jabolko___Black_rot",
    6: "Jagoda_angular_leafspot",
    7: "Jagoda_gray_mold",
    8: "Jagoda___Leaf_scorch",
    9: "Koruza___Blight_in_corn_Leaf",
    10: "Koruza___Common_rust",
    11: "Koruza___Listne_pege",
    12: "Mango___Bakterijske_bolezni",
    13: "Mango___Glivične_bolezni",
    14: "Mango___Škodljivci",
    15: "Paradajz___Septoria_leaf_spot",
    16: "Paradajz___Target_Spot",
    17: "Paradajz___Virusne_bolezni"
}

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

    # Send tensor to CPU
    image_tensor = image_tensor.to(device)

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
    
    # Predict the health status
    health_prediction = health_model.predict([input_data])
    
    # Convert the predicted class index to a label
    health_status = health_labels.get(int(health_prediction[0]), "Unknown")
    
    return jsonify({'health_status': health_status})

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
    
    # Predict the species
    species_prediction = species_model.predict([input_data])
    
    # Get confidence score
    species_confidence = species_model.predict_proba([input_data])
    confidence_percentage = round(float(species_confidence.max()) * 100, 2)
    
    # Convert the predicted class index to a label
    species_name = species_labels.get(int(species_prediction[0]), "Unknown")
    
    return jsonify({
        'species_name': species_name,
        'confidence': confidence_percentage
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
    
    # Predict the disease
    sickness_prediction = sickness_model.predict([input_data])
    
    # Convert the predicted class index to a label
    disease = sickness_labels.get(int(sickness_prediction[0]), "Unknown")
    
    return jsonify({'disease': disease})

if __name__ == '__main__':
    app.run(debug=True)
