import Orange  # For handling data tables and machine learning models
import pickle  # For loading serialized models
from flask import Flask, request, jsonify  # For creating the Flask API
from PIL import Image  # For image processing
import torch  # For tensor computations
import torchvision.transforms as transforms  # For image transformations
from torchvision import models  # For pre-trained models
import numpy as np  # For numerical operations

app = Flask(__name__)

# Load the health classification model
with open('klasifikacija_zdravja_logistic_regression.pkcls', 'rb') as f:
    health_model = pickle.load(f)

# Load the species classification model
with open('klasifikacija_sort_logistic_regression.pkcls', 'rb') as f:
    species_model = pickle.load(f)

# Load all species-specific sickness models
with open('klasifikacija_grozdje_bolezni_logistic_regression.pkcls', 'rb') as f:
    grape_sickness_model = pickle.load(f)

with open('klasifikacija_jabolko_bolezni_logistic_regression.pkcls', 'rb') as f:
    apple_sickness_model = pickle.load(f)

with open('klasifikacija_jagoda_bolezni_logistic_regression.pkcls', 'rb') as f:
    strawberry_sickness_model = pickle.load(f)

with open('klasifikacija_koruza_bolezni_logistic_regression.pkcls', 'rb') as f:
    corn_sickness_model = pickle.load(f)

with open('klasifikacija_mango_bolezni_logistic_regression.pkcls', 'rb') as f:
    mango_sickness_model = pickle.load(f)

with open('klasifikacija_paradajz_bolezni_logistic_regression.pkcls', 'rb') as f:
    tomato_sickness_model = pickle.load(f)

# Extract class labels from models
health_class_labels = health_model.domain.class_var.values
species_class_labels = species_model.domain.class_var.values

# Load SqueezeNet for image embeddings
squeezenet = models.squeezenet1_1(weights='DEFAULT')
squeezenet.eval()
device = torch.device("cpu")
squeezenet = squeezenet.to(device)

# Transformations for input images
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# Function to extract image embeddings
def extract_image_embedding(image):
    image_tensor = transform(image).unsqueeze(0)
    image_tensor = image_tensor.to(device)
    with torch.no_grad():
        embedding = squeezenet(image_tensor)
    embedding_np = embedding.cpu().numpy().flatten()
    return embedding_np

# Convert embeddings to Orange data table
def convert_to_orange_table(embedding, model):
    domain = model.domain
    num_features = len(domain.attributes)
    X = embedding[:num_features].reshape(1, -1)
    Y = np.array([None])

    num_metas = len(domain.metas)
    if num_metas > 0:
        metas = np.array([None] * num_metas).reshape(1, -1)
    else:
        metas = None

    return Orange.data.Table.from_numpy(domain, X, Y, metas)

# Main page
@app.route('/')
def index():
    return "Welcome to the Plant Classification API. Use /health, /species, or /sickness endpoints."

# Health classification endpoint
@app.route('/health', methods=['POST'])
def classify_health():
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400

    file = request.files['file']
    image = Image.open(file.stream).convert('RGB')

    # Extract features
    embedding = extract_image_embedding(image)

    # Prepare data table
    table = convert_to_orange_table(embedding, health_model)

    # Predict health status
    health_prediction = health_model(table)[0]
    health_status = health_class_labels[int(health_prediction)]

    return jsonify({'health_status': health_status})

# Species classification endpoint
@app.route('/species', methods=['POST'])
def classify_species():
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400

    file = request.files['file']
    image = Image.open(file.stream).convert('RGB')

    # Extract features
    embedding = extract_image_embedding(image)

    # Prepare data table
    table = convert_to_orange_table(embedding, species_model)

    # Predict species
    species_prediction = species_model(table)[0]
    probabilities = species_model.predict_proba(table)[0]
    confidence = max(probabilities) * 100

    # Get species name
    species_name = species_class_labels[int(species_prediction)].split("/")[0]

    return jsonify({'species_name': species_name, 'confidence': confidence})

# Modified sickness classification endpoint
@app.route('/sickness', methods=['POST'])
def classify_sickness():
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400

    file = request.files['file']
    image = Image.open(file.stream).convert('RGB')

    # Step 1: Extract image embedding
    embedding = extract_image_embedding(image)

    # Step 2: Classify species
    species_table = convert_to_orange_table(embedding, species_model)
    species_prediction = species_model(species_table)[0]
    species_probabilities = species_model.predict_proba(species_table)[0]
    species_confidence = max(species_probabilities) * 100
    species_name = species_class_labels[int(species_prediction)].split("/")[0]

    # Normalize species name
    normalized_species_name = species_name.strip().lower()

    # Log the predicted species name
    print(f"Predicted species_name: '{normalized_species_name}'")

    # Step 3: Select the correct sickness model
    if normalized_species_name == "mango":
        sickness_model = mango_sickness_model
    elif normalized_species_name == "grozdje":
        sickness_model = grape_sickness_model
    elif normalized_species_name == "jabolko":
        sickness_model = apple_sickness_model
    elif normalized_species_name == "jagoda":
        sickness_model = strawberry_sickness_model
    elif normalized_species_name == "koruza":
        sickness_model = corn_sickness_model
    elif normalized_species_name == "paradajz":
        sickness_model = tomato_sickness_model
    else:
        return jsonify({'error': f"Unknown species: {normalized_species_name}"}), 400

    # Step 4: Classify sickness
    sickness_table = convert_to_orange_table(embedding, sickness_model)
    sickness_prediction = sickness_model(sickness_table)[0]
    sickness_probabilities = sickness_model.predict_proba(sickness_table)[0]
    sickness_confidence = max(sickness_probabilities) * 100

    # Get sickness name
    sickness_labels = sickness_model.domain.class_var.values
    sickness = sickness_labels[int(sickness_prediction)]

    return jsonify({
        'species_name': species_name,
        'species_confidence': species_confidence,
        'disease': sickness,
        'disease_confidence': sickness_confidence
    })


if __name__ == '__main__':
    app.run(debug=True)
