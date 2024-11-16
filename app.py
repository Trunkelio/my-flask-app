import Orange  # Za delo z podatkovnimi tabelami in modeli strojnega učenja
import pickle  # Za nalaganje serializiranih modelov
from flask import Flask, request, jsonify  # Za ustvarjanje Flask API-ja
from PIL import Image  # Za obdelavo slik
import torch  # Za delo s tenzorji
import torchvision.transforms as transforms  # Za transformacije slik
from torchvision import models  # Za uporabo predtreniranih modelov
import numpy as np  # Za numerične operacije

app = Flask(__name__)

# Naloži model za klasifikacijo zdravja rastlin
with open('klasifikacija_zdravja_logistic_regression.pkcls', 'rb') as f:
    health_model = pickle.load(f)

# Naloži model za klasifikacijo vrst rastlin
with open('klasifikacija_sort_logistic_regression.pkcls', 'rb') as f:
    species_model = pickle.load(f)

# Naloži modele za bolezni, specifične za posamezne vrste
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

# Izvleci oznake razredov iz modelov
health_class_labels = health_model.domain.class_var.values
species_class_labels = species_model.domain.class_var.values

# Naloži SqueezeNet za pridobivanje značilk iz slik (embedding)
squeezenet = models.squeezenet1_1(weights='DEFAULT')
squeezenet.eval()  # Nastavi model v način ocenjevanja
device = torch.device("cpu")  # Uporabi CPU za izračune
squeezenet = squeezenet.to(device)

# Definiraj transformacije za vhodne slike
transform = transforms.Compose([
    transforms.Resize((256, 256)),  # Spremeni velikost slike na 256x256 pikslov
    transforms.ToTensor(),  # Pretvori sliko v tenzor
    transforms.Normalize([0.485, 0.456, 0.406],  # Normaliziraj srednje vrednosti kanalov
                         [0.229, 0.224, 0.225])  # Normaliziraj standardne odklone kanalov
])

# Funkcija za pridobivanje vektorske predstavitve slike
def extract_image_embedding(image):
    image_tensor = transform(image).unsqueeze(0)  # Uporabi transformacije in dodaj dimenzijo za batch
    image_tensor = image_tensor.to(device)
    with torch.no_grad():
        embedding = squeezenet(image_tensor)  # Pridobi vektorsko predstavitev iz modela
    embedding_np = embedding.cpu().numpy().flatten()  # Pretvori v numpy array in splošči
    return embedding_np

# Pretvori embedding v Orange podatkovno tabelo
def convert_to_orange_table(embedding, model):
    domain = model.domain
    num_features = len(domain.attributes)
    X = embedding[:num_features].reshape(1, -1)  # Uporabi ustrezno število značilk
    Y = np.array([None])  # Dummy ciljna spremenljivka

    num_metas = len(domain.metas)
    if num_metas > 0:
        metas = np.array([None] * num_metas).reshape(1, -1)
    else:
        metas = None

    return Orange.data.Table.from_numpy(domain, X, Y, metas)

# Glavna stran API-ja
@app.route('/')
def index():
    return "Dobrodošli v Plant Analizator API. Uporabite končne točke /health, /species, /sickness."

# Končna točka za klasifikacijo zdravja
@app.route('/health', methods=['POST'])
def classify_health():
    if 'file' not in request.files:
        return jsonify({'error': 'Ni predložene datoteke'}), 400

    file = request.files['file']
    image = Image.open(file.stream).convert('RGB')

    # Pridobi vektorsko predstavitev slike
    embedding = extract_image_embedding(image)

    # Pripravi podatkovno tabelo za model
    table = convert_to_orange_table(embedding, health_model)

    # Napovej zdravstveno stanje
    health_prediction = health_model(table)[0]
    health_status = health_class_labels[int(health_prediction)]

    return jsonify({'health_status': health_status})

# Končna točka za klasifikacijo vrste rastline
@app.route('/species', methods=['POST'])
def classify_species():
    if 'file' not in request.files:
        return jsonify({'error': 'Ni predložene datoteke'}), 400

    file = request.files['file']
    image = Image.open(file.stream).convert('RGB')

    # Pridobi vektorsko predstavitev slike
    embedding = extract_image_embedding(image)

    # Pripravi podatkovno tabelo za model
    table = convert_to_orange_table(embedding, species_model)

    # Napovej vrsto rastline
    species_prediction = species_model(table)[0]
    probabilities = species_model.predict_proba(table)[0]
    confidence = max(probabilities) * 100  # Izračunaj stopnjo zaupanja

    # Pridobi ime vrste
    species_name = species_class_labels[int(species_prediction)].split("/")[0]

    return jsonify({'species_name': species_name, 'confidence': confidence})

# Končna točka za klasifikacijo bolezni
@app.route('/sickness', methods=['POST'])
def classify_sickness():
    if 'file' not in request.files:
        return jsonify({'error': 'Ni predložene datoteke'}), 400

    file = request.files['file']
    image = Image.open(file.stream).convert('RGB')

    # Korak 1: Pridobi vektorsko predstavitev slike
    embedding = extract_image_embedding(image)

    # Korak 2: Klasificiraj vrsto rastline
    species_table = convert_to_orange_table(embedding, species_model)
    species_prediction = species_model(species_table)[0]
    species_probabilities = species_model.predict_proba(species_table)[0]
    species_confidence = max(species_probabilities) * 100
    species_name = species_class_labels[int(species_prediction)].split("/")[0]

    # Normaliziraj ime vrste za primerjavo
    normalized_species_name = species_name.strip().lower()

    # Logiraj napovedano vrsto
    print(f"Predicted species_name: '{normalized_species_name}'")

    # Korak 3: Izberi ustrezen model za bolezen glede na vrsto
    if normalized_species_name == "mango":
        sickness_model = mango_sickness_model
        print(f"Uporabljen model: mango_sickness_model")
    elif normalized_species_name == "grozdje":
        sickness_model = grape_sickness_model
        print(f"Uporabljen model: grozdje_sickness_model")
    elif normalized_species_name == "jabolko":
        sickness_model = apple_sickness_model
        print(f"Uporabljen model: jabolko_sickness_model")
    elif normalized_species_name == "jagoda":
        sickness_model = strawberry_sickness_model
        print(f"Uporabljen model: jagoda_sickness_model")
    elif normalized_species_name == "koruza":
        sickness_model = corn_sickness_model
        print(f"Uporabljen model: koruza_sickness_model")
    elif normalized_species_name == "paradajz":
        sickness_model = tomato_sickness_model
        print(f"Uporabljen model: paradajz_sickness_model")
    else:
        return jsonify({'error': f"Neznana vrsta: {normalized_species_name}"}), 400

    # Korak 4: Klasificiraj bolezen
    sickness_table = convert_to_orange_table(embedding, sickness_model)
    sickness_prediction = sickness_model(sickness_table)[0]
    sickness_probabilities = sickness_model.predict_proba(sickness_table)[0]
    sickness_confidence = max(sickness_probabilities) * 100

    # Pridobi ime bolezni
    sickness_labels = sickness_model.domain.class_var.values
    sickness = sickness_labels[int(sickness_prediction)]

    return jsonify({
        'species_name': species_name,
        'species_confidence': species_confidence,
        'disease': sickness,
        'disease_confidence': sickness_confidence
    })

# Zaženi aplikacijo
if __name__ == '__main__':
    app.run(debug=True)
