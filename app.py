from flask import Flask, request, jsonify
import pickle
import numpy as np
from PIL import Image
import torch
import torchvision.transforms as transforms
from torchvision import models

app = Flask(__name__)

# Naloži vse tri modele s pomočjo .pkcls datotek
with open('klasifikacija_zdravja_logistic_regression.pkcls', 'rb') as f:
    health_model = pickle.load(f)

with open('klasifikacija_vrste_logistic_regression.pkcls', 'rb') as f:
    species_model = pickle.load(f)

with open('klasifikacija_bolezni_logistic_regression.pkcls', 'rb') as f:
    sickness_model = pickle.load(f)

# Naloži pred-učen model SqueezeNet za pretvorbo slik v vektorske značilnosti (na CPU)
squeezenet = models.squeezenet1_1(pretrained=True)
squeezenet.eval()  # Preklopi model v način za evaluacijo (izklopi treniranje)

# Zakaj uporabiti CPU namesto GPU:
# V tej aplikaciji uporabljamo CPU, ker obdelava slik ni zelo zahtevna ali časovno kritična,
# aplikacija pa verjetno ne bo delovala na sistemu z GPU (npr. strežniki brez grafičnih kartic).
# Z uporabo CPU zagotovimo, da bo aplikacija delovala na širši paleti naprav.

# Uporabi CPU za obdelavo (namesto GPU)
device = torch.device("cpu")
squeezenet = squeezenet.to(device)

# Kaj naredi squeezenet.eval():
# Funkcija eval() preklopi model v evalvacijski način, kar pomeni, da se mreža ne trenira.
# To onemogoči funkcije, kot je dropout, ki se uporabljajo med treniranjem, in zagotovi
# pravilno obnašanje modela med napovedovanjem.

# Definiraj transformacije slike (spremeni velikost, normaliziraj, pretvori v tenzor)
transform = transforms.Compose([
    transforms.Resize((256, 256)),  # Spremeni velikost slike, da ustreza vhodu za SqueezeNet
    transforms.ToTensor(),  # Pretvori sliko v PyTorch tenzor
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # Normalizacija, uporabljena v SqueezeNet
])

# Popravljene mape oznak za klasifikacijo zdravja, vrste in bolezni
health_labels = {
    0: "Healthy",  # Zdrav
    1: "Sick"  # Bolehen
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
    Pretvori sliko v vektorske značilnosti s pomočjo modela SqueezeNet.
    Argumenti:
    - image: Objekt PIL slike.

    Vrne:
    - NumPy matrika, ki predstavlja značilnosti slike (vektor značilnosti).
    """
    # Uporabi transformacije slike (spremeni velikost, pretvorba v tenzor, normalizacija)
    image_tensor = transform(image).unsqueeze(0)  # Dodaj dimenzijo paketa (1, 3, 256, 256)

    # Pošlji tenzor na CPU
    image_tensor = image_tensor.to(device)

    # Dobimo značilnosti slike (naprej skozi SqueezeNet)
    with torch.no_grad():
        embedding = squeezenet(image_tensor)

    # Pretvori tenzor značilnosti v NumPy matriko in splošči
    embedding_np = embedding.cpu().numpy().flatten()

    return embedding_np

# Glavna stran (za preverjanje, če API deluje)
@app.route('/')
def index():
    return "Dobrodošli v API za klasifikacijo rastlin. Uporabi /health, /species, ali /sickness endpointe za pridobivanje predikcij."

# Pot za klasifikacijo zdravja (sprejema sliko)
@app.route('/health', methods=['POST'])
def classify_health():
    print("Prejeta zahteva za klasifikacijo zdravja")  # Natisni za debug
    if 'file' not in request.files:
        return jsonify({'error': 'Nobena datoteka ni bila poslana'}), 400
    
    file = request.files['file']
    
    # Odpri sliko s pomočjo PIL
    image = Image.open(file.stream)
    
    # Izvleči značilnosti slike s pomočjo SqueezeNet
    input_data = extract_image_embedding(image)
    
    # Napovej zdravstveno stanje
    health_prediction = health_model.predict([input_data])
    
    # Pretvori predviden indeks v oznako
    health_status = health_labels.get(int(health_prediction[0]), "Unknown")
    
    return jsonify({'health_status': health_status})

# Pot za klasifikacijo vrste rastline (sprejema sliko)
@app.route('/species', methods=['POST'])
def classify_species():
    print("Prejeta zahteva za klasifikacijo vrste rastline")  # Natisni za debug
    if 'file' not in request.files:
        return jsonify({'error': 'Nobena datoteka ni bila poslana'}), 400
    
    file = request.files['file']
    
    # Odpri sliko s pomočjo PIL
    image = Image.open(file.stream)
    
    # Izvleči značilnosti slike s pomočjo SqueezeNet
    input_data = extract_image_embedding(image)
    
    # Napovej vrsto rastline
    species_prediction = species_model.predict([input_data])
    
    # Dobimo stopnjo zaupanja v napoved
    species_confidence = species_model.predict_proba([input_data])
    confidence_percentage = round(float(species_confidence.max()) * 100, 2)
    
    # Pretvori predviden indeks v oznako
    species_name = species_labels.get(int(species_prediction[0]), "Unknown")
    
    return jsonify({
        'species_name': species_name,
        'confidence': confidence_percentage
    })

# Pot za klasifikacijo bolezni rastline (sprejema sliko)
@app.route('/sickness', methods=['POST'])
def classify_sickness():
    print("Prejeta zahteva za klasifikacijo bolezni rastline")  # Natisni za debug
    if 'file' not in request.files:
        return jsonify({'error': 'Nobena datoteka ni bila poslana'}), 400
    
    file = request.files['file']
    
    # Odpri sliko s pomočjo PIL
    image = Image.open(file.stream)
    
    # Izvleči značilnosti slike s pomočjo SqueezeNet
    input_data = extract_image_embedding(image)
    
    # Napovej bolezen rastline
    sickness_prediction = sickness_model.predict([input_data])
    
    # Pretvori predviden indeks v oznako bolezni
    disease = sickness_labels.get(int(sickness_prediction[0]), "Unknown")
    
    return jsonify({'disease': disease})

if __name__ == '__main__':
    app.run(debug=True)
