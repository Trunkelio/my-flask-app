import Orange  # Uvozi Orange za delo s podatkovnimi tabelami in modeli
from flask import Flask, request, jsonify
from PIL import Image
import torch
import torchvision.transforms as transforms
from torchvision import models
import numpy as np

app = Flask(__name__)

# Nalaganje modelov Orange s pomočjo Orange loaderja za .pkcls datoteke
# Uporabimo Orange.misc.pickle_load, ker modeli niso združljivi z običajnim pickle
health_model = Orange.misc.pickle_load(open('klasifikacija_zdravja_logistic_regression.pkcls', 'rb'))
species_model = Orange.misc.pickle_load(open('klasifikacija_vrste_logistic_regression.pkcls', 'rb'))
sickness_model = Orange.misc.pickle_load(open('klasifikacija_bolezni_logistic_regression.pkcls', 'rb'))

# Nalaganje pred-učenega modela SqueezeNet za pridobivanje vektorskih značilnosti iz slik
# Model SqueezeNet se uporablja za pretvorbo slik v numerične vektorske značilnosti
squeezenet = models.squeezenet1_1(pretrained=True)
squeezenet.eval()  # Preklopi model v način evalvacije, da se izklopi treniranje
device = torch.device("cpu")  # Uporabimo CPU, ker je dovolj za obdelavo
squeezenet = squeezenet.to(device)

# Definiramo transformacije, ki bodo uporabljene na sliki (npr. sprememba velikosti, normalizacija)
# To zagotovi, da je vhod v ustrezni obliki za model SqueezeNet
transform = transforms.Compose([
    transforms.Resize((256, 256)),  # Spremeni velikost slike na 256x256
    transforms.ToTensor(),  # Pretvori sliko v PyTorch tenzor
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # Normaliziraj sliko
])

# Funkcija za pridobivanje vektorskih značilnosti iz slike s pomočjo SqueezeNet
def extract_image_embedding(image):
    """Pretvori sliko v vektorske značilnosti z modelom SqueezeNet."""
    image_tensor = transform(image).unsqueeze(0)  # Doda dimenzijo paketa (za batch obdelavo)
    image_tensor = image_tensor.to(device)  # Pošlji tenzor na CPU
    with torch.no_grad():  # Izklopi računanje gradientov za hitrejšo napoved
        embedding = squeezenet(image_tensor)  # Dobimo vektorske značilnosti slike
    embedding_np = embedding.cpu().numpy().flatten()  # Pretvori tenzor v NumPy matriko
    return embedding_np  # Vrnemo vektor značilnosti (embedding)

# Funkcija za pretvorbo vektorja značilnosti v Orange data table format
# Orange modeli pričakujejo podatke v obliki Orange.data.Table, kar vključuje tudi domene (metapodatki o lastnostih)
def convert_to_orange_table(embedding):
    """Pretvori NumPy vektor značilnosti v Orange.data.Table za združljivost z modeli."""
    domain = health_model.domain  # Uporabi domeno modela, da zagotovimo ustrezno strukturo podatkov
    table = Orange.data.Table(domain, [embedding])  # Ustvari tabelo z vektorjem značilnosti
    return table  # Vrnemo tabelo v ustreznem formatu za model

# Glavna stran, ki vrne osnovno sporočilo za preverjanje, če API deluje
@app.route('/')
def index():
    return "Dobrodošli v API za klasifikacijo rastlin. Uporabi /health, /species, ali /sickness endpointe za pridobivanje predikcij."

# Endpoint za klasifikacijo zdravja (prejme sliko in vrne zdravje rastline)
@app.route('/health', methods=['POST'])
def classify_health():
    if 'file' not in request.files:  # Preveri, ali je bila slika poslana v zahtevi
        return jsonify({'error': 'Nobena datoteka ni bila poslana'}), 400
    
    file = request.files['file']  # Dobimo datoteko iz zahteve
    image = Image.open(file.stream)  # Odpri sliko s pomočjo PIL
    
    # Pridobi značilnosti slike
    embedding = extract_image_embedding(image)
    
    # Pretvori vektorske značilnosti v Orange data table
    table = convert_to_orange_table(embedding)
    
    # Napoved zdravja s pomočjo modela Orange
    health_prediction = health_model(table)[0]  # Napoved z uporabo modela
    health_status = "Healthy" if health_prediction == 0 else "Sick"  # Razlaga napovedi
    
    return jsonify({'health_status': health_status})  # Vrni rezultat kot JSON

# Endpoint za klasifikacijo vrste rastline (prejme sliko in vrne vrsto rastline)
@app.route('/species', methods=['POST'])
def classify_species():
    if 'file' not in request.files:
        return jsonify({'error': 'Nobena datoteka ni bila poslana'}), 400
    
    file = request.files['file']
    image = Image.open(file.stream)
    
    # Pridobi značilnosti slike
    embedding = extract_image_embedding(image)
    
    # Pretvori v Orange tabelo
    table = convert_to_orange_table(embedding)
    
    # Napoved vrste rastline
    species_prediction = species_model(table)[0]  # Napoved z uporabo modela Orange
    confidence = max(species_model(table, ret="Probabilities")[0]) * 100  # Verjetnost napovedi
    
    # Mapa oznak vrst
    species_labels = {
        0: "Grozdje",
        1: "Jabolko",
        2: "Jagoda",
        3: "Koruza",
        4: "Mango",
        5: "Paradajz"
    }
    species_name = species_labels.get(int(species_prediction), "Unknown")  # Pridobi ime vrste iz napovedi
    
    return jsonify({'species_name': species_name, 'confidence': confidence})  # Vrni vrsto in natančnost napovedi

# Endpoint za klasifikacijo bolezni rastline (prejme sliko in vrne bolezen)
@app.route('/sickness', methods=['POST'])
def classify_sickness():
    if 'file' not in request.files:
        return jsonify({'error': 'Nobena datoteka ni bila poslana'}), 400
    
    file = request.files['file']
    image = Image.open(file.stream)
    
    # Pridobi značilnosti slike
    embedding = extract_image_embedding(image)
    
    # Pretvori v Orange tabelo
    table = convert_to_orange_table(embedding)
    
    # Napoved bolezni z modelom Orange
    sickness_prediction = sickness_model(table)[0]  # Napoved bolezni
    
    # Mapa oznak bolezni
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
    disease = sickness_labels.get(int(sickness_prediction), "Unknown")  # Pridobi ime bolezni
    
    return jsonify({'disease': disease})  # Vrni napovedano bolezen

if __name__ == '__main__':
    app.run(debug=True)  # Zagon aplikacije Flask
