import Orange  # Uvozi Orange za delo s podatkovnimi tabelami in modeli strojnega učenja
import pickle  # Uporablja se za serializacijo in deserializacijo Python objektov (shranjevanje in nalaganje modelov)
from flask import Flask, request, jsonify  # Flask je mikro spletni framework za izdelavo API-jev
from PIL import Image  # Knjižnica za odpiranje in obdelavo slik
import torch  # PyTorch, odprtokodni framework za strojno učenje z uporabo tenzorjev
import torchvision.transforms as transforms  # Modul za transformacije slik iz knjižnice torchvision
from torchvision import models  # Vsebuje predtrenirane modele za različne naloge strojnega učenja
import numpy as np  # Knjižnica za delo z večdimenzionalnimi polji in matematičnimi operacijami

app = Flask(__name__)  # Inicializacija Flask aplikacije

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
health_class_labels = health_model.domain.class_var.values  # Oznake za razrede zdravja (npr. 'Healthy', 'Sick')
species_class_labels = species_model.domain.class_var.values  # Oznake za vrste rastlin

# Load SqueezeNet for image embeddings
squeezenet = models.squeezenet1_1(weights='DEFAULT')  # Uporabimo prednastavljene uteži modela
squeezenet.eval()  # Preklopimo model v evalvacijski način (izklopimo treniranje)
device = torch.device("cpu")  # Uporabimo CPU za računanje (lahko uporabite 'cuda' za GPU, če je na voljo)
squeezenet = squeezenet.to(device)  # Premaknemo model na izbrano napravo

# Transformacije slik za model SqueezeNet
transform = transforms.Compose([
    transforms.Resize((256, 256)),  # Velikost slike 256x256
    transforms.ToTensor(),  # Pretvorimo sliko v tenzor PyTorch
    transforms.Normalize([0.485, 0.456, 0.406],  # Normaliziramo sliko z uporabo povprečij...
                         [0.229, 0.224, 0.225])  # ...in standardnih deviacij za vsak barvni kanal (RGB)
])

# Funkcija za ekstrakcijo značilk iz slike z uporabo modela SqueezeNet
def extract_image_embedding(image):
    image_tensor = transform(image).unsqueeze(0)  # Uporabimo transformacije in dodamo dimenzijo za batch
    image_tensor = image_tensor.to(device)  # Premaknemo tenzor na izbrano napravo (CPU ali GPU)
    with torch.no_grad():  # Izklopimo sledenje gradientov za hitrejše računanje
        embedding = squeezenet(image_tensor)  # Preženemo sliko skozi model in pridobimo značilke
    embedding_np = embedding.cpu().numpy().flatten()  # Pretvorimo tenzor v NumPy polje in ga sploščimo
    return embedding_np  # Vrnemo vektor značilk

# Convert image embedding to Orange data table
def convert_to_orange_table(embedding, model):
    """Pretvori NumPy vektor značilk v Orange podatkovno tabelo."""
    domain = model.domain  # Pridobimo domeno modela (vsebuje informacije o atributih, razredih in meta-podatkih)
    num_features = len(domain.attributes)  # Število atributov, ki jih model pričakuje
    X = embedding[:num_features].reshape(1, -1)  # Prilagodimo vektor značilk velikosti, ki jo model pričakuje
    Y = np.array([None])  # Ustvarimo placeholder za vrednost razreda (ni znana pri napovedovanju)

    # Priprava meta-atributov, če obstajajo
    num_metas = len(domain.metas)  # Število meta-atributov v domeni
    if num_metas > 0:
        metas = np.array([None] * num_metas).reshape(1, -1)  # Ustvarimo placeholder za meta-atribute
    else:
        metas = None  # Če ni meta-atributov, nastavimo na None

    return Orange.data.Table.from_numpy(domain, X, Y, metas)  # Ustvarimo Orange podatkovno tabelo

# Glavna stran API-ja (dobrodošlica)
@app.route('/')
def index():
    return "Dobrodošli v API za klasifikacijo rastlin. Izberite ustrezne endpointe za dostop do podatkov: /health, /species, ali /sickness."

# Endpoint za klasifikacijo zdravja rastline
@app.route('/health', methods=['POST'])
def classify_health():
    if 'file' not in request.files:
        return jsonify({'error': 'Nobena datoteka ni bila poslana'}), 400  # Vrne napako, če ni priložene datoteke

    file = request.files['file']
    image = Image.open(file.stream).convert('RGB')  # Odpre sliko iz zahteve in jo pretvori v RGB format

    # Ekstrakcija značilk iz slike
    embedding = extract_image_embedding(image)

    # Pretvorba v Orange podatkovno tabelo z uporabo modela za zdravje
    table = convert_to_orange_table(embedding, health_model)

    # Napoved zdravja rastline
    health_prediction = health_model(table)[0]  # Pridobimo napovedani razred
    health_status = health_class_labels[int(health_prediction)]  # Pridobimo ime razreda iz oznak

    return jsonify({'health_status': health_status})  # Vrne rezultat kot JSON odgovor

# Endpoint za klasifikacijo vrste rastline
@app.route('/species', methods=['POST'])
def classify_species():
    if 'file' not in request.files:
        return jsonify({'error': 'Nobena datoteka ni bila poslana'}), 400

    file = request.files['file']
    image = Image.open(file.stream).convert('RGB')

    # Ekstrakcija značilk iz slike
    embedding = extract_image_embedding(image)

    # Pretvorba v Orange podatkovno tabelo z uporabo modela za vrste
    table = convert_to_orange_table(embedding, species_model)

    # Napoved vrste rastline
    species_prediction = species_model(table)[0]
    probabilities = species_model.predict_proba(table)[0]  # Dobimo verjetnosti
    confidence = max(probabilities) * 100  # Izračunamo zaupanje v napoved (v odstotkih)

    # Pridobimo ime vrste iz oznak razredov, keeping only the part before "/"
    species_name = species_class_labels[int(species_prediction)].split("/")[0]

    return jsonify({'species_name': species_name, 'confidence': confidence})  # Vrne rezultat kot JSON

# Endpoint za klasifikacijo bolezni glede na vrsto rastline
@app.route('/sickness', methods=['POST'])
def classify_sickness():
    if 'file' not in request.files or 'species_name' not in request.form:
        return jsonify({'error': 'Nobena datoteka ali vrsta rastline ni bila poslana'}), 400

    file = request.files['file']
    species_name = request.form['species_name']
    image = Image.open(file.stream).convert('RGB')

    # Step 2: Select the correct sickness model based on the provided species
    if species_name == "Mango":
        sickness_model = mango_sickness_model
    elif species_name == "Grozdje":
        sickness_model = grape_sickness_model
    elif species_name == "Jabolko":
        sickness_model = apple_sickness_model
    elif species_name == "Jagoda":
        sickness_model = strawberry_sickness_model
    elif species_name == "Koruza":
        sickness_model = corn_sickness_model
    elif species_name == "Paradižnik":
        sickness_model = tomato_sickness_model
    else:
        return jsonify({'error': f"Unknown species: {species_name}"}), 400

    # Step 3: Classify the sickness for the identified species
    embedding = extract_image_embedding(image)
    sickness_table = convert_to_orange_table(embedding, sickness_model)
    sickness_prediction = sickness_model(sickness_table)[0]
    sickness_probabilities = sickness_model.predict_proba(sickness_table)[0]
    confidence = max(sickness_probabilities) * 100  # Get the highest confidence

    # Get the predicted sickness
    sickness_labels = sickness_model.domain.class_var.values
    sickness = sickness_labels[int(sickness_prediction)]

    return jsonify({'species_name': species_name, 'disease': sickness, 'confidence': confidence})

if __name__ == '__main__':
    app.run(debug=True)  # Start Flask app
