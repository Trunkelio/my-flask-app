import Orange  # Uvozi Orange za delo s podatkovnimi tabelami in modeli strojnega učenja
import pickle  # Uporablja se za serializacijo in deserializacijo Python objektov (shranjevanje in nalaganje modelov)
from flask import Flask, request, jsonify  # Flask je mikro spletni framework za izdelavo API-jev
from PIL import Image  # Knjižnica za odpiranje in obdelavo slik
import torch  # PyTorch, odprtokodni framework za strojno učenje z uporabo tenzorjev
import torchvision.transforms as transforms  # Modul za transformacije slik iz knjižnice torchvision
from torchvision import models  # Vsebuje predtrenirane modele za različne naloge strojnega učenja
import numpy as np  # Knjižnica za delo z večdimenzionalnimi polji in matematičnimi operacijami

app = Flask(__name__)  # Inicializacija Flask aplikacije

# Nalaganje Orange modelov z uporabo pickle.load
with open('klasifikacija_zdravja_logistic_regression.pkcls', 'rb') as f:
    health_model = pickle.load(f)  # Naloži model za klasifikacijo zdravja rastlin

with open('klasifikacija_vrste_logistic_regression.pkcls', 'rb') as f:
    species_model = pickle.load(f)  # Naloži model za klasifikacijo vrst rastlin

with open('klasifikacija_bolezni_logistic_regression.pkcls', 'rb') as f:
    sickness_model = pickle.load(f)  # Naloži model za klasifikacijo bolezni rastlin

# Izvlečemo oznake razredov iz modelov
health_class_labels = health_model.domain.class_var.values  # Oznake za razrede zdravja (npr. 'Healthy', 'Sick')
species_class_labels = species_model.domain.class_var.values  # Oznake za vrste rastlin
sickness_class_labels = sickness_model.domain.class_var.values  # Oznake za bolezni rastlin

# Nalaganje predtreniranega modela SqueezeNet za ekstrakcijo značilk iz slik
squeezenet = models.squeezenet1_1(weights='DEFAULT')  # Uporabimo prednastavljene uteži modela
squeezenet.eval()  # Preklopimo model v evalvacijski način (izklopimo treniranje)
device = torch.device("cpu")  # Uporabimo CPU za računanje (lahko uporabite 'cuda' za GPU, če je na voljo)
squeezenet = squeezenet.to(device)  # Premaknemo model na izbrano napravo

# Definicija transformacij, ki jih uporabimo na vhodnih slikah
transform = transforms.Compose([
    # No more resizing here since the client will send the image already resized
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

    # Pridobimo ime vrste iz oznak razredov
    species_name = species_class_labels[int(species_prediction)]

    return jsonify({'species_name': species_name, 'confidence': confidence})  # Vrne rezultat kot JSON

# Other endpoints remain unchanged...

if __name__ == '__main__':
    app.run(debug=True)  # Zažene Flask aplikacijo v načinu za razhroščevanje
