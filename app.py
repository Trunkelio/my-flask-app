from flask import Flask, request, jsonify
import pickle
import numpy as np

app = Flask(__name__)

# Load all three models using .pkcls files (or .pa files)
with open('klasifikacija_zdravja_logistic_regression.pkcls', 'rb') as f:
    health_model = pickle.load(f)

with open('klasifikacija_vrste_logistic_regression.pkcls', 'rb') as f:
    species_model = pickle.load(f)

with open('klasifikacija_bolezni_logistic_regression.pkcls', 'rb') as f:
    sickness_model = pickle.load(f)

# Root route (for testing if the API is up)
@app.route('/')
def index():
    return "Dobrodošli v API za klasifikacijo rastlin. Uporabi /health, /species, ali /sickness endpointe za pridobivanje predikcij."

# Health classification route
@app.route('/health', methods=['POST'])
def classify_health():
    data = request.json
    input_data = np.array(data['features']).reshape(1, -1)
    
    # Predict if the plant is healthy or sick
    health_prediction = health_model.predict(input_data)
    
    return jsonify({'health_status': int(health_prediction[0])})

# Species classification route
@app.route('/species', methods=['POST'])
def classify_species():
    data = request.json
    input_data = np.array(data['features']).reshape(1, -1)
    
    # Predict the species and return the confidence score
    species_prediction = species_model.predict(input_data)
    species_confidence = species_model.predict_proba(input_data)
    
    species_name = int(species_prediction[0])
    return jsonify({
        'species_name': species_name,
        'confidence': species_confidence.max() * 100
    })

# Sickness classification route
@app.route('/sickness', methods=['POST'])
def classify_sickness():
    data = request.json
    input_data = np.array(data['features']).reshape(1, -1)
    
    # Predict the disease affecting the plant
    sickness_prediction = sickness_model.predict(input_data)
    
    return jsonify({'disease': int(sickness_prediction[0])})

if __name__ == '__main__':
    app.run(debug=True)
