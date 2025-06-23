from flask import Flask, request, jsonify
import face_recognition
import numpy as np
import os
import pickle
from datetime import datetime
from scipy.spatial import distance

app = Flask(__name__)
# Rutas para almacenar caras conocidas y subidas dentro del directorio de la aplicación :D
# Le podes el nombre que quieras, pero no lo cambies en el código
KNOWN_FACES_DIR = 'known_faces'
UPLOADS_DIR = 'uploads'
trustValue = 0.6  # Valor de confianza para reconocimiento

os.makedirs(KNOWN_FACES_DIR, exist_ok=True)
os.makedirs(UPLOADS_DIR, exist_ok=True)

# Cargar todas las caras conocidas
def load_known_faces():
    known = {}
    for file in os.listdir(KNOWN_FACES_DIR):
        with open(os.path.join(KNOWN_FACES_DIR, file), 'rb') as f:
            known[file.replace(".pkl", "")] = pickle.load(f)
    return known

@app.route('/register', methods=['POST'])
def register():
    name = request.form['name']
    file = request.files['image']
    filepath = os.path.join(UPLOADS_DIR, file.filename)
    file.save(filepath)

    image = face_recognition.load_image_file(filepath)
    encodings = face_recognition.face_encodings(image)

    if len(encodings) == 0:
        return jsonify({"error": "No face found"}), 400

    encoding = encodings[0]
    with open(os.path.join(KNOWN_FACES_DIR, f"{name}.pkl"), 'wb') as f:
        pickle.dump(encoding, f)

    return jsonify({"message": f"{name} registered successfully"})

@app.route('/recognize', methods=['POST'])
def recognize():
    files = request.files.getlist("images")  # permite múltiples imágenes
    known_faces = load_known_faces()
    all_results = []

    for file in files:
        filepath = os.path.join(UPLOADS_DIR, file.filename)
        file.save(filepath)

        image = face_recognition.load_image_file(filepath)
        encodings = face_recognition.face_encodings(image)
        print(f"Image {file.filename} → Faces detected: {len(encodings)}")
        
        for i, encoding in enumerate(encodings):
            matched_name = "Unknown"
            min_dist = 1.0

            for name, known_encoding in known_faces.items():
                dist = distance.euclidean(known_encoding, encoding)
                if dist < trustValue and dist < min_dist:
                    matched_name = name
                    min_dist = dist

            # Calcular porcentaje de confianza
            confidence = max(0, min(1, 1 - min_dist / trustValue))
            confidence_percent = f"{round(confidence * 100)}%"

            all_results.append({
                "image": file.filename,
                "name": matched_name,
                "distance": [round(min_dist, 4), confidence_percent]
            })

    return jsonify({"recognized": all_results})


if __name__ == '__main__':
    app.run(debug=True)
