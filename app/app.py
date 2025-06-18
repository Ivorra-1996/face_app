from flask import Flask, request, jsonify
import face_recognition
import numpy as np
import os
import pickle
from datetime import datetime

app = Flask(__name__)
KNOWN_FACES_DIR = 'known_faces'
UPLOADS_DIR = 'uploads'

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
    file = request.files['image']
    filepath = os.path.join(UPLOADS_DIR, file.filename)
    file.save(filepath)

    image = face_recognition.load_image_file(filepath)
    encodings = face_recognition.face_encodings(image)

    if len(encodings) == 0:
        return jsonify({"error": "No face found"}), 400

    known_faces = load_known_faces()
    results = []

    for encoding in encodings:
        print(f"Recognizing face with encoding: {encoding}")
        matched_name = "Unknown"
        for name, known_encoding in known_faces.items():
            match = face_recognition.compare_faces([known_encoding], encoding, tolerance=0.6)
            if match[0]:
                matched_name = name
                break
        results.append(matched_name)

    return jsonify({"recognized": results})


if __name__ == '__main__':
    app.run(debug=True)
