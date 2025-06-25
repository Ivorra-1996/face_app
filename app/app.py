import cv2
import face_recognition
from flask import Flask, request, jsonify , Response
import pickle
import os
from scipy.spatial import distance

app = Flask(__name__)

KNOWN_FACES_DIR = 'known_faces'
UPLOADS_DIR = 'uploads'
trustValue = 0.6

def load_known_faces():
    known = {}
    for file in os.listdir(KNOWN_FACES_DIR):
        with open(os.path.join(KNOWN_FACES_DIR, file), 'rb') as f:
            known[file.replace(".pkl", "")] = pickle.load(f)
    return known

def recognize_faces_in_frame(frame, known_faces):
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    locations = face_recognition.face_locations(rgb_frame)
    encodings = face_recognition.face_encodings(rgb_frame, locations)
    names = []

    for encoding in encodings:
        matched_name = "Unknown"
        min_dist = 1.0
        for name, known_encoding in known_faces.items():
            dist = distance.euclidean(known_encoding, encoding)
            if dist < trustValue and dist < min_dist:
                matched_name = name
                min_dist = dist
        names.append(matched_name)

    return locations, names

def generate_frames():
    known_faces = load_known_faces()
    cap = cv2.VideoCapture(1)  # Cambiar al índice de tu cámara o url

    if not cap.isOpened():
        print("No se pudo abrir la cámara")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        locations, names = recognize_faces_in_frame(frame, known_faces)

        for (top, right, bottom, left), name in zip(locations, names):
            # Dibujar rectángulo
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
            # Texto nombre
            print(f"Nombre detectado: {name}")
            cv2.putText(frame, name, (250, 460), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255))


        # Codificar frame a JPEG
        ret, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()

        # Stream multipart MJPEG para Flask
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

    cap.release()

@app.route('/register_from_frame', methods=['POST'])
def register_from_frame():
    name = request.form['name']
    lastname = request.form['lastname']

    # Capturar un frame desde la cámara
    cap = cv2.VideoCapture(1)
    ret, frame = cap.read()
    cap.release()

    if not ret:
        return jsonify({"error": "No se pudo capturar imagen desde la cámara"}), 500

    # Guardar temporalmente el frame como imagen (opcional)
    filename = f"{name}_{lastname}.jpg"
    filepath = os.path.join(UPLOADS_DIR, filename)
    cv2.imwrite(filepath, frame)

    # Extraer encoding facial
    image = face_recognition.load_image_file(filepath)
    encodings = face_recognition.face_encodings(image)

    if len(encodings) == 0:
        return jsonify({"error": "No face found"}), 400

    encoding = encodings[0]
    with open(os.path.join(KNOWN_FACES_DIR, f"{name} {lastname}.pkl"), 'wb') as f:
        pickle.dump(encoding, f)

    return jsonify({"message": f"{name} {lastname} registered successfully from camera"})


@app.route('/video_feed', methods=['GET'])
def video_feed():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(debug=True)
