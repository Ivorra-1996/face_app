from flask import Flask, request, jsonify
import face_recognition
import numpy as np
import os
import pickle
from datetime import datetime
from scipy.spatial import distance
import cv2
import threading
import time

# NO ES LO IDEAL COMENTAR CADA LINEA, PERO ES UNA BUENA PRÁCTICA PARA ENTENDER EL CÓDIGO. PORQUE DESPUES NO TE ACORDÁS QUE HACE CADA COSA. :D

app = Flask(__name__)
# Rutas para almacenar caras conocidas y subidas dentro del directorio de la aplicación :D
# Le podes el nombre que quieras, pero no lo cambies en el código
KNOWN_FACES_DIR = 'known_faces'
UPLOADS_DIR = 'uploads'
trustValue = 0.6  # Valor de confianza para reconocimiento

# Variables globales
frame = None
ret = False
stop_thread = False

os.makedirs(KNOWN_FACES_DIR, exist_ok=True)
os.makedirs(UPLOADS_DIR, exist_ok=True)

def configure_camera(cap):
    # Configurar la cámara si es necesario
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)
    print("Cámara configurada: 640x480 a 30 FPS")

def camera_reader(cap):
    """Lee frames de la cámara en un hilo separado."""
    global frame, ret, stop_thread
    while not stop_thread:
        try:
            ret, frame = cap.read()
            time.sleep(0.01)  # Pequeño retraso para reducir carga de CPU
        except Exception as e:
            print(f"Error al leer la cámara: {e}")
            stop_thread = True


# Cargar todas las caras conocidas .pkl
def load_known_faces():
    known = {}
    for file in os.listdir(KNOWN_FACES_DIR):
        with open(os.path.join(KNOWN_FACES_DIR, file), 'rb') as f:
            known[file.replace(".pkl", "")] = pickle.load(f)
    return known


# Esto no se usa todavia
def run_inference():
    print("Iniciando captura de video...")
    cap = cv2.VideoCapture(CAMERA_URL)
    if not cap.isOpened():
        print("Error: No se puede conectar a la cámara.")
        return

    configure_camera(cap)  # Configura la cámara con los parámetros

    # Inicia el hilo de lectura de la cámara
    camera_thread = threading.Thread(target=camera_reader, args=(cap,))
    camera_thread.start()

    print("Presiona 'q' para salir.")
    try:
        prev_time = time.time()  # Controla el tiempo entre inferencias
        while True:
            if not ret:
                continue  # Si no hay frame válido, sigue esperando

            # Limita la tasa de inferencia
            current_time = time.time()
            if current_time - prev_time >= 0.1:  # Inferencia cada 100 ms
                results = recognize_faces_from_camera(frame) # Llama a la función de reconocimiento de caras
                prev_time = current_time

                # Anota y muestra el frame
                annotated_frame = results[0].plot()
                cv2.imshow('Detección en tiempo real', annotated_frame)

            # Salir al presionar 'q'
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    except KeyboardInterrupt:
        print("Interrupción detectada. Cerrando...")
    except Exception as e:
        print(f"Error inesperado: {e}")
    finally:
        # Cerrar correctamente la aplicación
        print("Finalizando...")
        stop_thread = True
        camera_thread.join()
        cap.release()
        cv2.destroyAllWindows()


def recognize_faces_from_camera(frame):
    known_faces = load_known_faces()
    all_results = []

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # <- importante convertir el color
    encodings = face_recognition.face_encodings(rgb_frame)
    print(f"Frame → Faces detected: {len(encodings)}")

    for i, encoding in enumerate(encodings):
        matched_name = "Unknown"
        min_dist = 1.0

        for name, known_encoding in known_faces.items():
            dist = distance.euclidean(known_encoding, encoding)
            if dist < trustValue and dist < min_dist:
                matched_name = name
                min_dist = dist

        confidence = max(0, min(1, 1 - min_dist / trustValue))
        #confidence_percent = f"{round(confidence * 100)}%"

        all_results.append({
            "name": matched_name,
            #"distance": [round(min_dist, 4), confidence_percent]
        })

    return all_results

@app.route('/register', methods=['POST'])
def register():
    # Del formulario se espera un campo 'name' y un archivo 'image'
    name = request.form['name']
    lastname = request.form['lastname']
    file = request.files['image']
    # Esto crea una ruta completa (filepath) donde se va a guardar el archivo
    filepath = os.path.join(UPLOADS_DIR, file.filename)
    # Guardar el archivo en la ruta especificada
    file.save(filepath)

    # Carga la imagen en memoria. Por medio del path del archivo
    image = face_recognition.load_image_file(filepath)
    # Usa la librería face_recognition para extraer "face encodings" (vectores numéricos que representan el rostro)
    encodings = face_recognition.face_encodings(image)

    # Si no se detectó ninguna cara en la imagen, devuelve un error JSON con el mensaje
    if len(encodings) == 0:
        return jsonify({"error": "No face found"}), 400

    # Lo guarda con pickle en un archivo .pkl con el nombre(name) del usuario, dentro de la carpeta KNOWN_FACES_DIR.
    encoding = encodings[0]
    with open(os.path.join(KNOWN_FACES_DIR, f"{name} {lastname}.pkl"), 'wb') as f:
        pickle.dump(encoding, f)
    
    return jsonify({"message": f"{name} registered successfully"})

@app.route('/recognize', methods=['POST'])
def recognize():
    cap = cv2.VideoCapture(1)  # o usa tu IP como 'http://192.168.x.x:4747/video'

    if not cap.isOpened():
        print("No se pudo abrir la cámara")
        return jsonify({"error": "No se pudo abrir la cámara"}), 500

    # Captura un solo frame
    ret, frame = cap.read()
    print(f"Frame capturado: {ret}")
    cap.release()

    if not ret:
        return jsonify({"error": "No se pudo capturar la imagen"}), 500

    results = recognize_faces_from_camera(frame)
    return jsonify({"recognized": results})


if __name__ == '__main__':
    app.run(debug=True)
