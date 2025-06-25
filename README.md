# 🧠 Face Recognition App (Flask + Python)

Este proyecto implementa un sistema de **reconocimiento facial en vivo** usando Python, Flask, OpenCV y `face_recognition`. Permite registrar personas mediante una captura directa desde la cámara y luego reconocerlas automáticamente en tiempo real.

---

## 📦 Tecnologías y Librerías Utilizadas

- [Python 3.x](https://www.python.org/)
- [Flask](https://flask.palletsprojects.com/) – Microframework web para el backend
- [face_recognition](https://github.com/ageitgey/face_recognition) – Librería basada en `dlib` para detección y reconocimiento facial
- [OpenCV (opencv-python)](https://pypi.org/project/opencv-python/) – Para procesamiento de imágenes y captura de video
- [Pickle](https://docs.python.org/3/library/pickle.html) – Serialización de datos (para guardar las caras registradas)
- [SciPy](https://www.scipy.org/) – Usado para calcular distancias entre encodings faciales

---

## 🗂️ Estructura del Proyecto
```text
face_app/
├── app.py # Servidor Flask principal
├── known_faces/ # Embeddings de personas registradas (.pkl)
├── uploads/ # Imágenes temporales recibidas por POST
├── requirements.txt # Dependencias del proyecto
├── README.md
```

---

## 🔧 Requisitos

- Python 3.10.x
- Visual Studio con herramientas para C++ (necesario para compilar `dlib`)
- pip actualizado
- Cámara web funcional conectada o celular conectado con Iriun Webcam

---

## 📦 Instalación

1. Cloná este repositorio o copiá los archivos.

2. Instalá las dependencias:

```bash
pip install -r requirements.txt
```

## 🚀 Cómo Ejecutar el Proyecto

--- 

Ejecutá la API:
```bash
python app.py
```
---

## 🔍 Funcionalidades

### 📸 Registro de rostros

Registra una nueva persona capturando una imagen desde la cámara:

**Endpoint:** `POST /register_from_frame`  
**Parámetros (form-data):**

- `name`: Nombre de la persona  
- `lastname`: Apellido de la persona

**Ejemplo con `curl`:**
```bash
curl -X POST http://localhost:5000/register_from_frame \
  -F "name=Juan" \
  -F "lastname=Pérez"
```
Guarda el embedding facial en el directorio known_faces/.

## 🎥 Stream de video en vivo con reconocimiento
Visualiza la transmisión en vivo de la cámara con reconocimiento facial en tiempo real:

**Endpoint:** `GET /video_feed`
Abre en el navegador:
```bash
http://localhost:5000/video_feed
```
Los nombres reconocidos se mostrarán sobre los rostros detectados en el video. :D 
