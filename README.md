# 🧠 Face Recognition App (Flask + Python)

Este proyecto implementa un sistema de **reconocimiento facial** usando Python, Flask y la librería `face_recognition`. Permite registrar personas mediante una imagen de su cara, y luego reconocerlas automáticamente en nuevas fotos, incluso con múltiples personas presentes.

---

## 📦 Tecnologías y Librerías Utilizadas

- [Python 3.x](https://www.python.org/)
- [Flask](https://flask.palletsprojects.com/) - Microframework web para el backend
- [face_recognition](https://github.com/ageitgey/face_recognition) - Librería basada en dlib para detección y reconocimiento facial
- [OpenCV (opencv-python)](https://pypi.org/project/opencv-python/) - Para procesamiento de imágenes (opcional)
- [NumPy](https://numpy.org/) - Manipulación de matrices y vectores
- [Pickle](https://docs.python.org/3/library/pickle.html) - Serialización de datos (para guardar las caras conocidas)

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

---

## 📦 Instalación

1. Cloná este repositorio o copiá los archivos.

2. Instalá las dependencias:

```bash
pip install -r requirements.txt


## 🚀 Cómo Ejecutar el Proyecto


Ejecutá la API:
```bash
python app.py
