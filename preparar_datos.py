import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split

RUTA_DATASET = "Dataset"
IMG_ANCHO = 28
IMG_ALTO = 28

mapa_etiquetas = {
    "cero": 0, "uno": 1, "dos": 2, "tres": 3, "cuatro": 4,
    "cinco": 5, "seis": 6, "siete": 7, "ocho": 8, "nueve": 9
}

datos = []
etiquetas = []

print("Empezando a cargar imágenes...")

for amigo in os.listdir(RUTA_DATASET):
    ruta_amigo = os.path.join(RUTA_DATASET, amigo)
    if os.path.isdir(ruta_amigo):
        ruta_imagenes = os.path.join(ruta_amigo, "imagenes")
        if os.path.isdir(ruta_imagenes):
            for nombre_archivo in os.listdir(ruta_imagenes):
                if nombre_archivo.endswith(".png"):

                    base_nombre = os.path.splitext(nombre_archivo)[0]

                    etiqueta_texto = base_nombre.rstrip("0123456789")

                    if etiqueta_texto in mapa_etiquetas:

                        etiqueta_num = mapa_etiquetas[etiqueta_texto]

                        ruta_completa = os.path.join(ruta_imagenes, nombre_archivo)

                        img = cv2.imread(ruta_completa, cv2.IMREAD_GRAYSCALE)

                        img = 255 - img

                        img_redimensionada = cv2.resize(img, (IMG_ANCHO, IMG_ALTO), interpolation=cv2.INTER_AREA)

                        datos.append(img_redimensionada)
                        etiquetas.append(etiqueta_num)

print(f"¡Carga completa! Se encontraron {len(datos)} imágenes.")

datos = np.array(datos, dtype="float32")
etiquetas = np.array(etiquetas)

datos /= 255.0

datos = datos.reshape(datos.shape[0], IMG_ANCHO, IMG_ALTO, 1)

(X_train, X_test, y_train, y_test) = train_test_split(
    datos, etiquetas, test_size=0.2, stratify=etiquetas, random_state=42
)

np.save("X_train.npy", X_train)
np.save("y_train.npy", y_train)
np.save("X_test.npy", X_test)
np.save("y_test.npy", y_test)

print("Datos procesados y guardados en archivos .npy")
print(f"Forma de X_train: {X_train.shape}")
print(f"Forma de y_train: {y_train.shape}")