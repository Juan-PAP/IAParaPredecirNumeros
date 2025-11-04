import tkinter as tk
from PIL import Image, ImageDraw
import numpy as np
import tensorflow as tf
import cv2

# --- Cargar el modelo entrenado ---
try:
    model = tf.keras.models.load_model("modelo_digitos.h5")
    print("¡Modelo 'modelo_digitos.h5' cargado exitosamente!")
except Exception as e:
    print(f"Error cargando el modelo: {e}")
    exit()

# Lista de etiquetas de números en palabras (para la salida)
numeros = ["cero", "uno", "dos", "tres", "cuatro", "cinco", "seis", "siete", "ocho", "nueve"]

# --- Interfaz Gráfica (Tu diseño original) ---
ventana = tk.Tk()
ventana.title("Reconocedor de Dígitos")

# Lienzo y dibujo
ancho, alto = 200, 200
canvas = tk.Canvas(ventana, width=ancho, height=alto, bg="white")
canvas.grid(row=1, column=0, columnspan=3, padx=10, pady=10)

# Imagen de respaldo para guardar el dibujo
imagen = Image.new("L", (ancho, alto), 255)  # Escala de grises
draw = ImageDraw.Draw(imagen)

# Variables para el trazo
x_anterior, y_anterior = None, None


# --- Funciones ---
def empezar_trazo(event):
    global x_anterior, y_anterior
    x_anterior, y_anterior = event.x, event.y


def trazar(event):
    global x_anterior, y_anterior
    if x_anterior is not None and y_anterior is not None:
        # --- USAMOS WIDTH=5 --- (Igual al script de recolección de datos)
        grosor = 5
        canvas.create_line(x_anterior, y_anterior, event.x, event.y, width=grosor, fill="black", capstyle=tk.ROUND)
        draw.line((x_anterior, y_anterior, event.x, event.y), fill=0, width=grosor)
    x_anterior, y_anterior = event.x, event.y


def finalizar_trazo(event):
    global x_anterior, y_anterior
    x_anterior, y_anterior = None, None


def borrar_canvas():
    canvas.delete("all")
    global imagen, draw
    imagen = Image.new("L", (ancho, alto), 255)
    draw = ImageDraw.Draw(imagen)
    # Resetear la etiqueta
    etiqueta.config(text="Dibuja un número...", fg="black")


def predecir_digito():
    """
    Esta función reemplaza a tu antigua 'guardar_imagen'.
    Toma la imagen del canvas, la procesa y la envía al modelo.
    """
    # Convertir la imagen de PIL a un array de NumPy
    img_np = np.array(imagen)

    # --- Pre-procesamiento IDÉNTICO al de entrenamiento ---
    # 1. Invertir (dígito blanco, fondo negro)
    img_invertida = 255 - img_np

    # 2. Redimensionar a 28x28 (el tamaño que espera la IA)
    img_redimensionada = cv2.resize(img_invertida, (28, 28), interpolation=cv2.INTER_AREA)

    # 3. Normalizar (0 a 1)
    img_normalizada = img_redimensionada / 255.0

    # 4. Añadir dimensiones (1, 28, 28, 1)
    img_final = img_normalizada.reshape(1, 28, 28, 1)

    # --- Hacer la Predicción ---
    prediccion = model.predict(img_final)

    numero_predicho_idx = np.argmax(prediccion)
    nombre_numero = numeros[numero_predicho_idx]
    confianza = prediccion[0][numero_predicho_idx] * 100

    # Mostrar el resultado en la etiqueta
    etiqueta.config(text=f"{nombre_numero.upper()} ({confianza:.2f}%)", fg="green")


# --- Widgets (Tu layout original) ---
etiqueta = tk.Label(ventana, text="Dibuja un número...", font=("Arial", 20, "bold"))
etiqueta.grid(row=0, column=0, columnspan=3, pady=5)

boton_borrar = tk.Button(ventana, text="Borrar", command=borrar_canvas, bg="red", fg="white", font=("Arial", 12))
boton_borrar.grid(row=2, column=0, padx=10, pady=5, sticky="ew")

# El botón "Siguiente" ahora es "Predecir" y llama a la IA
boton_predecir = tk.Button(ventana, text="Predecir", command=predecir_digito, bg="green", fg="white",
                           font=("Arial", 12, "bold"))
boton_predecir.grid(row=2, column=2, padx=10, pady=5, sticky="ew")

# Enlazar eventos del mouse
canvas.bind("<ButtonPress-1>", empezar_trazo)
canvas.bind("<B1-Motion>", trazar)
canvas.bind("<ButtonRelease-1>", finalizar_trazo)

# Ejecutar
ventana.mainloop()