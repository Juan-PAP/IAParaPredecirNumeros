import tkinter as tk
from PIL import Image, ImageDraw
import numpy as np
import tensorflow as tf
import cv2

try:
    model = tf.keras.models.load_model("modelo_digitos.h5")
    print("¡Modelo 'modelo_digitos.h5' cargado exitosamente!")
except Exception as e:
    print(f"Error cargando el modelo: {e}")
    exit()

numeros = ["cero", "uno", "dos", "tres", "cuatro", "cinco", "seis", "siete", "ocho", "nueve"]

ventana = tk.Tk()
ventana.title("Reconocedor de Dígitos")

ancho, alto = 200, 200
canvas = tk.Canvas(ventana, width=ancho, height=alto, bg="white")
canvas.grid(row=1, column=0, columnspan=3, padx=10, pady=10)

imagen = Image.new("L", (ancho, alto), 255)  # Escala de grises
draw = ImageDraw.Draw(imagen)

x_anterior, y_anterior = None, None

def empezar_trazo(event):
    global x_anterior, y_anterior
    x_anterior, y_anterior = event.x, event.y


def trazar(event):
    global x_anterior, y_anterior
    if x_anterior is not None and y_anterior is not None:
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
    etiqueta.config(text="Dibuja un número...", fg="black")


def predecir_digito():
    img_np = np.array(imagen)

    img_invertida = 255 - img_np

    img_redimensionada = cv2.resize(img_invertida, (28, 28), interpolation=cv2.INTER_AREA)

    img_normalizada = img_redimensionada / 255.0

    img_final = img_normalizada.reshape(1, 28, 28, 1)

    prediccion = model.predict(img_final)

    numero_predicho_idx = np.argmax(prediccion)
    nombre_numero = numeros[numero_predicho_idx]
    confianza = prediccion[0][numero_predicho_idx] * 100

    etiqueta.config(text=f"{nombre_numero.upper()} ({confianza:.2f}%)", fg="green")

etiqueta = tk.Label(ventana, text="Dibuja un número...", font=("Arial", 20, "bold"))
etiqueta.grid(row=0, column=0, columnspan=3, pady=5)

boton_borrar = tk.Button(ventana, text="Borrar", command=borrar_canvas, bg="red", fg="white", font=("Arial", 12))
boton_borrar.grid(row=2, column=0, padx=10, pady=5, sticky="ew")

boton_predecir = tk.Button(ventana, text="Predecir", command=predecir_digito, bg="green", fg="white",
                           font=("Arial", 12, "bold"))
boton_predecir.grid(row=2, column=2, padx=10, pady=5, sticky="ew")

canvas.bind("<ButtonPress-1>", empezar_trazo)
canvas.bind("<B1-Motion>", trazar)
canvas.bind("<ButtonRelease-1>", finalizar_trazo)

ventana.mainloop()