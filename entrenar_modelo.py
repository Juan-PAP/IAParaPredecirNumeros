import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np

# --- Cargar los datos preparados ---
X_train = np.load("X_train.npy")
y_train = np.load("y_train.npy")
X_test = np.load("X_test.npy")
y_test = np.load("y_test.npy")

print("Datos cargados. Empezando a construir el modelo...")

# --- Construir la Arquitectura de la CNN ---
# Esta es una arquitectura simple pero efectiva para dígitos

model = models.Sequential()
# Capa 1: Convolución + Pooling
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)))
model.add(layers.MaxPooling2D((2, 2)))
# Capa 2: Convolución + Pooling
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
# Capa 3: Convolución
model.add(layers.Conv2D(64, (3, 3), activation='relu'))

# Aplanar la imagen 3D a un vector 1D
model.add(layers.Flatten())
# Capa Densa (totalmente conectada)
model.add(layers.Dense(64, activation='relu'))
# Capa de salida: 10 neuronas (una por dígito, 0-9)
# 'softmax' convierte las salidas en probabilidades
model.add(layers.Dense(10, activation='softmax'))

model.summary()  # Imprime un resumen de la arquitectura

# --- Compilar el Modelo ---
# 'adam' es un optimizador eficiente
# 'sparse_categorical_crossentropy' es la función de pérdida para clasificación (cuando las etiquetas son números simples)
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy']) # Queremos monitorear la precisión

# --- Entrenar el Modelo ---
print("\nIniciando entrenamiento...")
# Epochs (Épocas): Cuántas veces el modelo verá todos los datos de entrenamiento
# Con 2184 imágenes, 10-15 épocas es un buen punto de partida.
model.fit(X_train, y_train, epochs=15,
          validation_data=(X_test, y_test))

# --- Guardar el Modelo Entrenado ---
# Guardamos el modelo completo (arquitectura, pesos, etc.) en un solo archivo
model.save("modelo_digitos.h5")

print("\n¡Entrenamiento finalizado! Modelo guardado como 'modelo_digitos.h5'")

# --- Evaluar el Modelo ---
# Comprobamos la precisión final usando los datos de prueba que el modelo nunca vio
test_loss, test_acc = model.evaluate(X_test, y_test, verbose=2)
print(f"\nPrecisión final en el conjunto de prueba: {test_acc*100:.2f}%")