import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np

X_train = np.load("X_train.npy")
y_train = np.load("y_train.npy")
X_test = np.load("X_test.npy")
y_test = np.load("y_test.npy")

print("Datos cargados. Empezando a construir el modelo...")

model = models.Sequential()

model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))

model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(10, activation='softmax'))

model.summary()

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

print("\nIniciando entrenamiento...")

model.fit(X_train, y_train, epochs=15,
          validation_data=(X_test, y_test))

model.save("modelo_digitos.h5")

print("\n¡Entrenamiento finalizado! Modelo guardado como 'modelo_digitos.h5'")

test_loss, test_acc = model.evaluate(X_test, y_test, verbose=2)
print(f"\nPrecisión final en el conjunto de prueba: {test_acc*100:.2f}%")