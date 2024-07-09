import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
import os
import json

# Cargar el modelo guardado
model = tf.keras.models.load_model('animal_species_classifier.keras')

# Cargar los índices de las clases
with open('class_indices.json', 'r') as f:
    class_indices = json.load(f)
class_indices = {v: k for k, v in class_indices.items()}

# Función para hacer predicciones
def predict_image(img_path):
    # Cargar la imagen y preprocesarla
    test_image = image.load_img(img_path, target_size=(64, 64))
    test_image = image.img_to_array(test_image)
    test_image = np.expand_dims(test_image, axis=0)
    test_image = test_image / 255.0  # Asegurarse de que la imagen se escale igual que en el entrenamiento
    
    # Realizar la predicción
    result = model.predict(test_image)
    
    # Obtener la clase predicha
    predicted_class_index = np.argmax(result)
    predicted_class = class_indices[predicted_class_index]
    
    return predicted_class

# Predicción de una imagen en particular
image_path = 'single_prediction/image.jpg'  # Ruta de tu imagen
if not os.path.exists(image_path):
    raise FileNotFoundError(f"La imagen en la ruta {image_path} no existe.")
    
predicted_class = predict_image(image_path)
print(f'Predicción: {predicted_class}')
