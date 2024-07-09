import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
import os
import json

# Load the saved model
model = tf.keras.models.load_model('animal_species_classifier.keras')

# Load the class indices
with open('class_indices.json', 'r') as f:
    class_indices = json.load(f)
class_indices = {v: k for k, v in class_indices.items()}

# Confidence threshold
CONFIDENCE_THRESHOLD = 0.5

# Function to make predictions
def predict_image(img_path):
    # Load and preprocess the image
    test_image = image.load_img(img_path, target_size=(64, 64))
    test_image = image.img_to_array(test_image)
    test_image = np.expand_dims(test_image, axis=0)
    test_image = test_image / 255.0  # Ensure the image is scaled the same way as in training
    
    # Make the prediction
    result = model.predict(test_image)
    
    # Get the predicted class
    predicted_class_index = np.argmax(result)
    confidence = result[0][predicted_class_index]
    
    if confidence < CONFIDENCE_THRESHOLD:
        predicted_class = "unknown"
    else:
        predicted_class = class_indices[predicted_class_index]
    
    return predicted_class

# Predict all images in a directory
image_directory = 'single_prediction'  # Directory containing images
if not os.path.exists(image_directory):
    raise FileNotFoundError(f"The directory {image_directory} does not exist.")

for filename in os.listdir(image_directory):
    if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
        image_path = os.path.join(image_directory, filename)
        predicted_class = predict_image(image_path)
        print(f'Prediction for {filename}: {predicted_class}')
