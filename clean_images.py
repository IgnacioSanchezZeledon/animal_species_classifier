import os
import warnings
from PIL import Image, UnidentifiedImageError

def delete_corrupt_images(directory):
    for root, _, files in os.walk(directory):
        for file in files:
            file_path = os.path.join(root, file)
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("error", UserWarning)
                    with Image.open(file_path) as img:
                        img.verify()  # Verifica si la imagen está corrupta
            except (IOError, SyntaxError, UnidentifiedImageError, UserWarning) as e:
                print(f'Eliminando imagen corrupta: {file_path}')
                os.remove(file_path)

# Directorios de entrenamiento y prueba
train_directory = 'dataset/train'
test_directory = 'dataset/test'

delete_corrupt_images(train_directory)
delete_corrupt_images(test_directory)
