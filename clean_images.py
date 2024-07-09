import os
from PIL import Image

def verify_and_remove_corrupt_images(directory):
    for root, dirs, files in os.walk(directory):
        for file in files:
            try:
                file_path = os.path.join(root, file)
                img = Image.open(file_path)
                img.verify()  # Verifica que el archivo se puede abrir y no está corrupto
            except (IOError, SyntaxError, Image.DecompressionBombError, Image.UnidentifiedImageError) as e:
                print(f"Imagen corrupta detectada y eliminada: {file_path} - Error: {e}")
                os.remove(file_path)

# Verificar y limpiar las imágenes en los directorios de train y test
verify_and_remove_corrupt_images('dataset/train')
verify_and_remove_corrupt_images('dataset/test')
