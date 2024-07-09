# Animal Species Classifier

Este proyecto utiliza una red neuronal convolucional (CNN) para identificar especies de animales a partir de imágenes.

## Estructura de Archivos

- `dataset/`: Contiene datos de entrenamiento y prueba.
- `single_prediction/`: Contiene imágenes para predicción individual.
- `animal_species_classifier.py`: Script principal para entrenar y evaluar el modelo.
- `predict.py`: Script para hacer predicciones con el modelo entrenado.
- `requirements.txt`: Dependencias del proyecto.
- `README.md`: Documentación del proyecto.

## Uso

1. Coloca las imágenes de entrenamiento y prueba en las carpetas correspondientes.
2. Ejecuta el script `animal_species_classifier.py` para entrenar y evaluar el modelo.
3. Coloca una imagen en `single_prediction/` y utiliza el script `predict.py` para predecir la especie.

## Instalación

Ejecuta el siguiente comando para instalar las dependencias:

```bash
pip install -r requirements.txt
