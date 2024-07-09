import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import json

# Configuración de los generadores de datos con aumentación
train_datagen = ImageDataGenerator(
    rescale=1./255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2
)

test_datagen = ImageDataGenerator(rescale=1./255)

# Crear generadores de datos
training_set = train_datagen.flow_from_directory(
    'dataset/train',
    target_size=(64, 64),
    batch_size=32,
    class_mode='categorical'
)

test_set = test_datagen.flow_from_directory(
    'dataset/test',
    target_size=(64, 64),
    batch_size=32,
    class_mode='categorical'
)

# Guardar los índices de las clases
with open('class_indices.json', 'w') as f:
    json.dump(training_set.class_indices, f)

# Crear datasets de TensorFlow a partir de los generadores y repetirlos
train_dataset = tf.data.Dataset.from_generator(
    lambda: training_set,
    output_signature=(tf.TensorSpec(shape=(None, 64, 64, 3), dtype=tf.float32),
                      tf.TensorSpec(shape=(None, len(training_set.class_indices)), dtype=tf.float32))
).repeat()

val_dataset = tf.data.Dataset.from_generator(
    lambda: test_set,
    output_signature=(tf.TensorSpec(shape=(None, 64, 64, 3), dtype=tf.float32),
                      tf.TensorSpec(shape=(None, len(test_set.class_indices)), dtype=tf.float32))
).repeat()

# Construcción del modelo
model = Sequential([
    tf.keras.Input(shape=(64, 64, 3)),
    Conv2D(32, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(len(training_set.class_indices), activation='softmax')
])

# Compilación del modelo
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Ajuste de steps_per_epoch y validation_steps
steps_per_epoch = training_set.samples // training_set.batch_size
validation_steps = test_set.samples // test_set.batch_size

# Entrenamiento del modelo
model.fit(
    train_dataset,
    steps_per_epoch=steps_per_epoch,
    epochs=100,
    validation_data=val_dataset,
    validation_steps=validation_steps
)

# Evaluación del modelo
loss, accuracy = model.evaluate(val_dataset, steps=validation_steps)
print(f'Loss: {loss}, Accuracy: {accuracy}')

# Guardar el modelo en el formato recomendado
model.save('animal_species_classifier.keras')
