import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os

data_dir = "ml/data/disease"
img_size = (128, 128)
batch_size = 8

datagen = ImageDataGenerator(validation_split=0.2, rescale=1./255)

train_gen = datagen.flow_from_directory(
    data_dir, target_size=img_size, batch_size=batch_size, subset="training"
)
val_gen = datagen.flow_from_directory(
    data_dir, target_size=img_size, batch_size=batch_size, subset="validation"
)

model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(16, (3,3), activation='relu', input_shape=(128,128,3)),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(len(train_gen.class_indices), activation='softmax')
])

model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
model.fit(train_gen, validation_data=val_gen, epochs=3)

# Export TFLite model
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

os.makedirs("ml/export_models", exist_ok=True)
with open("ml/export_models/disease_model.tflite", "wb") as f:
    f.write(tflite_model)

print("✅ Disease model saved at ml/export_models/disease_model.tflite")