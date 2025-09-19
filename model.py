import tensorflow as tf
from tensorflow.keras import layers, models, utils
import os
import numpy as np

Image_size = (64, 64)
Batch_size = 16
input_shape = (Image_size[0], Image_size[1], 1)

MODEL_PATH = 'Camera_classifier.keras'
DATA_DIR = r"Before execution put in the path to directory where these files are here"

def load_data():
    train_ds = tf.keras.utils.image_dataset_from_directory(
        DATA_DIR,
        image_size=Image_size,
        batch_size=Batch_size,
        color_mode="grayscale"
    )

    normalization_layer = layers.Rescaling(1./255)
    train_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))

    train_ds = train_ds.shuffle(1000)
    val_size = int(len(train_ds) * 0.2)
    val_ds = train_ds.take(val_size)
    train_ds = train_ds.skip(val_size)

    AUTOTUNE = tf.data.AUTOTUNE
    train_ds = train_ds.prefetch(buffer_size=AUTOTUNE)
    val_ds = val_ds.prefetch(buffer_size=AUTOTUNE)

    return train_ds, val_ds

def create_model():
    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        layers.MaxPooling2D(2, 2),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D(2, 2),
        layers.Flatten(),
        layers.Dense(64, activation='relu'),
        layers.Dense(2, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

def train():
    train_ds, val_ds = load_data()
    model = create_model()
    model.fit(train_ds, epochs=10, validation_data=val_ds)
    model.save(MODEL_PATH)
    print("Model trained and saved.")
    return model

def load_trained_model():
    if os.path.exists(MODEL_PATH):
        return tf.keras.models.load_model(MODEL_PATH)
    else:
        print("No trained model found. Please train first.")
        return None

def predict(frame, model):
    # Ensure frame is 3D: (H,W,1)
    if len(frame.shape) == 2:  # grayscale
        frame = np.expand_dims(frame, axis=-1)  # add channel

    # Resize expects 3D (H,W,C)
    img = tf.image.resize(frame, Image_size)

    # Add batch dimension
    img = tf.expand_dims(img, axis=0)  # shape: (1,64,64,1)

    # Normalize
    img = img / 255.0

    # Predict
    pred = model.predict(img)
    return np.argmax(pred, axis=1)[0]
