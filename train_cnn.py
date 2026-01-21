import tensorflow as tf
from tensorflow.keras import layers, models
import os

print("Program started")

# ----------------------------
# CONFIG
# ----------------------------
DATASET_PATH = r"C:\Users\mjala\OneDrive\Desktop\cnn_dataset\spectrogram_output2"
IMG_HEIGHT = 128
IMG_WIDTH = 128
BATCH_SIZE = 32
EPOCHS = 10

print("Dataset path:", DATASET_PATH)

# ----------------------------
# LOAD DATASET
# ----------------------------
print("Loading dataset...")

train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    DATASET_PATH,
    image_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE,
    shuffle=True
)

class_names = train_ds.class_names
num_classes = len(class_names)

print("Classes found:", class_names)
print("Number of classes:", num_classes)

# Improve performance
AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)

# ----------------------------
# CNN MODEL
# ----------------------------
model = models.Sequential([
    layers.Input(shape=(IMG_HEIGHT, IMG_WIDTH, 3)),
    layers.Rescaling(1./255),

    layers.Conv2D(32, (3, 3), activation='relu'),
    layers.MaxPooling2D(),

    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D(),

    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D(),

    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(num_classes, activation='softmax')
])

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()

# ----------------------------
# TRAIN
# ----------------------------
print("Starting training...")
history = model.fit(
    train_ds,
    epochs=EPOCHS
)

# ----------------------------
# SAVE MODEL
# ----------------------------
model.save("cnn_spectrogram_model.keras")
print("Model saved as cnn_spectrogram_model.keras")





#run the below 1,2 commands in the terminal
# 1.python tensorboard_view.py
# 2. tensorboard --logdir=logs

#next,
#copy this in browser for visually accesing the cnn models
# 3.http://localhost:6006


