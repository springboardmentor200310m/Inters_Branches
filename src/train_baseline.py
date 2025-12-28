import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# -----------------------------
# PATHS
# -----------------------------
DATASET_PATH = "data/spectrograms"
MODEL_PATH = "model/instrunet_cnn_baseline_clean.h5"

# -----------------------------
# CONFIG
# -----------------------------
IMG_SIZE = (128, 128)
BATCH_SIZE = 32
EPOCHS = 15

# -----------------------------
# DATA GENERATOR
# -----------------------------
datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2
)

train_gen = datagen.flow_from_directory(
    DATASET_PATH,
    target_size=IMG_SIZE,
    color_mode="grayscale",
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    subset="training"
)

val_gen = datagen.flow_from_directory(
    DATASET_PATH,
    target_size=IMG_SIZE,
    color_mode="grayscale",
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    subset="validation"
)

NUM_CLASSES = train_gen.num_classes

# -----------------------------
# SIMPLE BASELINE CNN (STABLE)
# -----------------------------
model = Sequential([
    Conv2D(32, (3,3), activation="relu", input_shape=(128,128,1)),
    MaxPooling2D(2,2),

    Conv2D(64, (3,3), activation="relu"),
    MaxPooling2D(2,2),

    Conv2D(128, (3,3), activation="relu"),
    MaxPooling2D(2,2),

    Flatten(),
    Dense(128, activation="relu"),
    Dropout(0.3),
    Dense(NUM_CLASSES, activation="softmax")
])

# -----------------------------
# COMPILE (DEFAULT ADAM)
# -----------------------------
model.compile(
    optimizer="adam",
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()

# -----------------------------
# TRAIN
# -----------------------------
history = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=EPOCHS
)

# -----------------------------
# SAVE MODEL
# -----------------------------
model.save(MODEL_PATH)
print("✅ Clean baseline model trained successfully.")
