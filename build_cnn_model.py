import tensorflow as tf

# ----------------------------
# 1. Dataset path
# ----------------------------
DATASET_PATH = r"C:\Users\mjala\OneDrive\Desktop\cnn_dataset\spectrogram_output2"

# ----------------------------
# 2. Image & batch parameters
# ----------------------------
IMG_HEIGHT = 128
IMG_WIDTH = 128
BATCH_SIZE = 32

# ----------------------------
# 3. Load dataset (only to get classes)
# ----------------------------
train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    DATASET_PATH,
    image_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE
)

class_names = train_ds.class_names
num_classes = len(class_names)

print("Classes found:", class_names)
print("Number of classes:", num_classes)

# ----------------------------
# 4. Design CNN Architecture
# ----------------------------
model = tf.keras.Sequential([

    # Input + Rescaling
    tf.keras.layers.Input(shape=(IMG_HEIGHT, IMG_WIDTH, 3)),
    tf.keras.layers.Rescaling(1./255),

    # -------- CONV BLOCK 1 --------
    tf.keras.layers.Conv2D(32, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(),

    # -------- CONV BLOCK 2 --------
    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(),

    # -------- CONV BLOCK 3 --------
    tf.keras.layers.Conv2D(128, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(),

    # -------- FLATTEN --------
    tf.keras.layers.Flatten(),

    # -------- FULLY CONNECTED --------
    tf.keras.layers.Dense(128, activation='relu'),

    # -------- REGULARIZATION --------
    tf.keras.layers.Dropout(0.5),

    # -------- OUTPUT LAYER --------
    tf.keras.layers.Dense(num_classes, activation='softmax')
])

# ----------------------------
# 5. Compile Model
# ----------------------------
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# ----------------------------
# 6. Verify Model
# ----------------------------
print("\nCNN Model Summary:\n")
model.summary()


#to run this use command

#cd C:\Users\mjala\OneDrive\Desktop\cnn_dataset
#python build_cnn_model.py     
