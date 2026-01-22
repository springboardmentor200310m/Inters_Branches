import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam

from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

# --------------------
# LOAD DATA
# --------------------
X_train = np.load("X_train.npy")
X_val   = np.load("X_val.npy")
y_train = np.load("y_train.npy")   # integers (0,1,2)
y_val   = np.load("y_val.npy")
class_names = np.load("label_classes.npy")

num_classes = len(class_names)

print("Training data shape:", X_train.shape)
print("Validation data shape:", X_val.shape)
print("Classes:", class_names)

print("Unique training labels:", np.unique(y_train))
print("Unique validation labels:", np.unique(y_val))

# --------------------
# BUILD CNN MODEL
# --------------------
model = Sequential([
    Conv2D(32, (3, 3), activation="relu", input_shape=X_train.shape[1:]),
    MaxPooling2D((2, 2)),

    Conv2D(64, (3, 3), activation="relu"),
    MaxPooling2D((2, 2)),

    Flatten(),
    Dense(64, activation="relu"),
    Dropout(0.3),
    Dense(num_classes, activation="softmax")
])

# --------------------
# COMPILE MODEL
# --------------------
model.compile(
    optimizer=Adam(learning_rate=0.0003),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()

# --------------------
# TRAIN MODEL
# --------------------
history = model.fit(
    X_train,
    y_train,
    epochs=25,
    batch_size=32,
    validation_data=(X_val, y_val)
)

# --------------------
# TRAINING CURVES (WEEK 5–6 GRAPHS)
# --------------------
epochs = range(1, len(history.history["accuracy"]) + 1)

# Accuracy Curve
plt.figure(figsize=(6, 4))
plt.plot(epochs, history.history["accuracy"], label="Training Accuracy")
plt.plot(epochs, history.history["val_accuracy"], label="Validation Accuracy")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.title("Training vs Validation Accuracy")
plt.legend()
plt.grid(True)
plt.savefig("accuracy_curve.png")
plt.show()

# Loss Curve
plt.figure(figsize=(6, 4))
plt.plot(epochs, history.history["loss"], label="Training Loss")
plt.plot(epochs, history.history["val_loss"], label="Validation Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Training vs Validation Loss")
plt.legend()
plt.grid(True)
plt.savefig("loss_curve.png")
plt.show()

# --------------------
# FINAL EVALUATION
# --------------------
train_loss, train_acc = model.evaluate(X_train, y_train, verbose=0)
val_loss, val_acc     = model.evaluate(X_val, y_val, verbose=0)

print(f"\nFinal Training Accuracy: {train_acc * 100:.2f}%")
print(f"Final Validation Accuracy: {val_acc * 100:.2f}%")

# --------------------
# CONFUSION MATRIX
# --------------------
y_pred_probs = model.predict(X_val)
y_pred = np.argmax(y_pred_probs, axis=1)

cm = confusion_matrix(y_val, y_pred)

plt.figure(figsize=(6, 5))
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=class_names,
    yticklabels=class_names
)
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix")
plt.savefig("confusion_matrix.png")
plt.show()

# --------------------
# CLASS-WISE METRICS
# --------------------
print("\nClassification Report:")
print(classification_report(y_val, y_pred, target_names=class_names))

# --------------------
# ERROR ANALYSIS
# --------------------
print("\nMisclassification Analysis:")
for i in range(len(class_names)):
    for j in range(len(class_names)):
        if i != j and cm[i, j] > 0:
            print(f"{class_names[i]} misclassified as {class_names[j]}: {cm[i, j]} samples")

# --------------------
# SAVE MODEL
# --------------------
model.save("instrument_cnn_model.h5")
print("\nModel saved as instrument_cnn_model.h5")
