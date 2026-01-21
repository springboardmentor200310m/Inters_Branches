import tensorflow as tf
from datetime import datetime

# Load model
model = tf.keras.models.load_model("cnn_spectrogram_model.keras")

# Create logs directory
log_dir = "logs/model_graph/" + datetime.now().strftime("%Y%m%d-%H%M%S")

# TensorBoard callback
tensorboard_callback = tf.keras.callbacks.TensorBoard(
    log_dir=log_dir,
    histogram_freq=1,
    write_graph=True
)

# Dummy compile (required)
model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

# Write graph (no training needed)
model.fit(
    tf.random.normal([1, 128, 128, 3]),
    tf.constant([0]),
    epochs=1,
    callbacks=[tensorboard_callback]
)

print("TensorBoard logs created at:", log_dir)
