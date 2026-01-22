import os
import numpy as np
from collections import defaultdict
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

# --------------------
# CONFIG
# --------------------
ARRAYS_DIR = "arrays"
TARGET_WIDTH = 128
TEST_SIZE = 0.2
RANDOM_STATE = 42

# --------------------
# SPECTROGRAM AUGMENTATION
# --------------------
def augment_spectrogram(spec):
    spec_aug = spec.copy()

    # 1. Add small Gaussian noise
    noise = 0.01 * np.random.randn(*spec_aug.shape)
    spec_aug = spec_aug + noise

    # 2. Time masking
    time_mask_width = np.random.randint(0, 10)
    t0 = np.random.randint(0, max(1, spec_aug.shape[1] - time_mask_width))
    spec_aug[:, t0:t0 + time_mask_width] = 0

    # 3. Frequency masking
    freq_mask_width = np.random.randint(0, 10)
    f0 = np.random.randint(0, max(1, spec_aug.shape[0] - freq_mask_width))
    spec_aug[f0:f0 + freq_mask_width, :] = 0

    return spec_aug

# --------------------
# LOAD DATA
# --------------------
X = []
y = []

for instrument in os.listdir(ARRAYS_DIR):
    instrument_path = os.path.join(ARRAYS_DIR, instrument)

    if not os.path.isdir(instrument_path):
        continue

    for file in os.listdir(instrument_path):
        if file.endswith(".npy"):
            file_path = os.path.join(instrument_path, file)
            spec = np.load(file_path)

            # Ensure fixed width
            if spec.shape[1] < TARGET_WIDTH:
                pad_width = TARGET_WIDTH - spec.shape[1]
                spec = np.pad(spec, ((0, 0), (0, pad_width)))
            else:
                spec = spec[:, :TARGET_WIDTH]

            # Original sample
            X.append(spec)
            y.append(instrument)

            # Augmented sample
            spec_aug = augment_spectrogram(spec)
            X.append(spec_aug)
            y.append(instrument)

X = np.array(X)
y = np.array(y)

print("Loaded data shape (after augmentation):", X.shape)
print("Loaded labels:", set(y))

# --------------------
# UNDERSAMPLING (BALANCE CLASSES)
# --------------------
data_by_class = defaultdict(list)

for sample, label in zip(X, y):
    data_by_class[label].append(sample)

min_samples = min(len(samples) for samples in data_by_class.values())
print("Minimum samples per class:", min_samples)

X_balanced = []
y_balanced = []

for label, samples in data_by_class.items():
    np.random.shuffle(samples)
    samples = samples[:min_samples]
    X_balanced.extend(samples)
    y_balanced.extend([label] * min_samples)

X = np.array(X_balanced)
y = np.array(y_balanced)

print("Balanced class distribution:")
unique, counts = np.unique(y, return_counts=True)
print(dict(zip(unique, counts)))

# --------------------
# ADD CHANNEL DIMENSION
# --------------------
X = X[..., np.newaxis]  # (samples, height, width, channels)

# --------------------
# LABEL ENCODING
# --------------------
encoder = LabelEncoder()
y_encoded = encoder.fit_transform(y)

print("Label mapping:")
for cls, idx in zip(encoder.classes_, range(len(encoder.classes_))):
    print(f"{cls} -> {idx}")

# --------------------
# TRAIN–VALIDATION SPLIT
# --------------------
X_train, X_val, y_train, y_val = train_test_split(
    X,
    y_encoded,
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE,
    stratify=y_encoded
)

# --------------------
# NORMALIZATION (Z-SCORE – SAFE)
# --------------------
def zscore_normalize(X):
    X = X.astype("float32")
    mean = np.mean(X)
    std = np.std(X) + 1e-8
    return (X - mean) / std

X_train = zscore_normalize(X_train)
X_val   = zscore_normalize(X_val)

print("Normalization check:")
print("X_train max:", np.max(X_train))
print("X_train min:", np.min(X_train))
print("X_train mean:", np.mean(X_train))

# --------------------
# SAVE OUTPUT FILES
# --------------------
np.save("X_train.npy", X_train)
np.save("X_val.npy", X_val)
np.save("y_train.npy", y_train)
np.save("y_val.npy", y_val)
np.save("label_classes.npy", encoder.classes_)

print("\nSaved files:")
print("X_train.npy, X_val.npy, y_train.npy, y_val.npy, label_classes.npy")
