#resulted in accuracy of 73.86%
# Import necessary libraries
import os
import numpy as np
import pandas as pd
from pathlib import Path
import cv2
from sklearn.model_selection import train_test_split
from sklearn.utils import compute_class_weight
import tensorflow as tf
from tensorflow.keras import layers, regularizers, Model
from tensorflow.keras.applications import EfficientNetB5
from tensorflow.keras.applications.efficientnet import preprocess_input
from tensorflow.keras.callbacks import ModelCheckpoint
import matplotlib.pyplot as plt

# Define the paths
images_path = Path("/content/brain_tumor_data/images")
labels_path = Path("/content/brain_tumor_data/labels")

# List all image and label files
image_files = list(images_path.glob("*"))
label_files = list(labels_path.glob("*"))

# Function to get base filename (without extension)
def get_base_filename(path):
    return path.stem

# Create dictionaries to map base filenames to paths
image_map = {get_base_filename(p): p for p in image_files}
label_map = {get_base_filename(p): p for p in label_files}

# Match image and label files and extract tumor presence
matched_data = []
for base_name, label_path in label_map.items():
    if base_name in image_map:
        with open(label_path, 'r') as f:
            content = f.read().strip()
            if content:
                first_class = int(content.split()[0])  # 1 for tumor, 0 for no tumor
                matched_data.append({
                    "filename": base_name,
                    "image_path": str(image_map[base_name]),
                    "label_path": str(label_path),
                    "has_tumor": first_class == 1
                })

# Create a DataFrame from matched data
matched_df = pd.DataFrame(matched_data)

# Split data into training and test sets with stratification
train_df, test_df = train_test_split(
    matched_df,
    test_size=0.2,
    stratify=matched_df["has_tumor"],
    random_state=42
)
train_df = train_df.reset_index(drop=True)
test_df = test_df.reset_index(drop=True)

# Define image size and input shape
IMAGE_SIZE = (224, 224)
input_shape = (224, 224, 3)

# Function to load and preprocess images
def load_image(path, size=IMAGE_SIZE):
    img = cv2.imread(str(path))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, size)
    return img.astype('float32')  # Keep in [0, 255] for preprocessing later

# Load training and test data
X_train = np.array([load_image(p) for p in train_df["image_path"]])
y_train = np.array(train_df["has_tumor"]).astype(int)
X_test = np.array([load_image(p) for p in test_df["image_path"]])
y_test = np.array(test_df["has_tumor"]).astype(int)

# Compute class weights to handle imbalance
class_weights = compute_class_weight('balanced', classes=np.array([0, 1]), y=y_train)
class_weights_dict = dict(enumerate(class_weights))
print(f"Class Weights: {class_weights_dict}")

# Define data augmentation layer
data_augmentation = tf.keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.2),
    layers.RandomZoom(0.2),
    layers.RandomBrightness(0.3),
    layers.RandomContrast(0.1)
], name="data_augmentation")

# Build the model using EfficientNetB5
base_model = EfficientNetB5(input_shape=input_shape, include_top=False, weights='imagenet')
base_model.trainable = False  # Freeze the base model

inputs = layers.Input(shape=input_shape)
x = data_augmentation(inputs)
x = layers.Lambda(lambda x: preprocess_input(x))(x)  # Preprocess for EfficientNet
x = base_model(x, training=False)
x = layers.GlobalAveragePooling2D()(x)
x = layers.BatchNormalization()(x)
x = layers.Dropout(0.3)(x)
x = layers.Dense(256, activation='relu', kernel_regularizer=regularizers.l2(1e-4))(x)
x = layers.BatchNormalization()(x)
x = layers.Dropout(0.3)(x)
outputs = layers.Dense(1, activation='sigmoid')(x)
model = Model(inputs, outputs)

# Compile the model
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# Define callbacks without EarlyStopping
callbacks = [
    ModelCheckpoint("efficientnetB5_best.keras", save_best_only=True, monitor='val_accuracy')
]

# Train the model
history = model.fit(
    X_train, y_train,
    validation_data=(X_test, y_test),
    epochs=50,
    batch_size=16,
    callbacks=callbacks,
    class_weight=class_weights_dict
)

# Evaluate the model on the test set
loss, acc = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {acc * 100:.2f}%")

# Plot training and validation accuracy and loss
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Val Accuracy')
plt.title('Accuracy')
plt.xlabel('Epoch')
plt.legend()
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.title('Loss')
plt.xlabel('Epoch')
plt.legend()
plt.grid(True)
plt.show()

# Save the final model in .keras format
model.save("efficientnetB5_final.keras")