import streamlit as st
import tensorflow as tf
import pandas as pd
import numpy as np
import os
import random
from PIL import Image
import matplotlib.pyplot as plt

# Constants
CLASS_NAMES = ['Cargo', 'Military', 'Carrier', 'Cruise', 'Tankers']
IMG_SIZE = (224, 224)
CSV_PATH = "/kaggle/input/game-of-deep-learning-ship-datasets/train/train.csv"
IMG_DIR = "/kaggle/input/game-of-deep-learning-ship-datasets/train/images"
SAMPLE_COUNT = 50

# Load model
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("ship_classifier.h5")

model = load_model()

# Load and filter data
@st.cache_data
def load_data():
    df = pd.read_csv(CSV_PATH)
    valid_labels = {0, 1, 2, 3, 4}
    df = df[df['category'].isin(valid_labels)]
    label_map = {0: 'Cargo', 1: 'Military', 2: 'Carrier', 3: 'Cruise', 4: 'Tankers'}
    df['category'] = df['category'].map(label_map)
    return df

df = load_data()

# Random sample
random.seed(42)
sample_df = df.sample(n=SAMPLE_COUNT).reset_index(drop=True)

# Preprocessing function
def load_and_preprocess_image(image_path):
    img = Image.open(image_path).convert("RGB").resize(IMG_SIZE)
    return np.array(img) / 255.0

# Load images and true labels
images = []
true_labels = []

for _, row in sample_df.iterrows():
    image_path = os.path.join(IMG_DIR, row['image'])
    img_array = load_and_preprocess_image(image_path)
    images.append(img_array)
    true_labels.append(row['category'])

images = np.array(images)
true_labels = np.array(true_labels)

# Predict
pred_probs = model.predict(images)
pred_labels = [CLASS_NAMES[np.argmax(p)] for p in pred_probs]

# Accuracy
accuracy = np.mean(pred_labels == true_labels)

# Streamlit UI
st.title("Ship Classifier – Evaluation on 50 Samples")
st.write(f"✅ **Accuracy:** `{accuracy * 100:.2f}%`")

# Bar Chart: Class Distribution
fig, ax = plt.subplots()
true_counts = [np.sum(true_labels == c) for c in CLASS_NAMES]
pred_counts = [pred_labels.count(c) for c in CLASS_NAMES]

x = np.arange(len(CLASS_NAMES))
ax.bar(x - 0.2, true_counts, width=0.4, label="True")
ax.bar(x + 0.2, pred_counts, width=0.4, label="Predicted")
ax.set_xticks(x)
ax.set_xticklabels(CLASS_NAMES, rotation=45)
ax.set_ylabel("Count")
ax.set_title("True vs Predicted Class Counts")
ax.legend()
st.pyplot(fig)

# Display images and predictions
st.subheader("🔍 Predictions on Random Images")
for i in range(0, SAMPLE_COUNT, 5):
    cols = st.columns(5)
    for j, col in enumerate(cols):
        index = i + j
        img = (images[index] * 255).astype(np.uint8)
        true = true_labels[index]
        pred = pred_labels[index]
        caption = f"T: {true} | P: {pred}"
        col.image(img, use_container_width=True, caption=caption)
