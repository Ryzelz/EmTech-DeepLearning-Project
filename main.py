import streamlit as st
from PIL import Image
import numpy as np
import torch
import torchvision.transforms as transforms
import onnxruntime as ort
import matplotlib.pyplot as plt

class_names = ['Cargo', 'Military', 'Carrier', 'Cruise', 'Tankers']

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

@st.cache_resource
def load_model():
    return ort.InferenceSession("ship_classifier.onnx")

session = load_model()

st.title("Ship Classification with Deep Learning 🚢")
st.write("Upload a ship image and get the predicted category.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_container_width=True)
    
    img_tensor = transform(image).unsqueeze(0).numpy()  

    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
    result = session.run([output_name], {input_name: img_tensor})[0]

    probs = torch.nn.functional.softmax(torch.tensor(result[0]), dim=0).numpy()
    pred_index = np.argmax(probs)
    pred_class = class_names[pred_index]

    st.write(f"### Predicted Class: **{pred_class}**")
    
    fig, ax = plt.subplots()
    ax.barh(class_names, probs, color='skyblue')
    ax.set_xlim(0, 1)
    ax.set_xlabel("Confidence")
    ax.invert_yaxis()
    st.pyplot(fig)
