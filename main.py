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

# Custom CSS for white background and clean UI
st.markdown(
    """
    <style>
    .main {
        background-color: #FFFFFF;
    }
    .css-18e3th9 {
        padding: 2rem 5rem;
        background-color: white;
    }
    .css-1d391kg {
        padding: 3.5rem 1rem;
        background-color: white;
    }
    h1 {
        text-align: center;
        color: #262730;
    }
    p {
        text-align: center;
        color: #262730;
    }
    .stFileUploader {
        width: 100%;
        max-width: 600px;
        margin: 0 auto;
        border: 2px dashed #cccccc;
        border-radius: 5px;
        padding: 20px;
        text-align: center;
    }
    .st-b7 {
        color: #666666;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Header section
st.markdown("<h1 style='text-align: center; margin-bottom: 10px;'>Vessel Classification</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; margin-bottom: 5px;'>Upload a image of a vessel and get the predicted category.</p>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: #666666; margin-bottom: 30px;'>[Cargo, Military, Carrier, Cruise, Tankers]</p>", unsafe_allow_html=True)

# File uploader with custom styling
uploaded_file = st.file_uploader(
    "Choose an image...",
    type=["jpg", "png", "jpeg"],
    help="Drag and drop file here or browse files (JPG, PNG, JPEG)"
)

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    img_tensor = transform(image).unsqueeze(0).numpy()

    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
    result = session.run([output_name], {input_name: img_tensor})[0]

    probs = torch.nn.functional.softmax(torch.tensor(result[0]), dim=0).numpy()
    pred_index = np.argmax(probs)
    pred_class = class_names[pred_index]

    st.markdown(
        f"<h3 style='text-align: center; margin-top: 20px;'>Predicted Class: <strong style='color: #1f77b4;'>{pred_class}</strong></h3>",
        unsafe_allow_html=True
    )

    fig, ax = plt.subplots()
    ax.barh(class_names, probs, color='#1f77b4')
    ax.set_xlim(0, 1)
    ax.set_xlabel("Confidence", color='#666666')
    ax.tick_params(colors='#666666')
    ax.set_facecolor('white')
    fig.patch.set_facecolor('white')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_color('#eeeeee')
    ax.spines['left'].set_color('#eeeeee')
    ax.invert_yaxis()
    st.pyplot(fig)
