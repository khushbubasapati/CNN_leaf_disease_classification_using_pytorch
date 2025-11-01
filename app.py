import streamlit as st
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import warnings
warnings.filterwarnings('ignore')



#1.Page setup

st.set_page_config(page_title="Potato Leaf Disease Classifier", page_icon="🥔", layout="centered")

st.title("🥔 Potato Leaf Disease Classifier")
st.write("Upload an image of a potato leaf, and the model will predict the disease type.")


#2. Device setup

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


#3. Load trained model

@st.cache_resource
def load_model():
    model = torch.jit.load("model_traced.pt", map_location=device)
    model.eval()
    return model

model = load_model()


#4. Class labels

CLASS_NAMES = ['Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy']


#5. Image preprocessing

transform = transforms.Compose([
     transforms.Resize((256,256)),
     transforms.ToTensor(),
])


#6. File uploader

uploaded_file = st.file_uploader("Choose a leaf image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display uploaded image
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_container_width=True)

    st.write("Classifying... ⏳")

    # Preprocess image
    img_tensor = transform(image).unsqueeze(0).to(device)

   
    #7. Make prediction
    
    with torch.no_grad():
        outputs = model(img_tensor)
        probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
        confidence, predicted_class = torch.max(probabilities, 0)

    
    #8. Display results
  
    predicted_label = CLASS_NAMES[predicted_class.item()]
    st.success(f"**Prediction:** {predicted_label}")
    st.info(f"**Confidence:** {confidence.item() * 100:.2f}%")

    # Show all class probabilities
    st.subheader("Class Probabilities:")
    for i, cls in enumerate(CLASS_NAMES):
        st.write(f"{cls}: {probabilities[i].item() * 100:.2f}%")
