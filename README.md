# Potato Leaf Disease Classification using CNN

This project uses a **Convolutional Neural Network (CNN)** to classify potato leaf diseases from images.  
It identifies whether a potato leaf is **Healthy**, affected by **Early Blight**, or **Late Blight**.  
The trained deep learning model is deployed with an interactive **Streamlit UI** for easy image-based predictions.

---

## 🚀 Features

- Deep Learning model built using **PyTorch**
- Clean and interactive **Streamlit** interface
- Accepts user-uploaded leaf images
- Displays:
  - Predicted class (disease type)
  - Confidence score
  - Class-wise probability breakdown
- Compatible with both **CPU** and **GPU**

---

## 🧠 Model Overview

- **Architecture:** Custom CNN built and trained in Jupyter Notebook (`.ipynb`)
- **Framework:** PyTorch
- **Training Data:** Potato leaf images (Healthy, Early Blight, Late Blight)
- **Saved Model:** `model_traced.pt` (TorchScript version for deployment)

---

## 🗂️ Project Structure
├── app.py # Streamlit UI
├── model_traced.pt # Traced PyTorch model
├── Potato_Leaf_CNN.ipynb # Training notebook (CNN model)
├── requirements.txt # Dependencies
├── README.md # Project documentation
└── sample_images/ # Example input images