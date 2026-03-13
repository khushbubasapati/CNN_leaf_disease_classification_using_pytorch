# Potato Leaf Disease Classification using CNN

This project uses a **Convolutional Neural Network (CNN)** to classify potato leaf diseases from images. It identifies whether a potato leaf is **Healthy**, affected by **Early Blight**, or **Late Blight**. The trained deep learning model is deployed with an interactive **Streamlit UI** for easy image-based predictions.

---

## 🚀 Features

* Deep Learning model built using **PyTorch**
* Clean and interactive **Streamlit** interface
* Accepts user-uploaded leaf images
* Displays:

  * Predicted class (disease type)
  * Confidence score
  * Class-wise probability breakdown
* Compatible with both **CPU** and **GPU**

---

## 📂 Dataset

- **Source:** [PlantVillage Dataset](https://www.kaggle.com/datasets/arjuntejaswi/plant-village)
- **Total images:** 2,152
  - Early Blight: 1,000 images
  - Late Blight: 1,000 images
  - Healthy: 152 images
- **Split:** 80% training / 20% validation

---

## 🧠 Model Overview

* **Architecture:** Custom CNN built and trained in Jupyter Notebook (`.ipynb`)
* **Framework:** PyTorch
* **Training Data:** Potato leaf images (Healthy, Early Blight, Late Blight)
* **Saved Model:** `model_traced.pt` (TorchScript version for deployment)

The model was trained on **Google Colab** using a GPU (CUDA), then exported using `torch.jit.trace` as `model_traced.pt` for deployment.

---

## 🗂️ Project Structure

```
├── app.py                    # Streamlit UI
├── model_traced.pt          # Traced PyTorch model
├── Potato_Leaf_CNN.ipynb    # Training notebook (CNN model)
├── requirements.txt         # Dependencies
├── README.md                # Project documentation
└── sample_images/           # Example input images
```

---

## 🌐 Deployment on Streamlit Cloud

This project is deployed on **Streamlit Cloud**, allowing anyone to try the potato leaf disease classifier directly from the browser.

### 🔗 Live App Link

👉 **Streamlit App:** *https://khushbubasapati-cnn-leaf-disease-classification.streamlit.app/*

You can upload any potato leaf image and the model will instantly predict the disease class along with confidence scores.

**Key features:**
- Upload JPG/PNG images of potato leaves
- Displays predicted disease with confidence level (High / Moderate / Low)
- Shows per-class probability bars
- **Out-of-Distribution (OOD) detection** using Shannon entropy — if the uploaded image doesn't resemble a potato leaf, the model flags it as unrecognised instead of giving a false confident prediction
- Displays a real-time uncertainty meter

---

## 📊 Results

| Split      | Accuracy |
|------------|----------|
| Training   | 99.71%   |
| Validation | 99.07%   |

---

## 💡 Key Learnings

- Built and trained a CNN from scratch without transfer learning
- Used `torch.jit.trace` to export the model for production use
- Implemented **entropy-based OOD detection** to prevent the model from confidently misclassifying random non-leaf images
- Managed Python version compatibility between Google Colab (3.12) and local machine (3.13)

---

## 🚀 Future Improvements

- **Add a "Not a Potato Leaf" class** — retrain with a 4th class containing non-potato leaf images so the model explicitly learns to reject unrelated inputs rather than relying on entropy thresholds alone
- **Transfer learning with ResNet18** — replace the custom CNN with a pretrained ResNet18 backbone for richer visual understanding and better out-of-distribution robustness
- **Data augmentation** — add random flips, rotations, and colour jitter during training to improve generalisation beyond the PlantVillage dataset
- **Expand to more crops** — extend the model to detect diseases in other plants like tomato, corn, and rice using the full PlantVillage collection


If you like this project, feel free to ⭐ the repository and explore the code!
