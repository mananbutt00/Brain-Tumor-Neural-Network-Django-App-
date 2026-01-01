# Brain Tumor Detection Web App


## Demo Screenshots
![Upload Page](screenshots/upload_page.png)
## Description
This project is a web application that allows users to upload brain MRI images and detect whether the scan shows a tumor. The application uses a Convolutional Neural Network (CNN) trained on a brain MRI dataset to classify images into four categories:

- Glioma
- Meningioma
- Pituitary
- No Tumor

The application is built using **Django** for the backend and **TensorFlow/Keras** for the deep learning model. Users can upload an image through the web interface, and the model predicts the tumor type along with the confidence level.

---

## Features
- Upload brain MRI images via web interface.
- Predict tumor type: Glioma, Meningioma, Pituitary, or No Tumor.
- Displays prediction confidence.
- Handles invalid or non-MRI images gracefully.
- Uses a trained CNN model (`brain_tumor_model.h5`).

---

## Requirements
- Python 3.8+
- Django 4.x
- TensorFlow 2.x
- Django REST Framework
- NumPy
- Pillow (for image processing)

---

## Installation
1. Clone the repository:
```bash
git clone https://github.com/yourusername/brain-tumor-detection.git
