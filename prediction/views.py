from django.shortcuts import render
from rest_framework.decorators import api_view
from rest_framework.response import Response
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
from io import BytesIO

model_path = "brain_tumor_model.h5"
model = load_model(model_path)
CLASS_NAMES = ["glioma", "meningioma", "notumor", "pituitary"]

@api_view(['GET', 'POST'])
def predict_tumor(request):
    prediction = None
    confidence = None

    if request.method == "POST" and request.FILES.get("image"):
        try:
            img_file = request.FILES["image"]
            img_bytes = BytesIO(img_file.read())
            img = image.load_img(img_bytes, target_size=(224, 224))
            img_array = image.img_to_array(img) / 255.0
            img_array = np.expand_dims(img_array, axis=0)

            pred = model.predict(img_array)
            prediction = CLASS_NAMES[np.argmax(pred)]
            confidence = round(float(np.max(pred)) * 100, 2)
        except Exception as e:
            print(f"Error during prediction: {e}")
            prediction = None
            confidence = None

    return render(request, "brain_tumor_app/upload.html", {
        "prediction": prediction,
        "confidence": confidence
    })