import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split 
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers , models
import os
from tensorflow import keras
from tensorflow.keras.models import load_model


model_path = "brain_tumor_model.h5"

image_size = (224, 224)
batch_size = 32
history =  None
train_ds = tf.keras.utils.image_dataset_from_directory(
  "./Brain_Tumor_Dataset/Training",
  image_size=image_size,
  batch_size=batch_size)

val_ds = tf.keras.utils.image_dataset_from_directory(
  "./Brain_Tumor_Dataset/Testing",
  image_size=image_size,
  batch_size=batch_size)


class_names = ["glioma", "meningioma", "notumor", "pituitary"]
plt.figure(figsize=(10, 10))
for images, labels in train_ds.take(1):
    for i in range(9):
        ax = plt.subplot(3, 3, i + 1)
        plt.imshow(images[i].numpy().astype("uint8"))
        plt.title(class_names[labels[i]])
        plt.axis("off")
plt.tight_layout()
plt.show()
normalization_layer = tf.keras.layers.Rescaling(1./255)

train_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))
val_ds = val_ds.map(lambda x, y: (normalization_layer(x), y))

if os.path.exists(model_path):
  print("Loading the Saved Model")
  model = load_model(model_path)
else:
  model = models.Sequential([
      layers.Conv2D(32, (3,3), activation='relu', input_shape=(224,224,3)),
      layers.MaxPooling2D(),
      
      layers.Conv2D(64, (3,3), activation='relu'),
      layers.MaxPooling2D(),
      
      layers.Flatten(),
      layers.Dense(128, activation='relu'),
      layers.Dense(4, activation='softmax')
  ])

  model.compile(
      optimizer='adam',
      loss='sparse_categorical_crossentropy',
      metrics=['accuracy']
  )
  history = model.fit(
      train_ds,
      validation_data=val_ds,
      epochs=10
  )
  model.save("brain_tumor_model.h5")
if history is not None:
  plt.figure(figsize=(8, 8))
  plt.plot(history.history['accuracy'], label='accuracy')
  plt.plot(history.history['val_accuracy'], label = 'val_accuracy')
  plt.xlabel('Epoch')
  plt.ylabel('Accuracy')
  plt.ylim([0.5, 1])
  plt.legend(loc='lower right')
  plt.show()
else:
  pass

test_loss, test_acc = model.evaluate(val_ds, verbose=2)
print(test_acc)