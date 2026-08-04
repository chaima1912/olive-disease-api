from fastapi import FastAPI, File, UploadFile
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import EfficientNetB0
import numpy as np
from PIL import Image
import io

app = FastAPI()

# Rebuild the EXACT same architecture as training
base_model = EfficientNetB0(
    weights=None,  # don't load imagenet weights, we'll load our own trained ones
    include_top=False,
    input_shape=(224, 224, 3)
)
base_model.trainable = False

model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dense(128, activation="relu"),
    layers.Dropout(0.3),
    layers.Dense(3, activation="softmax")
])

# Build before loading weights
model.build(input_shape=(None, 224, 224, 3))

# Load your trained weights
model.load_weights('olive_disease_model.weights.h5')

class_names = ['Healthy', 'aculus_olearius', 'olive_peacock_spot']  # IMPORTANT: replace with your actual class names, in the same order train_dataset used

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    contents = await file.read()
    img = Image.open(io.BytesIO(contents)).convert('RGB')
    img = img.resize((224, 224))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    
    predictions = model.predict(img_array)
    predicted_class = class_names[np.argmax(predictions)]
    confidence = float(np.max(predictions))
    
    return {
        "prediction": predicted_class,
        "confidence": confidence
    }

@app.get("/")
def health_check():
    return {"status": "API is running"}