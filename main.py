from fastapi import FastAPI, File, UploadFile
import tensorflow as tf
import numpy as np
from PIL import Image
import io

app = FastAPI()

model = tf.keras.models.load_model('olive_disease_model.keras')
class_names = ['aculus_olearius', 'Healthy', 'olive_peacock_spot']  # replace with your actual classes

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