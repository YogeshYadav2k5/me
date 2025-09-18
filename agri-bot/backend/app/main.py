from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel
import joblib
import numpy as np
import tensorflow as tf
from PIL import Image
import io

app = FastAPI()

# Load models
crop_model = joblib.load("ml/export_models/crop_model.pkl")
disease_interpreter = tf.lite.Interpreter(model_path="ml/export_models/disease_model.tflite")
disease_interpreter.allocate_tensors()

input_details = disease_interpreter.get_input_details()
output_details = disease_interpreter.get_output_details()

class CropRequest(BaseModel):
    soil_ph: float
    moisture: float
    rainfall: float
    price: float

@app.post("/recommend")
def recommend_crop(req: CropRequest):
    X = np.array([[req.soil_ph, req.moisture, req.rainfall, req.price]])
    pred = crop_model.predict(X)[0]
    return {"recommended_crop": str(pred)}

@app.post("/classify_disease")
async def classify_disease(file: UploadFile = File(...)):
    contents = await file.read()
    image = Image.open(io.BytesIO(contents)).resize((128, 128))
    img_array = np.expand_dims(np.array(image) / 255.0, axis=0).astype(np.float32)

    disease_interpreter.set_tensor(input_details[0]['index'], img_array)
    disease_interpreter.invoke()
    output = disease_interpreter.get_tensor(output_details[0]['index'])
    pred = int(np.argmax(output))

    return {"disease_class": str(pred)}