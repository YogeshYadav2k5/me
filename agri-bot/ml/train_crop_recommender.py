import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import joblib

# Load data
df = pd.read_csv("ml/data/crop_training.csv")

X = df[["soil_ph", "moisture", "rainfall", "price"]]
y = df["crop_label"]

model = RandomForestClassifier()
model.fit(X, y)

joblib.dump(model, "ml/export_models/crop_model.pkl")
print("✅ Crop model saved at ml/export_models/crop_model.pkl")