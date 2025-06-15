from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional
import pandas as pd
import joblib
import datetime
import os

# Evidently
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset, DataQualityPreset

from fastapi.staticfiles import StaticFiles

# Inicializar app
app = FastAPI()

# Montar carpeta estática para los reportes
app.mount("/reports", StaticFiles(directory="reports"), name="reports")

# Cargar modelo y datos de referencia
model = joblib.load("rf_car_price_pipeline.joblib")
reference_data = pd.read_csv("reference_data.csv", sep=';')  # Muy importante

# Mapeo de nombres esperados por el modelo
column_map = {
    "Engine_Fuel_Type": "Engine Fuel Type",
    "Engine_HP": "Engine HP",
    "Engine_Cylinders": "Engine Cylinders",
    "Transmission_Type": "Transmission Type",
    "Number_of_Doors": "Number of Doors",
    "Market_Category": "Market Category",
    "Vehicle_Size": "Vehicle Size",
    "Vehicle_Style": "Vehicle Style",
    "highway_MPG": "highway MPG",
    "city_MPG": "city mpg"
}

# Clase con formato de entrada
class CarFeatures(BaseModel):
    Make: str
    Model: str
    Year: int
    Engine_Fuel_Type: str
    Engine_HP: Optional[float]
    Engine_Cylinders: Optional[float]
    Transmission_Type: str
    Driven_Wheels: str
    Number_of_Doors: Optional[float]
    Market_Category: Optional[str]
    Vehicle_Size: str
    Vehicle_Style: str
    highway_MPG: float
    city_MPG: float
    Popularity: int

@app.post("/predict")
def predict(features: CarFeatures):
    try:
        # Convertir input a DataFrame y mapear nombres
        input_dict = features.dict()
        transformed = {column_map.get(k, k): v for k, v in input_dict.items()}
        df = pd.DataFrame([transformed])

        # Generar reporte Evidently
        report = Report(metrics=[
            DataDriftPreset(),
            DataQualityPreset()
        ])
        report.run(reference_data=reference_data, current_data=df)

        # Guardar reporte en carpeta reports/
        # timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        report_filename = f"evidently_report.html"
        report_path = f"reports/{report_filename}"
        report.save_html(report_path)

        # Predecir precio
        prediction = model.predict(df)[0]

        # Respuesta con precio + link al reporte
        return {
            "predicted_price": float(prediction),
            "report_url": f"http://127.0.0.1:8000/reports/{report_filename}"
        }

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/", tags=["root"])
def read_root():
    return {"message": "hola mundo"}