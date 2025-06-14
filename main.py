from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional
import pandas as pd
import joblib

app = FastAPI()

# Cargar modelo previamente entrenado
model = joblib.load("rf_car_price_pipeline.joblib")

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

        # Predecir
        prediction = model.predict(df)[0]
        return {"predicted_price": float(prediction)}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/", tags=["root"])
def read_root():
    """
    Endpoint de prueba:
    - Método: GET
    - Ruta  : /
    - Respuesta: "hola mundo"
    """
    return {"message": "hola mundo"}