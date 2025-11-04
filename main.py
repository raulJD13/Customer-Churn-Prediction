# --- main.py ---
# API de FastAPI para el Modelo de Predicción de Churn
# ---------------------------------------------------

import pandas as pd
import joblib
import os
from fastapi import FastAPI
from pydantic import BaseModel

# --- 1. Definir la Aplicación ---
# "app" es el nombre estándar de la instancia de FastAPI
app = FastAPI(title="API de Predicción de Churn", version="1.0")


# --- 2. Cargar el Modelo (¡Versión Profesional con Rutas Relativas!) ---

# Construye una ruta relativa al script actual
# __file__ es la ubicación del script (main.py)
# os.path.dirname(__file__) es la carpeta del proyecto (Customer-Churn-Prediction/)
try:
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    MODEL_DIR = os.path.join(BASE_DIR, "models") # Asume que el modelo está en una carpeta 'models/'
    MODEL_PATH = os.path.join(MODEL_DIR, "rf_churn_model.joblib")
    
    print(f"Buscando modelo en: {MODEL_PATH}")
    model = joblib.load(MODEL_PATH)

except FileNotFoundError:
    print(f"Error: No se encontró el archivo del modelo en {MODEL_PATH}")
    print("Por favor, ejecuta los notebooks 01-04 para generar el archivo 'rf_churn_model.joblib'")
    print("Y asegúrate de colocarlo en una carpeta llamada 'models/' en el directorio principal.")
    model = None
except Exception as e:
    print(f"Error al cargar el modelo: {e}")
    model = None


# --- 3. Definir el "Schema" de Entrada (Pydantic) ---
# Define cómo deben ser los datos que recibimos.
# Coincide con las features que usamos para entrenar.
class CustomerFeatures(BaseModel):
    frequency: int
    monetary: float
    avg_payment_value: float
    avg_items_per_order: float
    avg_freight_value: float
    n_unique_categories: int
    main_category: str
    
    # Ejemplo de cómo se verían los datos de entrada
    class Config:
        json_schema_extra = {
            "example": {
                "frequency": 1,
                "monetary": 59.90,
                "avg_payment_value": 59.90,
                "avg_items_per_order": 1.0,
                "avg_freight_value": 15.50,
                "n_unique_categories": 1,
                "main_category": "sports_leisure"
            }
        }


# --- 4. Crear el "Endpoint" de Predicción ---

@app.post("/predict")
def predict_churn(features: CustomerFeatures):
    
    if model is None:
        return {"error": "Modelo no cargado. Revisa los logs del servidor."}

    # 1. Convertir los datos de entrada de Pydantic a un DataFrame
    # El pipeline de Scikit-learn espera un DataFrame de Pandas
    input_data = pd.DataFrame([features.model_dump()])
    
    # 2. Hacer la predicción de PROBABILIDAD
    # Usamos predict_proba() para obtener la confianza del modelo
    # El resultado es un array: [[prob_clase_0, prob_clase_1]]
    try:
        pred_proba = model.predict_proba(input_data)
        
        # 3. Extraer la probabilidad de churn (Clase 1)
        churn_probability = pred_proba[0][1]
        
        # 4. Devolver el resultado
        return {
            "churn_probability": round(churn_probability, 4),
            "prediction": "Churn" if churn_probability > 0.5 else "Active"
        }
        
    except Exception as e:
        return {"error": f"Error durante la predicción: {str(e)}"}
        

# --- 5. (Opcional) Endpoint Raíz ---
@app.get("/")
def read_root():
    return {"message": "Bienvenido a la API de Predicción de Churn. Usa el endpoint /docs para ver la documentación."}