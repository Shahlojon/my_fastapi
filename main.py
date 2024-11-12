# main.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import onnxruntime as ort
import numpy as np

# Инициализируем приложение
app = FastAPI()

# Загружаем ONNX модель
session = ort.InferenceSession("DecisionTree.onnx")

# Определяем структуру данных, ожидаемых в запросе
class PredictionRequest(BaseModel):
    features: list

@app.post("/predict")
async def predict(request: PredictionRequest):
    # Преобразуем входные данные в формат numpy
    input_data = np.array(request.features, dtype=np.float32)
    # Выполняем предсказание
    inputs = {session.get_inputs()[0].name: input_data}
    try:
        pred = session.run(None, inputs)
        return {"prediction": pred[0].tolist()}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
