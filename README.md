curl -X POST http://localhost:8000/predict \
-H "Content-Type: application/json" \
-d '{"features": [[0.5, 1.2, 3.4, 5.6, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9]]}'


docker build -t my_fastapi_onnx_app .

docker run -p 8000:8000 my_fastapi_onnx_app

docker run -p 8000:8000 shahlo/mlfastapiproject:c99c0ec2635a325bd94bfa983f070c62471903f1

Примерный код для теста

import onnxruntime as ort
import numpy as np

# Загружаем ONNX модель
session = ort.InferenceSession("DecisionTree.onnx")

# Пример данных с 24 признаками
input_features = [
    [0.5, 1.2, 3.4, 5.6, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9],  # 23 признака
    [1.5, 2.3, 3.8, 4.9, 1.1, 2.1, 3.1, 4.1, 5.1, 6.1, 7.1, 8.1, 9.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0, 2.1]   # 23 признака
]

# Преобразование в numpy array
input_data = np.array(input_features, dtype=np.float32)

# Подготавливаем вход для модели
inputs = {session.get_inputs()[0].name: input_data}
print(inputs)
# Выполняем предсказание
try:
    pred = session.run(None, inputs)
    print("Предсказания:", pred[0].tolist())
except Exception as e:
    print("Ошибка выполнения предсказания:", str(e))
