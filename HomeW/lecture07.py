%matplotlib inline
import zipfile
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
X, y = fetch_openml('mnist_784', version=1, return_X_y=True)

from xgboost import XGBClassifier as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
# немного уменьшим данные (опционально)
X = X[:10000]
y = y[:10000]
# преобразуем метки в числа
Y = list(map(int, y))
Y = np.array(Y)

X_train, X_test, y_train, y_test = train_test_split(X, Y, random_state=7)
model = XGBClassifier(max_depth=3, learning_rate=0.2, n_estimators=1000)
model.fit(X_train, y_train)

# делаем прогнозы для тестовых данных
y_pred = model.predict(X_test)
predictions = [round(value) for value in y_pred]

# оцениваем прогнозы
accuracy = accuracy_score(y_test, predictions)
print("Accuracy: %.2f%%" % (accuracy * 100.0))
#Accuracy: 95.36%
