#1. Построить наивный байесовский классификатор для количественных полей age, height, weight, ap_hi, ap_lo. 
#Исправить данные, если это необходимо. Привести матрицу неточностей и сравнить со значением полученным в ходе лекции. Попытаться объяснить разницу.

#загрузка библиотеки
%matplotlib inline
import numpy as np
import pandas as pd
import seaborn as sns
import sklearn
from matplotlib import pyplot as plt
import warnings
warnings.filterwarnings('ignore')

#загрузка файла
df = pd.read_csv("/content/mlbootcamp5_train.csv", 
                 sep=";", 
                 index_col="id")
df.head()

#1 Исправление данных
data = df[(df["ap_hi"] >= 80) & (df["ap_hi"] <= 200)]
data = data[(data["ap_lo"] >= 50) & (data["ap_lo"] <= 150)]
data = data[(data["height"] >= 120) & (data["height"] <= 225)]
data = data[(data["weight"] >= 30) & (data["weight"] <= 150)]

data[data["cardio"] == 1][["age", "height", "weight", "ap_hi", "ap_lo"]].hist(bins=20)
plt.tight_layout()

#байесовский классификатор для количественных полей
from sklearn.naive_bayes import GaussianNB
gnb = GaussianNB()

train = data[["age", "height", "weight", "ap_hi", "ap_lo"]]
target = data["cardio"]

model = gnb.fit(train, target)
predict = model.predict(train)
print(data.shape[0],
     (target == predict).sum() / data.shape[0])

predict

#матрица неточностей
from sklearn.metrics import confusion_matrix

cnf_matrix = confusion_matrix(target, predict)
cnf_matrix
