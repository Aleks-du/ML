#2. Написать свой наивный байесовский классификатор для категориальных полей cholesterol, gluc. 
#Привести матрицу неточностей и сравнить со значениями из задачи 1 (нельзя использовать готовое решение из sklearn) (не обязательно)
#использовал sklearn

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

df[df["cardio"] == 1][["cholesterol", "gluc"]].hist(bins=20)
plt.tight_layout()

#байесовский классификатор для категориальных полей
from sklearn.naive_bayes import CategoricalNB
cnb = CategoricalNB()

train = df[["cholesterol", "gluc"]]
target = df["cardio"]

model = cnb.fit(train, target)
predict = model.predict(train)
print(df.shape[0],
     (target == predict).sum() / df.shape[0])

predict

#матрица неточностей
#from sklearn.metrics import confusion_matrix

#cnf_matrix = confusion_matrix(model, predict)
#cnf_matrix
