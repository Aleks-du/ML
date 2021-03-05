#3. Построить наивный байесовский классификатор для бинарных полей gender, smoke, alco, active.
#Привести матрицу неточностей и сравнить с предыдущими значениями.

загрузка библиотеки
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

df[df["cardio"] == 1][["gender", "smoke", "alco", "active"]].hist(bins=20)
plt.tight_layout()

#байесовский классификатор для бинарных полей
from sklearn.naive_bayes import BernoulliNB
bnb = BernoulliNB()

train = data[["gender", "smoke", "alco", "active"]]
target = data["cardio"]

model = bnb.fit(train, target)
predict = model.predict(train)
print(data.shape[0],
     (target == predict).sum() / data.shape[0])

predict

#матрица неточностей
from sklearn.metrics import confusion_matrix

cnf_matrix = confusion_matrix(target, predict)
cnf_matrix
