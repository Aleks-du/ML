#Определить какой из трех классификаторов (kNN, наивный Байес, решающее дерево) лучший в метрике accuracy.
#наивный Байес
import numpy as np
import pandas as pd
import sklearn
from sklearn.naive_bayes import GaussianNB
gnb = GaussianNB()

data = pd.read_csv("mlbootcamp5_train.csv", 
                 sep=";", 
                 index_col="id")

train = data[["age", "weight", "height"]]
target = data["cardio"]

model = gnb.fit(train, target)
predict = model.predict(train)
print(data.shape[0],
     (target == predict).sum() / data.shape[0])
