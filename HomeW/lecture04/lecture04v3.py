#Определить какой из трех классификаторов (kNN, наивный Байес, решающее дерево) лучший в метрике accuracy.
#kNN
import numpy as np
import pandas as pd
import sklearn
from sklearn.neighbors import KNeighborsClassifier 
from sklearn.model_selection import train_test_split
df = pd.read_csv("mlbootcamp5_train.csv", 
                 sep=";", index_col='id')
y = df['cardio']
X = df[["age", "weight", "height"]]
df.head()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=27)  
KNN_model = KNeighborsClassifier(n_neighbors=5)  
KNN_model.fit(X_train, y_train)
KNN_prediction = KNN_model.predict(X_test)
print(accuracy_score(KNN_prediction, y_test))  
