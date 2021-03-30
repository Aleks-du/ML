%matplotlib inline
import zipfile
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

z = zipfile.ZipFile("archive.zip")
data = pd.read_csv(z.open("IMDB Dataset.csv"))

senti = data['sentiment']
# приводим к бинарному признаку
data['senti'] = data['sentiment'].map({'negative': 0, 'positive': 1})
#data.head()

X = data['review']
y = data['senti']
from sklearn.feature_extraction.text import CountVectorizer
# удаление стоп слов - увеличивает точность 
# ограничиваемся 200 "выборкой"
vectorizer = CountVectorizer(stop_words="english",max_features=200)
X_vec = vectorizer.fit_transform(X)
#print(vectorizer.get_feature_names())

from sklearn.model_selection import GridSearchCV # для поиска лучшего соседа
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split #разбивка на тест и трейн выборки
X_train, X_test, y_train, y_test = train_test_split(X_vec, y, test_size=0.2, random_state=0)
from sklearn.metrics import roc_auc_score

neigh = KNeighborsClassifier()
neigh.fit(X_test, y_test)

parametrs = { "n_neighbors": range(7, 30, 2)}
grid = GridSearchCV(neigh, parametrs, cv=5)
grid.fit(X_train, y_train)

#поиск "лучшего" соседа
grid.best_params_ 
