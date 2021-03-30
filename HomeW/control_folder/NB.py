%matplotlib inline
import zipfile
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

z = zipfile.ZipFile("archive.zip")
data = pd.read_csv(z.open("IMDB Dataset.csv"))

senti = data['sentiment']
# приводим к бинарному признаку не трогая родной столбец
data['senti'] = data['sentiment'].map({'negative': 0, 'positive': 1})
#выводим таблицу
#data.head()


X = data['review']
y = data['senti']
from sklearn.feature_extraction.text import CountVectorizer
# удаление стоп слов - увеличивает точность 
# ограничиваемся 200 "выборкой"
vectorizer = CountVectorizer(stop_words="english",max_features=200)
X_vec = vectorizer.fit_transform(X)
#print(vectorizer.get_feature_names())

#смотрим на первый текст
#data['review'][0]
#в векторном представлении
#X_vec[0].toarray()
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import roc_auc_score
clf = MultinomialNB()
clf.fit(X_vec, y)
roc_auc_score(y, clf.predict_proba(X_vec)[:, 1])
