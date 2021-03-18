#Определить какой из трех классификаторов (kNN, наивный Байес, решающее дерево) лучший в метрике accuracy.
#наивный Байес
from sklearn.naive_bayes import GaussianNB
gnb = GaussianNB()

train = data[["age", "weight", "height"]]
target = data["cardio"]

model = gnb.fit(train, target)
predict = model.predict(train)
print(data.shape[0],
     (target == predict).sum() / data.shape[0])
