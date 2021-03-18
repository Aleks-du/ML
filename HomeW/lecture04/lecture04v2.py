#Определить какой из трех классификаторов (kNN, наивный Байес, решающее дерево) лучший в метрике accuracy.
#решающее дерево
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score  


df = pd.read_csv("mlbootcamp5_train.csv", 
                 sep=";", index_col='id')
target = df['cardio']

train, test, target_train, target_test = train_test_split(
    df[['age','height','weight']], target, 
    test_size=0.3)
# Обучаем модель 
tree = DecisionTreeClassifier(max_depth=6, random_state=13)
tree.fit(train, target_train);
predict = tree.predict(test)
# Проверяем
accuracy_score(predict, target_test)
