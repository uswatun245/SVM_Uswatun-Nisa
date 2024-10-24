import numpy as np
import pandas as pd

dataset = pd.read_csv("iris-flower.csv")
dataset
dataset['Species'].unique()
dataset.info()
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
dataset['Species'] = le.fit_transform(dataset['Species'])
dataset
dataset['Species'].unique()
x = np.asarray(dataset.drop(['Id', 'Species'], axis = 1), dtype = np.float64)
y = np.asarray(dataset['Species'], dtype = np.int32)
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=0)
from sklearn.svm import SVC
model = SVC()
model.fit(x_train, y_train)
iris_anonim = np.array([[6.5, 3.0, 7, 4.8]])
print(model.predict(iris_anonim))
from sklearn.metrics import accuracy_score

y_pred = model.predict(x_train)
print(accuracy_score(y_train, y_pred))

y_pred_test = model.predict(x_test)
print(accuracy_score(y_test, y_pred_test))

from sklearn.model_selection import GridSearchCV

params ={
    'kernel' : ['poly', 'rbf', 'sigmoid'],
    'C' : [0.5, 1, 10, 100],
    'gamma' : ['scale', 1, 0.1, 0.001]
}

grid_search = GridSearchCV(estimator = SVC(), param_grid=params, n_jobs=3, verbose=1, scoring='accuracy')

grid_search.fit(x_train, y_train)

print(grid_search.best_score_)

best_params = grid_search.best_estimator_.get_params()

for param in params :
    print (f"{param} : {best_params[param]}")


