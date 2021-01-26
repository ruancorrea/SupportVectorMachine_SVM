from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.svm import SVC
from matplotlib import pyplot as plt
import numpy as np 
import pandas as pd 

cancer = load_breast_cancer()
X = cancer.data[:, :2]
y = cancer.target

train_X, test_X, train_y, test_y = train_test_split(X, y, test_size=0.3)

model = SVC(kernel='linear')
model.fit(train_X[:,:2], train_y)

pred = model.predict(test_X)

print(classification_report(test_y, pred))
print('\n')
print(confusion_matrix(test_y,pred))

# Grid Search

# plotando grafico

h = .02 
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)
plt.contourf(xx, yy, Z, cmap=plt.cm.coolwarm, alpha=0.8)
plt.scatter(train_X[:, 0], train_X[:, 1], c=train_y, cmap='winter') 
plt.show()