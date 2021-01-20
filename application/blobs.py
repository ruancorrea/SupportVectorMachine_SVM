from sklearn.datasets import make_blobs ## dados de um gerador de bolhas gaussianas isotrópicos para agrupamento.
from sklearn.model_selection import train_test_split ## modelo de seleçao para quebra de dados
from sklearn.metrics import classification_report, confusion_matrix ## usados para avaliar modelos de classificação
from sklearn.metrics import confusion_matrix
from sklearn.svm import SVC
from matplotlib import pyplot as plt 
import numpy as np
import pandas as pd 


## n_samples : numero de amostras
## centers : numero de centros 
## cluster_std : O desvio padrão dos clusters.
## random_state : determina a geração de números aleatórios para criação de conjuntos de dados.
X, y = make_blobs(n_samples=125, centers=2, cluster_std=0.60, random_state=0)
train_X, test_X, train_y, test_y = train_test_split(X, y, test_size=0.3, random_state=103)

plt.scatter(train_X[:, 0], train_X[:, 1], c=train_y, cmap='winter')

## definindo o modelo de classificação SVC
svc = SVC(kernel='linear')
svc.fit(train_X, train_y)

## avaliando modelo
pred = svc.predict(test_X)
print(classification_report(test_y, pred))
print('\n')
print(confusion_matrix(test_y, pred))


plt.scatter(train_X[:, 0], train_X[:, 1], c=train_y, cmap='winter')

ax = plt.gca()
xlim = ax.get_xlim()

ax.scatter(test_X[:, 0], test_X[:, 1], c=test_y, cmap='winter', marker='s')

w = svc.coef_[0]
a = -w[0] / w[1]
xx = np.linspace(xlim[0], xlim[1])
yy = a * xx - (svc.intercept_[0] / w[1])

plt.plot(xx, yy)
plt.show()