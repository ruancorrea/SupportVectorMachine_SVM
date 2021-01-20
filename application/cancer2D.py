from sklearn.datasets import load_breast_cancer ## dados de pacientes com ou sem cancer de pulmao.
from sklearn.model_selection import train_test_split ## modelo de seleçao para quebra de dados
from sklearn.metrics import classification_report, confusion_matrix ## usados para avaliar modelos de classificação
from sklearn.svm import SVC ## classe do modelo SVM, onde é importado o SVC que significa Support Vector Classification
from matplotlib import pyplot as plt 
import numpy as np
import pandas as pd 

pd.set_option('display.max_columns', None)  
cancer = load_breast_cancer() ## cancer é um dicionario
X = cancer.data
y = cancer.target
## criando dataFrame
df_cancer = pd.DataFrame(cancer['data'], columns=cancer['feature_names'])
## print(df_cancer.head())

X = X[:, :2]
h = .02 

df_target = pd.DataFrame(cancer['target'], columns=['Cancer'])
## print(df_target.head())

train_X, test_X, train_y, test_y = train_test_split(X, y, test_size=0.3, random_state=109)
## print(train_y)

## definindo o modelo de classificação SVC
modelo = SVC(kernel='linear', C=1)
modelo.fit(train_X[:,:2], train_y)

## avaliando modelo
pred = modelo.predict(test_X)
print(classification_report(test_y, pred))
print('\n')
print(confusion_matrix(test_y, pred))


## plotando gráfico
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

Z = modelo.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)
plt.contourf(xx, yy, Z, cmap=plt.cm.coolwarm, alpha=0.8)
plt.scatter(train_X[:, 0], train_X[:, 1], c=train_y, cmap='winter') 
plt.show()
