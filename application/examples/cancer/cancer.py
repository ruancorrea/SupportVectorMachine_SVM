from sklearn.datasets import load_breast_cancer ## dados de pacientes com ou sem cancer de pulmao.
from sklearn.model_selection import train_test_split ## modelo de seleçao para quebra de dados
from sklearn.metrics import classification_report, confusion_matrix ## usados para avaliar modelos de classificação
from sklearn.svm import SVC ## classe do modelo SVM, onde é importado o SVC que significa Support Vector Classification
from sklearn.model_selection import GridSearchCV ## grid search  é um processo para encontrar os melhores parâmetros para o nosso modelo, fazendo com que ele tenha um desempenho melhor.
from matplotlib import pyplot as plt 
import numpy as np
import pandas as pd 

cancer = load_breast_cancer() ## cancer é um dicionario
##print(cancer(['DESCR'])) ## 569 instancias

## criando dataFrame
df_cancer = pd.DataFrame(cancer['data'], columns=cancer['feature_names'])
## print(df_cancer.head())

df_cancer["target"] = pd.Series(cancer.target)

## dataFrame de target para criação do grafico(coordenadas)
## por conta do aprendizado supervisionado
## print(df_target.head())

train_X, test_X, train_y, test_y = train_test_split(df_cancer, np.ravel(df_cancer["target"]), test_size=0.3)
## print(train_y)

## definindo o modelo de classificação SVC
model = SVC(kernel = "linear")
model.fit(train_X, train_y)

## avaliando modelo
pred = model.predict(test_X)
print(classification_report(test_y, pred))
print('\n')
print(confusion_matrix(test_y, pred))

## utilizando grid search
## param_grid = {"C": [0.1, 1, 10, 100, 1000], "gamma": [1, 0.1, 0.01, 0.001, 0.0001], "kernel": ['rbf']}
## grid_svm = GridSearchCV(SVC(), param_grid, refit=True, verbose=3)
## grid_svm.fit(train_X, train_y)
## pred_grid = grid_svm.predict(test_X)
## print('\n GRID SEARCH')
## print(classification_report(test_y, pred_grid))
## print('\n')
## print(confusion_matrix(test_y, pred_grid))