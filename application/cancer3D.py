from sklearn.datasets import load_breast_cancer ## dados .
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split ## modelo de seleçao para quebra de dados
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

cancer = load_breast_cancer()
X = cancer.data[:, :3]  ## apenas pegamos os três primeiros recursos.
Y = cancer.target


train_X, test_X, train_y, test_y = train_test_split(X, Y, test_size=0.3, random_state=109)

## tornando um problema de classificação binária
# X = X[np.logical_or(Y==0,Y==1)]
# Y = Y[np.logical_or(Y==0,Y==1)]

model = SVC(kernel='linear')
clf = model.fit(train_X[:,:3], train_y)

## a equação do plano de separação é dada por todo x de forma que np.dot(svc.coef_[0], x) + b = 0.
## resolva para w3 (z)
z = lambda x,y: (-clf.intercept_[0]-clf.coef_[0][0]*x - clf.coef_[0][1]*y) / clf.coef_[0][2]


tmp = np.linspace(30, 0, 30)
x,y = np.meshgrid(tmp, tmp)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot3D(train_X[train_y==0,0], train_X[train_y==0,1], train_X[train_y==0,2],'ob')
ax.plot3D(train_X[train_y==1,0], train_X[train_y==1,1], train_X[train_y==1,2],'sr')
ax.plot_surface(x, y, z(x,y))
ax.view_init(30, 60)
plt.show()