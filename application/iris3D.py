from sklearn.datasets import load_iris ## dados .
from sklearn.svm import SVC
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

iris = load_iris()
X = iris.data[:, :3]  ## apenas pegamos os três primeiros recursos.
Y = iris.target

## tornando um problema de classificação binária
X = X[np.logical_or(Y==0,Y==1)]
Y = Y[np.logical_or(Y==0,Y==1)]

model = SVC(kernel='linear')
clf = model.fit(X, Y)

## a equação do plano de separação é dada por todo x de forma que np.dot(svc.coef_[0], x) + b = 0.
## resolva para w3 (z)
z = lambda x,y: (-clf.intercept_[0]-clf.coef_[0][0]*x - clf.coef_[0][1]*y) / clf.coef_[0][2]

tmp = np.linspace(-5,5,30)
x,y = np.meshgrid(tmp,tmp)

fig = plt.figure()
ax  = fig.add_subplot(111, projection='3d')
ax.plot3D(X[Y==0,0], X[Y==0,1], X[Y==0,2],'ob')
ax.plot3D(X[Y==1,0], X[Y==1,1], X[Y==1,2],'sr')
ax.plot_surface(x, y, z(x,y))
ax.view_init(30, 60)
plt.show()