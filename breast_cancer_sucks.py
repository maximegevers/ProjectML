import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.grid_search import GridSearchCV

#from sklearn.preprocessing import normalize

dataset=np.genfromtxt('wdbc.csv', delimiter=',', converters ={0:lambda x: 1.0*int(x[0] == 77)})

x = dataset[:,1:]
y = dataset[:, 0]
#normalize(x)
scaler = StandardScaler()
x = scaler.fit_transform(x)
x_train, x_test, y_train, y_test = train_test_split(x, y,  test_size = .2, shuffle = False)

lr = LogisticRegression()
lr.fit(x_train, y_train)
print('default log reg score on train ', lr.score(x_train, y_train))
#lrange = [0.001, 0.003, 0.01,0.03, 0.1, 0.3, 1]
#param_grid = dict(C = lrange)
param_grid=[{'C':[0.001, 0.003, 0.01,0.03, 0.1, 0.3, 1], 'max_iter':[2,3,4,5,10,20,30]}]
lrgrid = GridSearchCV(lr, param_grid, cv=10, scoring='accuracy')
lrgrid.fit(x_train, y_train)
print('10-fold CV log reg score on train', lrgrid.best_score_)
#print('best parameters are: ', lrgrid.best_params_)
lrbest = LogisticRegression(C = 0.1, max_iter = 4)
lrbest.fit(x_train, y_train)
print('default log reg on test ', lr.score(x_test, y_test))
print('10-fold CV log reg score on test: ', lrbest.score(x_test, y_test))