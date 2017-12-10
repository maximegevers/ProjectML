import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.grid_search import GridSearchCV
from sklearn.cross_validation import cross_val_score
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC

#from sklearn.preprocessing import normalize

dataset = np.genfromtxt('wdbc.csv', delimiter=',', converters ={0:lambda x: 1.0*int(x[0] == 77)})

x = dataset[:,1:]
y = dataset[:, 0]
#normalize(x)
scaler = StandardScaler()
x = scaler.fit_transform(x)
x_train, x_test, y_train, y_test = train_test_split(x, y,  test_size = .2, shuffle = False)

lr = LogisticRegression()
lr.fit(x_train, y_train)
lin_svm = svm.LinearSVC()
lin_svm.fit(x_train, y_train)
print('LOGISTIC REGRESSION \n')
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

print('LINEAR SVM \n')
lin_svm_param_grid = [
  {'C': [1, 10, 100, 1000]}
 ]
lin_svmgrid = GridSearchCV(lin_svm, lin_svm_param_grid, cv=10, scoring='accuracy')
lin_svmgrid.fit(x_train, y_train)
print('10-fold CV lin svm score on train', lin_svmgrid.best_score_)
print('best SVM parameters are: ', lin_svmgrid.best_params_)
lin_svmbest = svm.LinearSVC(C=1)
lin_svmbest.fit(x_train, y_train)
print('default lin svm on test ', lin_svm.score(x_test, y_test))
print('10-fold CV lin svm score on test: ', lin_svmbest.score(x_test, y_test))

print('SVM SVC \n')

svm=svm.SVC()
svm.fit(x_train, y_train)
print('default svm svc score on train ', svm.score(x_train, y_train))

svm_param_grid = [
  {'C': [1, 10, 100, 1000], 'kernel': ['linear']},
  {'C': [1, 10, 100, 1000], 'kernel': ['rbf']},
 ]
svmgrid = GridSearchCV(svm, svm_param_grid, cv=10, scoring='accuracy')
svmgrid.fit(x_train, y_train)
print('10-fold CV svm svc score on train', svmgrid.best_score_)
print('best parameters are: ', svmgrid.best_params_)
svmbest = svm(C = 1, kernel = 'linear')
svmbest.fit(x_train, y_train)
print('default svm svc on test ', svm.score(x_test, y_test))
print('10-fold CV svm svc score on test: ', svmbest.score(x_test, y_test))

print('RANDOM FORREST \n')
randfor=RandomForestClassifier()
randfor.fit(x_train, y_train)
print('default rand for on train ', randfor.score(x_train, y_train))

randfor_param_grid=[{'n_estimators':[10, 500, 1000], 'max_features': ['auto','log2', None], 'criterion': ['gini', 'entropy']}]
randforgrid = GridSearchCV(randfor, randfor_param_grid, cv=10, scoring='accuracy')
randforgrid.fit(x_train, y_train)
print('10-fold CV rand for score on train', randforgrid.best_score_)
print('best parameters are: ', randforgrid.best_params_)
randforbest = RandomForestClassifier(criterion = 'entropy', max_features = 'auto', n_estimators = 500)
randforbest.fit(x_train, y_train)
print('default rand for on test ', randfor.score(x_test, y_test))
print('10-fold CV rand for score on test: ', randforbest.score(x_test, y_test))


print('NEURAL NETWORKS \n') 
nn=MLPClassifier()
nn.fit(x_train, y_train)
print('default nn on train ', nn.score(x_train, y_train))
print('default parameters are: ', nn.get_params())
nn_param_grid=[{'learning_rate' : ['constant', 'invscaling', 'adaptive'], 
                'momentum':[0.1,0.3,0.9], 
                'activation':['identity', 'logistic', 'tanh', 'relu']}]
nngrid = GridSearchCV(nn, nn_param_grid, cv=10, scoring='accuracy')
nngrid.fit(x_train, y_train)
print('10-fold CV nn score on train', nngrid.best_score_)
print('best parameters are: ', nngrid.best_params_)
nnbest=MLPClassifier(activation = 'identity', learning_rate = 'constant', momentum = 0.9)
nnbest.fit(x_train, y_train)
print('default nn on test ', nn.score(x_test, y_test))
print('10-fold CV nn score on test: ', nnbest.score(x_test, y_test))

#def cv():
#k_scores = []
#for c in [0.001, 0.003, 0.01,0.03, 0.1, 0.3, 1]:
#    log = LogisticRegression(C = c)
#    scores = cross_val_score(log, x, y, cv=10, scoring='accuracy')
#    k_scores.append(scores.mean())
#    print(k_scores)

def trainer (cls, param_grid):
    cls.fit(x_train, y_train)
    print('train accuracy is: ', cls.score(x_train,y_train), '\n', 'test accuracy is: ', cls.score(x_test, y_test))
    grid = GridSearchCV(cls, param_grid, cv = 10, scoring= 'accuracy')
    grid.fit(x_train,y_train)
    print('cv train accuracy is: ', grid.score(x_train,y_train), '\n', 'cv test accuracy is: ', grid.score(x_test, y_test))
    print('best parameters for this classifier are: ', grid.best_params_)
