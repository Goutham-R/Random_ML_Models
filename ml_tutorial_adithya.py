# -*- coding: utf-8 -*-
"""
Created on Sun May 10 21:38:28 2020

@author: User
"""



import pandas 
import numpy 
           #for storing our model for future use
dataset_url="http://mlr.cs.umass.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv"
dt=pandas.read_csv(dataset_url,sep= ";")
dt.shape


#Conscidering using classification
from sklearn.svm import SVC
X=dt.iloc[:,:11].values
y=dt.iloc[:,-1].values
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
LE=LabelEncoder()
y=LE.fit_transform(y)
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=101)
sc=StandardScaler()

X_train=sc.fit_transform(X_train)
X_test=sc.transform(X_test)

sv=SVC(C=1.0,kernel='rbf',gamma=0.01,class_weight='balanced',verbose=10)
sv.fit(X_train,y_train)
y_pred=sv.predict(X_test)
from sklearn.metrics import classification_report,confusion_matrix
print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))

from sklearn.model_selection import GridSearchCV
parameters={'C':[0.1,1,10,0.01,],'kernel':['rbf','sigmoid','linear'],'gamma':[0.01,0.001,1,0.02,10]}
gs=GridSearchCV(estimator=SVC(), param_grid=parameters,cv=10)
gs.fit(X_train,y_train)
y_pred=gs.predict(X_test)
print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))
print(gs.best_params_)

#Conscidering using regression
from sklearn.linear_model import LinearRegression
lr=LinearRegression()
lr.fit(X_train,y_train)
y_pred=lr.predict(X_test)
from sklearn.metrics import mean_absolute_error
error=mean_absolute_error(y_test, y_pred)




from sklearn.svm import SVR
from sklearn.model_selection import GridSearchCV
parameters={'C':[1],'kernel':['rbf'],'gamma':[0.05]}
gs=GridSearchCV(estimator=SVR(), param_grid=parameters,cv=10,verbose=10)
gs.fit(X_train,y_train)
y_pred=gs.predict(X_test)
error=mean_absolute_error(y_test, y_pred)
gs.best_params_


from sklearn.ensemble import RandomForestRegressor
new=RandomForestRegressor(n_estimators=200)
new.fit(X_train,y_train)
y_pred=new.predict(X_test)
error=mean_absolute_error(y_test, y_pred)


from sklearn.ensemble import RandomForestClassifier
param_grid = { 
    'n_estimators': [200, 500],
    'max_features': ['auto', 'sqrt', 'log2'],
    'max_depth' : [4,5,6,7,8],
    'criterion' :['gini', 'entropy']
}
gs=GridSearchCV(RandomForestClassifier(), param_grid,cv=5,verbose=10)
gs.fit(X_train,y_train)
y_pred=gs.predict(X_test)
print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))
print(gs.best_params_)


new=RandomForestClassifier(n_estimators=800)
new.fit(X_train,y_train)
y_pred=new.predict(X_test)
print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))


from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
pipe=Pipeline([('pca',PCA()),('rnd',RandomForestClassifier())])
param_grid = { 
    'rnd__n_estimators': [200, 500],
    'rnd__max_features': ['auto', 'sqrt', 'log2'],
    'rnd__max_depth' : [4,5,6,7,8],
    'rnd__criterion' :['gini', 'entropy'],
    'pca__n_components':[6,7,8,9]
}
gs=GridSearchCV(pipe, param_grid,verbose=10)
gs.fit(X_train,y_train)
y_pred=gs.predict(X_test)

from sklearn.ensemble import RandomForestClassifier
fr=RandomForestClassifier(n_estimators=200,max_features='log2',max_depth=8,criterion='entropy')
fr.fit(X_train,y_train)
y_pred=fr.predict(X_test)
print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))


from sklearn.neighbors import KNeighborsClassifier
est=list(range(40,80))
error=[]
for i in est:
    knn=KNeighborsClassifier(n_neighbors=i)
    knn.fit(X_train,y_train)
    y_pred=knn.predict(X_test)
    error.append(abs((y_test-y_pred)).mean());
import seaborn as sns

import matplotlib.pyplot as plt
plt.plot(est,error)

knn=KNeighborsClassifier(n_neighbors=74)
knn.fit(X_train,y_train)
y_pred=knn.predict(X_test)
from sklearn.metrics import classification_report,confusion_matrix
print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))

sns.heatmap(dt.corr(),cmap='plasma',annot=True)
dt.drop(['fixed acidity','alcohol','citric acid','sulphates'],axis=1,inplace=True)
X=dt.iloc[:,:7].values
y=dt.iloc[:,-1].values
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
LE=LabelEncoder()
y=LE.fit_transform(y)
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=101)
sc=StandardScaler()

X_train=sc.fit_transform(X_train)
X_test=sc.transform(X_test)


from sklearn.model_selection import GridSearchCV
parameters={'C':[0.1,1,10,0.01,],'kernel':['rbf','sigmoid','linear'],'gamma':[0.01,0.001,1,0.02,10]}
gs=GridSearchCV(estimator=SVC(), param_grid=parameters,cv=10)
gs.fit(X_train,y_train)
y_pred=gs.predict(X_test)
print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))
print(gs.best_params_)
