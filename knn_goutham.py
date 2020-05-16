import pandas
import numpy as np
dataset_url="http://mlr.cs.umass.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv"
dt=pandas.read_csv(dataset_url,sep= ";")
#print(dt.head())
x=dt.drop("quality",axis=1)
print(x.head())
y=dt.iloc[:,-1].values
from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
x=sc.fit_transform(x)
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=69,shuffle=True)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
clf=KNeighborsClassifier()
clf.fit(x_train,y_train)
y_predict=clf.predict(x_test)
from sklearn.metrics import mean_squared_error
print(mean_squared_error(y_test,y_predict))

#SVC Model
from sklearn.svm import SVC
sv=SVC(C=1,gamma=0.1,kernel='linear')
sv.fit(X_train,y_train)
y_pred=sv.predict(X_test)
print(mean_squared_error(y_test,y_pred)) 
print(classification_report(y_test,y_pred))
from sklearn.model_selection import GridSearchCV
param_grid={'C':[1,0.1,0.01],'gamma':[0.1,1,0.01,0.001],'kernel':['linear','rbf']}
gs=GridSearchCV(SVC(),param_grid)
gs.fit(X_train,y_train)
y_pred=gs.predict(X_test)
print(mean_squared_error(y_test,y_pred)) 
print(classification_report(y_test,y_pred))
