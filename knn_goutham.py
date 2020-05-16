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




#Using Classification Techniques
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

#I dont know why but in most cases we fit the train set then transform the test set
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=69,shuffle=True)
X_train=sc.fit_transform(x_train)
X_test=sc.transform(x_test)
from sklearn.neighbors import KNeighborsClassifier
knn=KNeighborsClassifier()
knn.fit(X_train,y_train)
y_pred=knn.predict(X_test)
from sklearn.metrics import mean_squared_error,classification_report,confusion_matrix

#Mean Squared Error is used for regression since we have integer scores we can use classification
print(mean_squared_error(y_test,y_pred)) 
#MSE=0.56875
print(classification_report(y_test,y_pred))
#Avg Accuracy is 0.56 almost same as MSE

#Now we can tune our KNN
#I will just tune the n_neighbours parameter


#Even though n_neighbours=1 gives better result it is likely to overfit the data-0.60
knn=KNeighborsClassifier(n_neighbors=1)
knn.fit(X_train,y_train)
y_pred=knn.predict(X_test)
print(mean_squared_error(y_test,y_pred)) 
print(classification_report(y_test,y_pred))

#n_neighbors=2 0.57
knn=KNeighborsClassifier(n_neighbors=2)
knn.fit(X_train,y_train)
y_pred=knn.predict(X_test)
print(mean_squared_error(y_test,y_pred)) 
print(classification_report(y_test,y_pred))

#n_neighbors=3 0.57
knn=KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train,y_train)
y_pred=knn.predict(X_test)
print(mean_squared_error(y_test,y_pred)) 
print(classification_report(y_test,y_pred))

#n_neighbors=4 0.58
knn=KNeighborsClassifier(n_neighbors=4)
knn.fit(X_train,y_train)
y_pred=knn.predict(X_test)
print(mean_squared_error(y_test,y_pred)) 
print(classification_report(y_test,y_pred))

#n_neighbors=5 0.56
knn=KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train,y_train)
y_pred=knn.predict(X_test)
print(mean_squared_error(y_test,y_pred)) 
print(classification_report(y_test,y_pred))

import  matplotlib.pyplot  as plt
arr=list(range(1,60))
score=[]
from sklearn.model_selection import cross_val_score
for i in arr:
    knn=KNeighborsClassifier(n_neighbors=i)
    scores=cross_val_score(knn,X_train,y_train,cv=10,scoring='accuracy')
    score.append(scores.mean())
max(score)#0.63
mse=[1-x for x in score]

plt.figure(figsize=(12,6))
plt.plot(arr,mse)
plt.title('Accuracy Graph')
plt.xlabel("Number of Neighbours")
plt.ylabel("Misclassification Error Rates")
