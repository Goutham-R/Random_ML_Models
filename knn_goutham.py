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
