import pandas as pd
import numpy as np
import sys
from sklearn.model_selection import GridSearchCV,train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.externals import joblib 
from sklearn.metrics import mean_squared_error
#path="/home/goutham/data_analysis_and_ML/first_model.pkl"
#clf2=joblib.load(path)
index=[i for i in np.linspace(100,500,100,dtype=int)]
print(sys.path)
#print(clf2.get_params().keys())
#pipe=make_pipeline(clf2)
#hyperparameters={"clf2__randomforestregressor__n_estimators" : index}
#clf3=GridSearchCV(pipe,hyperparameters,cv=10)
#dataset_url="http://mlr.cs.umass.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv"
#dt=pd.read_csv(dataset_url,sep= ";")
#y=dt.quality
#x=dt.drop('quality',axis=1) #axis=1 specifies that column is to be dropped, not row
#x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=1,stratify=y)  #strtify for target variable is a good practice to be followed
#clf3.fit(x_train,y_train)
#y_predict=clf3.predict()
#score=mean_squared_error(y_test,y_predict)
#print(score)
