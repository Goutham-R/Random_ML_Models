import pandas #for reading the csv file
import numpy 
from sklearn.model_selection import train_test_split  #to split data inoto train and test databases
from sklearn import preprocessing   #for standardization of data
from sklearn.ensemble import RandomForestRegressor  #imprting a random forst family
from sklearn.pipeline import make_pipeline          #for cross-validation
from sklearn.model_selection import GridSearchCV    #for cross-validation
from sklearn.metrics import mean_squared_error,r2_score
from sklearn.externals import joblib                #for storing our model for future use
dataset_url="http://mlr.cs.umass.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv"
dt=pandas.read_csv(dataset_url,sep= ";")
# print(dt.head())
# print(dt.shape)
#print(dt.describe())
#splitting data to traning and test data sets
y=dt.quality
x=dt.drop('quality',axis=1) #axis=1 specifies that column is to be dropped, not row
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=1,stratify=y)  #strtify for target variable is a good practice to be followed
#scaler = preprocessing.StandardScaler().fit(x_train)        # mean and variance of training set is now stored in scaler
#x_train_scaled=scaler.transform(x_train)                    #standardized data now in the variable
pipeline = make_pipeline(preprocessing.StandardScaler(),RandomForestRegressor(n_estimators=100))
#print(pipeline.get_params())
hyperparameters = { 'randomforestregressor__max_features' : ['auto', 'sqrt', 'log2'],'randomforestregressor__max_depth': [None, 5, 3, 1]}
#now preparing for k-fold cross-validation
clf = GridSearchCV(pipeline, hyperparameters, cv=10)   #cv=10 refers to 10 fold CV
# Fit and tune model
clf.fit(x_train, y_train)
#print (clf.best_params_)
print(clf.refit)
y_predict=clf.predict(x_test)
print (r2_score(y_test,y_predict))
print(mean_squared_error(y_test,y_predict))
joblib.dump(clf, 'first_model.pkl')         #saves this model to a file
#clf2 = joblib.load('first_model.pkl')     #to load the model for future use
