import sys
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler,MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline,make_pipeline
from sklearn.linear_model import LogisticRegression
name=["Pregnancies",                 
"Glucose",                     
"BloodPressure",               
"SkinThickness",               
"Insulin",                     
"BMI",                         
"DiabetesPedigreeFunction",    
"Age",                         
"Outcome"]
index=[i for i in np.linspace(0,769,1)]
file_path="/home/goutham/data_analysis_and_ML/pima-indians-diabetes.csv"
dt=pd.read_csv(file_path,names=name)
sns.set(color_codes=True)
#print(dt.head())
x=dt.drop("Outcome",axis=1)
y=dt["Outcome"]
#sns.heatmap(dt.isnull())
#sns.distplot(dt["Pregnancies"])
#sns.boxplot(x=dt["Pregnancies"],y=dt["DiabetesPedigreeFunction"])
#sns.pairplot(data=dt)
#plt.show()
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=69)
cv_results = cross_val_score(LogisticRegression(), x_train, y_train, cv=10, scoring="accuracy")
print(cv_results.mean())
print("-*-"*25)
pipe=make_pipeline(MinMaxScaler(),LogisticRegression())
cv_results = cross_val_score(pipe, x_train, y_train, cv=10, scoring="accuracy")
print(cv_results.mean())
pipe=make_pipeline(StandardScaler(),LogisticRegression())
cv_results = cross_val_score(pipe, x_train, y_train, cv=10, scoring="accuracy")
print(cv_results.mean())