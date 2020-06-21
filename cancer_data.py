import pandas as pd
from sklearn import preprocessing
url="/home/goutham/data_analysis_and_ML/cancer-databases.csv"
cancer_data=pd.read_csv(url,sep=";") 
cols=["Database full name","Country id","Database licence"]
cancer_data=cancer_data.drop(cols,axis=1)                
print(cancer_data.dtypes)
le=preprocessing.LabelEncoder()

