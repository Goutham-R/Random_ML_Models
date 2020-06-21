dataset_url="http://mlr.cs.umass.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv"
import pandas as pd
import numpy as np
dt=pd.read_csv(dataset_url,sep=";")
print(dt.head())
import seaborn as sns
import matplotlib.pyplot as plt
sns.boxplot(dt["quality"])
plt.show()