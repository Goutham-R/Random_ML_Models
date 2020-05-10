#loads the iris data set, which is initially in a CSV format
import pandas
url = "https://gist.githubusercontent.com/curran/a08a1080b88344b0c8a7/raw/639388c2cbc2120a14dcf466e85730eb8be498bb/iris.csv"
dataset=pandas.read_csv(url)
print(dataset.head(10))