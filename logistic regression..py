import pandas as pd
import numpy as np 
import matplotlib as plt
import sklearn
from pandas import read_csv

data = pd.read_csv("C:\\Users\\USER\\Downloads\\framingham.csv")

print(data)
data.dtypes

#Outlier check

data["age"].plot(kind = "box", layout =(3,3))
data["education"].plot(kind = "box", layout =(3,3))
data["currentSmoker"].plot(kind = "box", layout =(3,3))
data["cigsPerDay"].plot(kind = "box", layout =(3,3))
data["BPMeds"].plot(kind = "box", layout =(3,3))
data["prevalentStroke"].plot(kind = "box", layout =(3,3))
data["prevalentHyp"].plot(kind = "box", layout =(3,3))
data["diabetes"].plot(kind = "box", layout =(3,3))
data["totChol"].plot(kind = "box", layout =(3,3))
data["sysBP"].plot(kind = "box", layout =(3,3))
data["diaBP"].plot(kind = "box", layout =(3,3))
data["BMI"].plot(kind = "box", layout =(3,3))
data["glucose"].plot(kind = "box", layout =(3,3))
data["TenYearCHD"].plot(kind = "box", layout =(3,3))

#No extreme outliers

data["education"].describe()
#cleaning the dataset
#checking which variables have missing data
missing_counts = data.isna().sum()
print(missing_counts)

#Replace missing values with the mean
#Education
if data["education"].isna().sum() > 0:
   data["education"].fillna(data["education"].mean(), inplace=True)

#cigsPerDay 

if data["cigsPerDay"].isna().sum() > 0:
       data["cigsPerDay"].fillna(data["cigsPerDay"].mean(), inplace=True)

#BPMeds

if data["BPMeds"].isna().sum() > 0:
   data["BPMeds"].fillna(data["BPMeds"].mean(), inplace=True)

#totChol

if data["totChol"].isna().sum() > 0:
   data["totChol"].fillna(data["totChol"].mean(), inplace=True)

#BMI

if data["BMI"].isna().sum() > 0:
   data["BMI"].fillna(data["BMI"].mean(), inplace=True)

#heartRate

if data["heartRate"].isna().sum() > 0:
   data["heartRate"].fillna(data["heartRate"].mean(), inplace=True)

#glucose

if data["glucose"].isna().sum() > 0:
   data["glucose"].fillna(data["glucose"].mean(), inplace=True)

missing_counts = data.isna().sum()
print(missing_counts)

data.info()

#Checking for duplicates
duplicates = data[data.duplicated()]
print("Number of duplicates:", duplicates.shape[0])

#correlation
correlations = data.corr()
print(correlations)




data.plot(kind= "hist", layout=(3,3), sharex = True, sharey=True)

y = data["male"]
x = data.drop(columns=["male"])

x.info()
y.info()

from sklearn import linear_model
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix


#creating a model and training it
model = LogisticRegression(solver = "liblinear", C=10.0, random_state = 0)
model.fit(x,y)

#Evaluating the model
p_pred = model.predict_proba(x)
y_pred = model.predict(x)
score_ = model.score(x,y)
conf_m = confusion_matrix(y, y_pred)
report = classification_report(y, y_pred)

print("intercept", model.intercept_)
print("coef:",model.coef_)

print("y_actual:", y)
print("y_pred:", y_pred)

print(conf_m)
print(report)



plt.plot(x, y_pred, kind = 'smooth')
plt.xlabel('x')
plt.ylabel('predicted probability')
plt.title('Sigmoid Curve')
plt.show()


prediction = model.predict(1)
print(prediction)
