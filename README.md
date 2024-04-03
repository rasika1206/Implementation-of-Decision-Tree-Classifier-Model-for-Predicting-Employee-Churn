# Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn

## AIM:
To write a program to implement the Decision Tree Classifier Model for Predicting Employee Churn.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. 
2. 
3. 
4. 

## Program:
```
/*
Program to implement the Decision Tree Classifier Model for Predicting Employee Churn.
Developed by: RASIKA M
RegisterNumber:  212222230117
*/
```
import pandas as pd
data=pd.read_csv("/content/Employee_EX6.csv")
data.head()
data.info()
data.isnull().sum()
data["left"].value_counts()
from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
data["salary"]=le.fit_transform(data["salary"])
data.head()
x=data[["satisfaction_level","last_evaluation","number_project","average_montly_hours",
        "time_spend_company","Work_accident","promotion_last_5years","salary"]]
x.head()
y=data["left"]
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=100)
from sklearn.tree import DecisionTreeClassifier
dt=DecisionTreeClassifier(criterion="entropy")
dt.fit(x_train,y_train)
y_pred=dt.predict(x_test)
from sklearn import metrics
accuracy=metrics.accuracy_score(y_test,y_pred)
accuracy
dt.predict([[0.5,0.8,9,260,6,0,1,2]])

## Output:
![Screenshot 2024-04-02 155112](https://github.com/poojasen05/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/150784373/882d3d2e-d594-4aad-84f8-6cd8e90d3430)

![Screenshot 2024-04-02 155146](https://github.com/poojasen05/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/150784373/ebe49a44-989b-4b4a-95ab-ce9d68839899)


![Screenshot 2024-04-02 155155](https://github.com/poojasen05/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/150784373/c28f4b0f-b4fe-43e7-8746-c7788933b38b)
![Screenshot 2024-04-02 155229](https://github.com/poojasen05/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/150784373/8dcac9fc-1347-43c4-8699-ddb4df766c3c)


## Result:
Thus the program to implement the  Decision Tree Classifier Model for Predicting Employee Churn is written and verified using python programming.
