# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Import the standard Libraries.
2.Set variables for assigning dataset values.
3.Import linear regression from sklearn.
4.Assign the points for representing in the graph.
5.Predict the regression for marks by using the representation of the graph.
6.Compare the graphs and hence we obtained the linear regression for the given datas.

## Program:
```
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error,mean_squared_error
df=pd.read_csv('student_scores.csv')
print(df)
df.head(0)
df.tail(0)
print(df.head())
print(df.tail())
x = df.iloc[:,:-1].values
print(x)
y = df.iloc[:,1].values
print(y)
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=1/3,random_state=0)
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x_train,y_train)
y_pred = regressor.predict(x_test)
print(y_pred)
print(y_test)
#Graph plot for training data
plt.scatter(x_train,y_train,color='black')
plt.plot(x_train,regressor.predict(x_train),color='blue')
plt.title("Hours vs Scores(Training set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()
#Graph plot for test data
plt.scatter(x_test,y_test,color='black')
plt.plot(x_train,regressor.predict(x_train),color='red')
plt.title("Hours vs Scores(Testing set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()
mse=mean_absolute_error(y_test,y_pred)
print('MSE = ',mse)
mae=mean_absolute_error(y_test,y_pred)
print('MAE = ',mae)
rmse=np.sqrt(mse)
print("RMSE= ",rmse)
/*
Program to implement the simple linear regression model for predicting the marks scored.
Developed by: G.Ramanujam 
RegisterNumber:  212224240129
*/
```

## Output:
## Dataset


![Screenshot 2025-05-01 214036](https://github.com/user-attachments/assets/2c2718f1-6dcb-4065-8aa2-97c867cee045)



## Head Values


![Screenshot 2025-05-01 214309](https://github.com/user-attachments/assets/70edcfbd-5583-407a-ad9b-c25297437647)

## Tail Values


![Screenshot 2025-05-01 214431](https://github.com/user-attachments/assets/98e60c7d-68a9-41b3-b89c-a2c5fa2eb375)



## X and Y values


![Screenshot 2025-05-01 214923](https://github.com/user-attachments/assets/b0d5f20e-1314-48c6-8140-76a01d0fe484)


## Predication values of X and Y


![Screenshot 2025-05-01 215028](https://github.com/user-attachments/assets/398ebc69-d2f4-49ed-b704-ef281642b870)


## MSE,MAE and RMSE


![Screenshot 2025-05-01 215132](https://github.com/user-attachments/assets/d2965bd6-9b70-455a-994f-ec165e2c33da)


## Training Set


![Screenshot 2025-05-01 215506](https://github.com/user-attachments/assets/07dcf91f-dab5-473f-8e0e-7b34985118c6)


## Testing Set


![Screenshot 2025-05-01 215630](https://github.com/user-attachments/assets/394aaffc-7ef6-4b58-8646-1ae7cce8a84f)


## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
