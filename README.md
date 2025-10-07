# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored
## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. The program begins by generating a random dataset with two features and corresponding binary labels based on a linear condition. A bias term (a column of ones) is added to the features, and the parameters (theta) are initialized to zeros for training the logistic regression model.

2.The sigmoid function is defined as the hypothesis, which converts the linear combination of inputs into probabilities ranging between 0 and 1. This function is the core of logistic regression, allowing the model to separate two classes effectively.

3.Gradient descent is applied iteratively for a fixed number of epochs. In each step, predictions are computed, gradients are calculated, and parameters are updated using the learning rate. This optimization process gradually reduces the error and improves model performance.

4.After training, predictions are generated using the learned parameters, and accuracy is measured by comparing them with actual labels. Finally, the dataset and the decision boundary are plotted to visualize the classifier’s effectiveness. 

## Program:
```
/*
Program to implement the simple linear regression model for predicting the marks scored.
Developed by:Pruthvisri A 
RegisterNumber:25013683
*/

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
data = {
    'Hours': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    'Marks': [20, 25, 35, 45, 50, 60, 65, 70, 80, 85]
}

df = pd.DataFrame(data)
print("Dataset:\n", df)
X = df[['Hours']]
y = df['Marks']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print("\nActual vs Predicted Marks:")
comparison = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
print(comparison)
print("\nMean Absolute Error:", mean_absolute_error(y_test, y_pred))
print("R2 Score:", r2_score(y_test, y_pred))
plt.scatter(X, y, color='blue', label='Actual Data')
plt.plot(X, model.predict(X), color='red', label='Regression Line')
plt.title('Hours vs Marks (Simple Linear Regression)')
plt.xlabel('Hours Studied')
plt.ylabel('Marks Scored')
plt.legend()
plt.show()
hours = float(input("\nEnter study hours to predict marks: "))
predicted_marks = model.predict([[hours]])
print(f"Predicted Marks for  {hours} hours of study = {predicted_marks[0]:.2f}")

```

## Output:
![alt text](<exp 2.png>)
![alt text](<exp 2 s-1.png>)




## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
