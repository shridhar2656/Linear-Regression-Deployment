# importing the libraries

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle

# importing the dataset

dataset = pd.read_csv("Salary_Data.csv")
X = dataset.iloc[:,:-1].values
y = dataset.iloc[:,-1].values


# splitting dataset into training and testing

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=0)

# training the simple linear regression model on training data set

from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train,y_train)

# predicting the test set data
y_pred = regressor.predict(X_test)


# real salary vs predicted
plt.scatter(X_train,y_train , color = 'red')
plt.plot(X_train , regressor.predict(X_train) , color='blue')
plt.title('Salary vs Experience (Training set)')
plt.xlabel('Years of experience')
plt.ylabel('Salary')
plt.show()


plt.scatter(X_test,y_test , color = 'red')
plt.plot(X_train , regressor.predict(X_train) , color='blue')
plt.title('Salary vs Experience (Testing set)')
plt.xlabel('Years of experience')
plt.ylabel('Salary')
plt.show()


#saving model to disk

pickle.dump(regressor,open('model.pkl','wb'))

# loading model to compare the results

model = pickle.load(open('model.pkl','rb'))
print(model.predict([2.5]))
