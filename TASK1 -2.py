import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt # MATLAB-like way of plotting

# sklearn package for machine learning in python:
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.datasets import make_regression
from sklearn.metrics import mean_squared_error, r2_score

# read data (make sure .csv in folder)
df = pd.read_csv('C:/Users/dessy/Downloads/houseprice_data.csv')
df.head()
# print(df.head())

print(df.head()) # print first 5 rows of data
print(df.tail()) # print last 5 rows of data
print(df.describe()) # print mean, standard deviation, max, min, etc.
print(df.info()) # prints info about a DataFrame, dtype, non-null values, etc.
print(df.corr(),'\n') # print correlation coefficient for data
X = df.iloc[:, [3]].values # inputs sqft_living
y = df.iloc[:, 0].values # outputs price



# visualise initial data set
fig1, ax1 = plt.subplots()
ax1.scatter(X, y, color='black')
ax1.set_xlabel('sqft_lot')
ax1.set_ylabel('House Price')
fig1.savefig('HOUSE PRICE LR_initial_plot.png')

# split the data into training and test sets:
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size= 1/3,
random_state=0)

# fit the linear least-squares regression line to the training data:
regr = LinearRegression()
regr.fit(X_train, y_train)

# The coefficients
print('Coefficients: ', regr.coef_)

# The intercept
print('Intercept: ', regr.intercept_)

# The mean squared error
print('Mean squared error: %.8f'
% mean_squared_error(y_test, regr.predict(X_test)))

# The mean squared error
print('Coefficient of determination: %.2f'
% r2_score(y_test, regr.predict(X_test)))

# visualise training data set results
fig2, ax2 = plt.subplots()
ax2.scatter(X_train, y_train, color='black')
ax2.plot(X_train, regr.predict(X_train), color='purple')
ax2.set_xlabel('sqft_lot')
ax2.set_ylabel('House price')
fig2.savefig('HOUSE PRICE LR_train_plot.png')

# visualise test data set results
fig3, ax3 = plt.subplots()
ax3.scatter(X_test, y_test, color='black')
ax3.plot(X_test, regr.predict(X_test), color='purple')
ax3.set_title('PRICE DETERMINATION BY SQFT_LOT')
ax3.set_xlabel('sqft_lot')
ax3.set_ylabel('Price of a house')
fig3.savefig('HOUSE PRICE LR_test_plot.png')