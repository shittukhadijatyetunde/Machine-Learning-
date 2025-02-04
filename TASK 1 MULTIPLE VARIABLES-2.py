import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt # MATLAB-like way of plotting
from mpl_toolkits.mplot3d import Axes3D


# sklearn package for machine learning in python:
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
df = pd.read_csv('C:/Users/dessy/Downloads/houseprice_data.csv')

 # read data (make sure .csv in folder)
print(df.head(),'\n') # print first 5 rows of data
X = df.iloc[:, [9,3,]].values # inputs
y = df.iloc[:, 0].values # target

# split the data into training and test sets:
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 1/3,
random_state = 0)

# fit the linear least-squares regression line to the training data:
regr = LinearRegression()
regr.fit(X_train, y_train)

# The coefficients
print('Coefficients: ', regr.coef_)

# The coefficients
print('Intercept: ', regr.intercept_)

# The mean squared error
print('Mean squared error: %.8f'
% mean_squared_error(y_test, regr.predict(X_test)))

# The coefficient of determination: 1 is perfect prediction
print('Coefficient of determination: %.2f'
% r2_score(y_test, regr.predict(X_test)))

#3d plot for multiple linear regression:
fig1 = plt.figure(figsize=(8,7))
ax1 = fig1.add_subplot(111, projection = '3d')
ax1.scatter(X[:,0], X[:,1], y, color = 'PURPLE')

# plot the plane
X1, X2 = np.meshgrid(range(-2,3), range(-2,3))
Z = regr.coef_[0]*X1+regr.coef_[1]*X2+regr.intercept_
ax1.plot_surface(X1, X2, Z, alpha=0.5)
ax1.azim = -60
ax1.dist = 10
ax1.elev = 10
ax1.set_xlabel('GRADE')
ax1.set_ylabel('SQFT_LIVING')
ax1.set_zlabel('PRICE')
fig1.tight_layout(pad=-2.0)
fig1.savefig('HOUSE PRICE MULTIPLE VARIABLES test_plot.png')


