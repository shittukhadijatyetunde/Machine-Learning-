import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt # MATLAB-like way of plotting

# sklearn package for machine learning in python:
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

from sklearn.datasets import make_classification
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer


df = pd.read_csv('C:/Users/dessy/Downloads/nba_rookie_data.csv')
print(df.head(),'\n')
print(df.info(), '\n')

df['3 Point Percent'].fillna(df['3 Point Percent'].mean(), inplace= True)

label_list = ['Name']
for l in label_list:
 df[l] = LabelEncoder().fit_transform(df[l])
 
 
# Assign featyres to data set x and y
print(df.head(),'\n')
X = df.iloc[:,[1]].values #input
y = df.iloc[:,-1].values #target


# split the data into training and test sets:
X_train, X_test, y_train, y_test = train_test_split(X, y,
test_size= 1/3, random_state=2)


scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


# setup the neural network architecture
mlp = MLPClassifier(hidden_layer_sizes=(10,50,20),
activation="logistic" ,random_state=0, max_iter=2000)

mlp.fit(X_train, y_train)

print('value prediction %d:' %mlp.predict([[15]]))

# performance metrics
print('Our Accuracy is %.2f' % mlp.score(X_test, y_test))

print('Number of mislabeled points out of a total %d points : %d'
% (X_test.shape[0], (y_test != mlp.predict(X_test)).sum()))


# Set up Logistic regresiion
logReg = LogisticRegression()
logReg.fit(X_train, y_train)
Logistic_predictions = logReg.predict(X_test)

y_pred = logReg.predict([[30]])
print('Predict a value:', y_pred)

# output the accuracy score
print('Our Accuracy is %.2f' % logReg.score(X_test, y_test))

# output the number of mislabeled points
print('Number of mislabeled points out of a total %d points : %d'
% (X_test.shape[0], (y_test != logReg.predict(X_test)).sum()))



gnb = GaussianNB()
gnb.fit(X_train, y_train)
gnb_predictions = gnb.predict(X_train)
y_pred = gnb.predict([[30]])
print('gnb Predict value:', y_pred)


# accuracy score
print('Number of mislabeled points out of a total of %d points: %d'
% (X_test.shape[0], (y_test != gnb.predict(X_test)).sum()))

# number of mislabeled points
print('Our accuracy is %.2f:' % gnb.score(X_test, y_test))




# visualise the model
fig1, ax1 = plt.subplots()
ax1.scatter(X_test, y_test, color='blue')
ax1.scatter(X_test, gnb.predict(X_test), color='red', marker='*')
ax1.scatter(X_test, gnb.predict_proba(X_test)[:,1], color='green', marker='.')
ax1.set_ylabel('NBA>5')
ax1.set_xlabel('GAMES PLAYED')
fig1.savefig('Class_plot.png')