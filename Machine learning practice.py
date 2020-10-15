# -*- coding: utf-8 -*-
"""
Created on Wed Dec 18 13:36:56 2019

@author: alex
"""

from sklearn.datasets import load_iris
iris_dataset = load_iris()
import pandas as pd

print("Keys of iris_dataset:\n", iris_dataset.keys())
print()
print("Target names:", iris_dataset['target_names'])
print()
print("Feature names:\n", iris_dataset['feature_names'])
print()
print("Type of data:", type(iris_dataset['data']))
print()
print("First five rows of data:\n", iris_dataset['data'][:5,:])
print()
print("Shape of data:", iris_dataset['data'].shape)
print()
print("Type of target:", type(iris_dataset['target']))
print()
print("Shape of target:", iris_dataset['target'].shape)
print()
print("Target:\n", iris_dataset['target'])
print()
print("Species names:", iris_dataset['target_names'])

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(iris_dataset['data'], 
                                                    iris_dataset['target'], 
                                                    random_state=0)
#%%

# create dataframe from data in X_train
iris_dataframe = pd.DataFrame(X_train, 
                              columns=iris_dataset.feature_names) # label columns using iris_dataset.feature_names


# create a scatter matrix from the dataframe, color by y_train

pd.plotting.scatter_matrix(iris_dataframe,        # data frame
                           c=y_train,             # colour by y_train
                           figsize=(10, 10),
                           marker='o', 
                           hist_kwds={'bins': 20},# plotting keyword arguments to be passed to hist function
                           s=60,                  # size of markers
                           alpha=.8,              # transparency of markers
                           cmap='viridis');   
                           
#%%

import numpy as np                   
from sklearn.neighbors import KNeighborsClassifier # import model

knn = KNeighborsClassifier(n_neighbors=1)       

knn.fit(X_train, y_train)

X_new = np.array([[5, 2.9, 1, 0.2]])
print("X_new.shape:", X_new.shape)
                          
prediction = knn.predict(X_new) # make prediction about new data

print(f"Prediction: {prediction}")

print(f"Predicted target name: {iris_dataset['target_names'][prediction]}")

#%%

y_pred = knn.predict(X_test)

print(f"Test set predictions:\n{y_pred}")

print(f"Test set score: {np.round(np.mean(y_pred == y_test), 3)}")

print(f"Test set score: {knn.score(X_test, y_test)}")

#%%

students = pd.read_csv('sample_data/sample_student_data.csv', skiprows=[1])

students = students.loc[: , ['Sex', 'Height', 'Weight']]

students['Sex'] = students['Sex'] == 'M'       # boolean array using comparison operator

students['Sex'] = students['Sex'].astype(int)  # integer array using type conversion

students.head()

X_train, X_test, y_train, y_test = train_test_split(students.loc[:,'Height' :], # features
                                                    students['Sex'],            # labels
                                                    random_state=0)

pd.plotting.scatter_matrix(X_train,                 # data frame
                           c=y_train,               # colour by y_train
                           figsize=(6, 6),
                           marker='o', 
                           hist_kwds={'bins': 20},  # plotting keyword arguments to be passed to hist function
                           s=60,                    # size of markers
                           alpha=.8,                # transparency of markers
                           cmap='viridis');         # colour map used for colour of each data plotted
                           
                           
knn = KNeighborsClassifier(n_neighbors=1)

knn.fit(X_train, y_train)

print(f"Test set score: {np.round(knn.score(X_test, y_test), 3)}")

#%%

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier    # 1. Import model
from sklearn.datasets import load_iris                # 2. Import data
iris_dataset = load_iris()



X_train, X_test, y_train, y_test = train_test_split(iris_dataset['data'],     # 3. Split the data
                                                    iris_dataset['target'], 
                                                    random_state=0)

knn = KNeighborsClassifier(n_neighbors=1) # 4. Instantaite the model including any model parameters

knn.fit(X_train, y_train)                 # 5. Fit the model to the training data

score = knn.score(X_test, y_test)         # 6. Evaluate the accuracy of the model on the test data
print(f"Test set score: {np.round(score, 3)}") 

knn.predict(X_test)                       # 7. Predict the targets of the test data

X_new = np.array([[5, 2.9, 1, 0.2]])      # 8. Predict the target of a new data point 
prediction = knn.predict(X_new)
print(f"Predicted target name: {iris_dataset['target_names'][prediction]}")































