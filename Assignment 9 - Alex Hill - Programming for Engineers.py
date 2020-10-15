# -*- coding: utf-8 -*-
"""
Created on Wed Dec 18 14:27:17 2019

@author: alex
"""

#Assignment 9 - Programming for Engineers - Alex Hill

#%%

#Review Exercise : KNN Classifier with Real Data

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
import numpy as np

# 1 : Import the planets data from the seaborn package

import seaborn as sns
planets = sns.load_dataset('planets')

#%%
# 2 : Drop null (NaN) values:
#
#  -  show how many null values appear in each column
#  -  drop any columns where less than half the values are non-null
#  -  drop all remaining rows containing null values

print(planets.isnull().sum())
planets = planets.dropna(axis='columns', thresh = np.floor(len(planets.values[:,0])/2))
planets = planets.dropna(axis ='rows')
#%%

# 3 : Create a column with a unique integer value to represent each
#     unique string value in the method column of the DataFrame.

methods = list(planets['method'].unique())

for n, m in enumerate(methods):
    planets = planets.replace({m: n})

for (method, group) in planets.groupby('method'):    
    # This print formatting leaves 30 spaces between printed item 0 and printed item 1
    print("DataFrame name : {0:3}, shape={1}".format(method, group.shape))

#%%

# 4 : Split the data set into training and test data.

X_train, X_test,y_train, y_test = train_test_split(planets.loc[:,'number':],
                                                    planets['method'], 
                                                    random_state=0)
    
    
    


#%%

# 5 : Create a scatter plot to check if the different
#     method classes/targets can be separated using the features.

pd.plotting.scatter_matrix(X_train,                 # data frame
                           c=y_train,               # colour by y_train
                           figsize=(6, 6),
                           marker='o', 
                           hist_kwds={'bins': 20},  # plotting keyword arguments to be passed to hist function
                           s=60,                    # size of markers
                           alpha=.8,                # transparency of markers
                           cmap='viridis');         # colour map used for colour of each data plotted
                           

#%%

# 6 : Import the KNN model, instantiate and fit the model to the training data.

from matplotlib import pyplot as plt

n=20

scores = []                         
for i in range(1,n):
    knn = KNeighborsClassifier(n_neighbors=i)
    knn.fit(X_train, y_train)
    score = knn.score(X_test, y_test)
    scores = np.append(scores,score)

best = scores.max()
plt.figure()
plt.scatter(range(1,n),scores)    
plt.show()
    

bests = []
for i in range(0,n-1):
    if scores[i] == best:
        bests = np.append(bests,i)

bestn = bests.min()
print(f'Best Accuracy : {round(best,5)}\nBest n : {bestn}')
                          
#%%

# 7 : What percentage of the test data does the model predict correctly?

print(round(best,3))

#%%

# 8 : Look at step 5 again.
#     Do some features seperate the classes better than others?
#     What happens if you remove the features that do not seperate the classes well?
#     How does this effect the accuracy of the model prediction?

#It seems like number isnt contributing much to the classifier,
#so lets remove it and see what happens!

X_train, X_test,y_train, y_test = train_test_split(planets[['orbital_period','year','distance']],
                                                    planets['method'], 
                                                    random_state=0)

n=20

scores = []                         
for i in range(1,n):
    knn = KNeighborsClassifier(n_neighbors=i)
    knn.fit(X_train, y_train)
    score = knn.score(X_test, y_test)
    scores = np.append(scores,score)

best = scores.max()
plt.figure()
plt.scatter(range(1,n),scores)    
plt.show()
    

bests = []
for i in range(0,n-1):
    if scores[i] == best:
        bests = np.append(bests,i)

bestn = bests.min()
print(f'Best Accuracy : {round(best,5)}\nBest n : {bestn}')
                           
# from messing around with the different features it was clear that the 
# classifier would work either the same or worse with less features. With number
# removed as a feature, the classifier had the exact same accuracy.
                           
                           
                           


