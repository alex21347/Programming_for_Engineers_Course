# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

# Assignment 10 - Alex Hill - Programming for Engineers

#%%

#Review Exercise 1 : Data Cleaning

#pd.set_option('display.max_columns', 6)  
#pd.set_option('display.max_rows', 20)

#Import the data from sample_data/analysis.csv
import numpy as np
import pandas as pd

cols = ['season','riversize','fluidvelocity','A','B','C','D','E','F','G','a_','b_','c_','d_','e_','f_','g_','h_']
data = pd.read_csv('C:/Users/alex/Documents/KyotoU/Programming for Engineers/GitHub/ILAS_PyEng2019/sample_data/analysis.csv', sep = ',',names = cols, dtype = str)

#Replacing X strings with null values
data = data.replace("XXXXX",np.nan,regex=True)


#fixing strings with more than one decimal point - what a pain
for i in range(3,len(cols)-3):
    for j in range(0,len(data.values[:,0])):
        if type(data.iloc[j,i]) == str:
            if data.iloc[j,i].count('.') > 1:
                data.iloc[j,i] = data.iloc[j,i][:5]
            

#converting strings to floats
for i in range(3,len(cols)-3):
    data.iloc[:,i] = data.iloc[:,i].astype('float',errors='raise')


numna = []
#counting null values
for i in range(0,len(cols)):
    numna = np.append(numna,data.iloc[:,i].isnull().sum())
numna = pd.DataFrame(numna, columns = ['# Null Values'], index = cols)
print(numna)


#dropping rows with null values
data = data.dropna()


#Option 2 : filling null values with column average. (This ends up being a terrible idea)
# =============================================================================
# data = data.fillna(data.mean())
# =============================================================================

#%%

#Review Exercise 2 : Linear Regression

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

#removing first 3 columns
datal = data.iloc[:,3:]

#removing last columns
datalin = datal.iloc[:,:9]

#splitting data set into explanatory data and response variables
expldata = datalin.iloc[:,:7]
algaeA = datalin.iloc[:,7]
algaeB = datalin.iloc[:,8]

#split into test and train data - Algae A
X_trainA, X_testA, y_trainA, y_testA = train_test_split(expldata, algaeA, random_state=42) 

#split into test and train data - Algae B
X_trainB, X_testB, y_trainB, y_testB = train_test_split(expldata, algaeB, random_state=42) 

#train model
lrA = LinearRegression().fit(X_trainA, y_trainA)
lrB = LinearRegression().fit(X_trainB, y_trainB)

print(f"Training set score A: {lrA.score(X_trainA, y_trainA)}")
print(f"Test set score A: {lrA.score(X_testA, y_testA)}")
print()
print(f"Training set score B: {lrB.score(X_trainB, y_trainB)}")
print(f"Test set score B: {lrB.score(X_testB, y_testB)}")
print()
print("Neither species of algae are predicted very well, but algae B \nis predicted more accurately.")

#%%

#Review Exercise 3 : Converting for use in a machine learning model

#Feature Extraction
data['max'] = data.iloc[:,10:18].astype('float',errors='raise').idxmax(axis = 1)

#Integer data to represent string data
seasons = list(data['season'].unique())

for n, m in enumerate(seasons):
    data = data.replace({m: n})


data.riversize = data.riversize.replace("medium","medium_",regex=True)
riversizes = list(data['riversize'].unique())

for n, m in enumerate(riversizes):
    data = data.replace({m: n})


fluidvelocities = list(data['fluidvelocity'].unique())

for n, m in enumerate(fluidvelocities):
    data = data.replace({m: n})
    

maxes = list(data['max'].unique())

for n, m in enumerate(maxes):
    data = data.replace({m: n})

#Normalization
datanorm = data

#converting strings to floats
for i in range(0,19):
    datanorm.iloc[:,i] = datanorm.iloc[:,i].astype('float',errors='raise')
    data.iloc[:,i] = data.iloc[:,i].astype('float',errors='raise')

for i in range(0,len(data.iloc[0,:])-1):
    datanorm.iloc[:,i] = (datanorm.iloc[:,i] - np.mean(datanorm.iloc[:,i]))/np.std(datanorm.iloc[:,i])

#%%
   
# Review Exercise 4 : Neural Network
# solver options : {‘lbfgs’, ‘sgd’, ‘adam’}
    
from sklearn.neural_network import MLPClassifier 
from sklearn.model_selection import train_test_split


X = datanorm.iloc[:,:10]
y = datanorm.iloc[:,18]
scoretest = []
scoretrain = []

for i in range(30,40):
    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=i, test_size = 0.33)
    mlp = MLPClassifier(solver = 'lbfgs',random_state=i+1,hidden_layer_sizes=[60, 60])
    mlp.fit(X_train, y_train)
    scoretest = np.append(scoretest,mlp.score(X_test, y_test))
    scoretrain = np.append(scoretrain,mlp.score(X_train, y_train))
    
scoretestav = np.mean(scoretest)
scoretrainav = np.mean(scoretrain)
    
print(f"Accuracy on scaled training set: {scoretrainav}")
print(f"Accuracy on scaled test set: {scoretestav}")


X1 = data.iloc[:,:10]
y1 = data.iloc[:,18]
scoretest1 = []
scoretrain1 = []

for i in range(20,30):
    X1_train, X1_test, y1_train, y1_test = train_test_split(X1, y1, stratify=y1, random_state=i, test_size = 0.33)
    mlp1 = MLPClassifier(solver = 'lbfgs',random_state=i+1,hidden_layer_sizes=[60, 60])
    mlp1.fit(X1_train, y1_train)
    scoretest1 = np.append(scoretest1,mlp1.score(X1_test, y1_test))
    scoretrain1 = np.append(scoretrain1,mlp1.score(X1_train, y1_train))
    
scoretestav1 = np.mean(scoretest1)
scoretrainav1 = np.mean(scoretrain1)

print(f"Accuracy on unscaled training set: {scoretrainav1}")
print(f"Accuracy on unscaled test set: {scoretestav1}")

if scoretestav1 < scoretestav:
    print("Scaling the data has increased the accuracy of this model")
else:
    print("Scaling the data has decreased the accuracy of this model")
    
    
    
#if you increase the number of hidden layers, the unscaled model actually
#turns out better? lets test this with a graph vvvv

# =============================================================================
# from matplotlib import pyplot as plt
# X = datanorm.iloc[:,:18]
# y = datanorm.iloc[:,18]
# 
# X1 = data.iloc[:,:18]
# y1 = data.iloc[:,18]
# 
# dif = []
# for n in range (1,20):
#     scoretest = []
#     scoretest1 = []
#     for i in range(30,40):
#         X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=i, test_size = 0.33)
#         mlp = MLPClassifier(solver = 'lbfgs',random_state=i+1,hidden_layer_sizes=[5*n, 5*n])
#         mlp.fit(X_train, y_train)
#         scoretest = np.append(scoretest,mlp.score(X_test, y_test))
#     scoretestav = np.mean(scoretest)
#     
#     for i in range(20,30):
#         X1_train, X1_test, y1_train, y1_test = train_test_split(X1, y1, stratify=y1, random_state=i, test_size = 0.33)
#         mlp1 = MLPClassifier(solver = 'lbfgs',random_state=i+1,hidden_layer_sizes=[5*n, 5*n])
#         mlp1.fit(X1_train, y1_train)
#         scoretest1 = np.append(scoretest1,mlp1.score(X1_test, y1_test))
#     scoretestav1 = np.mean(scoretest1)
#     dif = np.append(dif,100*(scoretestav - scoretestav1)/scoretestav1)
# 
# dif = np.array(dif)
# 
# plt.figure()
# plt.plot(np.arange(5,100,5),dif)
# plt.ylabel("Percentage Accuracy changed by scaling")
# plt.xlabel("hidden layer size")
# plt.plot([0,100],[0,0],"--")
# plt.show()
# 
# =============================================================================


#testing the effect of hidden layers on overall accuracy vvvvv
    
# =============================================================================
# testscore = np.zeros([40,40])
# for i in range(2,41):
#     for j in range(2,41):
#         mlp = MLPClassifier(solver = 'lbfgs',random_state=42,hidden_layer_sizes=[i, j])
#         mlp.fit(X_train, y_train)
#         testscore[i-2,j-2] = mlp.score(X_test, y_test)
# 
# plt.imshow(testscore, cmap='viridis', interpolation='nearest')
# plt.show()
#
# print(testscore)
# =============================================================================
    
#%%

#Review Exercise 5 : Decision Tree
    
from sklearn.tree import DecisionTreeClassifier  
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt

X = datanorm.iloc[:,:10]
y = datanorm.iloc[:,18]

X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=42, test_size = 0.33)
tree = DecisionTreeClassifier(random_state=1, max_depth = 5)
tree.fit(X_train, y_train)

# number of features
n_features = X.shape[1]

# bar chart showing importance of each feature
plt.barh(np.arange(n_features), 
         tree.feature_importances_,
         align='center')

plt.yticks(np.arange(n_features), cols[:10])
plt.xlabel("Feature importance")
plt.ylabel("Feature")
plt.ylim(-1, n_features)
plt.show()
print()
print(f"Accuracy on scaled training set: {tree.score(X_train, y_train)}")
print(f"Accuracy on scaled test set: {tree.score(X_test, y_test)}")



scores = []
scoress = []

for i in range(1,30):
    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=42, test_size = 0.33)
    tree = DecisionTreeClassifier(random_state=1, max_depth = i)
    tree.fit(X_train, y_train)
    scores = np.append(scores, tree.score(X_train, y_train))
    scoress = np.append(scoress, tree.score(X_test, y_test))

plt.figure()
plt.plot(range(0,len(scores)),scores, label = "Training Accuracy")
plt.plot(range(0,len(scoress)),scoress, label = "Test Accuracy")
plt.xlabel("Max depth")
plt.ylabel("Accuracy")
plt.legend()
plt.show()

