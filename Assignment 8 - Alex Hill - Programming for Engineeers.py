# -*- coding: utf-8 -*-
"""
Created on Thu Dec 12 17:57:41 2019

@author: alex
"""

# Assignment 8 - Alex Hill - Programming for Engineeers

import pandas as pd
from IPython.display import display

#%%

#Review Excercise: Restaurant Data

url = 'https://raw.githubusercontent.com/justmarkham/DAT8/master/data/chipotle.tsv'
data = pd.read_csv(url, sep = '\t')

# (a) What was the most popular item (item ordered the most times)?
#     Hint : Use groupby
print()
print("(a)")

mostpop = data.groupby('item_name').quantity.sum().sort_values( ascending = False).index[0]
print()
print(f' The most popular dish is \'{mostpop}\'.')

# (b) What was the average sum of money spent on a single order?
#     Hint : Convert prices to numerical data

print()
print("(b)")

data['item_price'] = data.item_price.replace('\$', '', regex=True).astype(float)
avprice = data.groupby("order_id").item_price.sum().mean()
print()
print(f' The average price per order is ${round(avprice,2)}.')

#(c) For each bowl, burrito and tacos item (Chicken Bowl etc) there
#    is a choice description. Produce a table showing the number of times
#    each choice description is selected for each bowl, burrito and tacos dish.

print()
print("(c)")

choice_index = data.item_name.str.contains('Bowl | Burrito | Tacos')
choice_items = data[choice_index]

choices = choice_items.groupby(['item_name', 'choice_description'])['quantity'].sum().unstack()
choices.fillna(0, inplace= True)
display(choices.head())

#%%

# Review Excercise: Time Series Data

# (a) Can you see a yearly repeating pattern in the data?
#     Plot the average monthly instances of each search keyword for
#     all years in the data set.

data2 = pd.read_csv('C:/Users/alex/Documents/KyotoU/Programming for Engineers/Assignments/sample_data/multiTimeline.csv',index_col="Month", parse_dates=True,skiprows=1)
data2.columns = ['diet', 'gym', 'finance']
by_month = data2.groupby(data2.index.month).mean()
by_month.plot(title = "Thoughout the year")

print()
print()
print("There is a downward trend throughout the year for searches of \n\'gym\',\'diet\' and \'finance\'.")


# (b) Are there any noticable long-term trends over the period 2004-2017?
#     It's difficult to see because of the annual fluctuations.
#     Smooth the data to reveal any longer-term trends in the search keyword data.

monthly = data2.resample('Y').mean().plot(title = "Over the Years") 
