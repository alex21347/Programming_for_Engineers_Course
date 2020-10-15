# -*- coding: utf-8 -*-
"""
Created on Sun Nov 17 16:25:14 2019

@author: alex
"""

#Assignment 4 - Alex Hill

import numpy as np
from matplotlib import pyplot as plt


#%%

#Review Exercise 1 : Simple function

def is_even(N):
    if N % 2 == 0:
        print(f'{N} is even')
        
    else:
        print(f'{N} is odd')
    
is_even(2)
is_even(3)


#%%

# Review Exercise 2: Using Data Structures as Function Arguments 

def counter(x):
    k=0
    for i in range(0,len(x)):
        if x[i] == "fizz":
            k=k+1
    print(f'There are {k} fizzes in {x}')       

my_list = ["fizz", "buzz", "buzz", "fizz", "fizz", "fizz"]
counter(my_list)



#%%

# Review Exercise 3: Using Functions as Function Arguments, Default Arguments. 

def is_even(N,func,func1):
    if func1(func(N)) % 2 == 0:
        print(f'{N} when rooted and floored is even')
        
    else:
        print(f'{N} when rooted and floored is odd')

for i in range (1,26):
    is_even(i,np.sqrt,np.floor)
    
#%%

# Review Exercise 4: Writing Functions


def pressure(h,rho=1000.0,g=9.81):
    return rho*g*h

x= np.linspace(0,10,1000)
y=pressure(x)

plt.figure()
plt.title("Pressure vs Height")
plt.plot(x,y)
plt.show()




