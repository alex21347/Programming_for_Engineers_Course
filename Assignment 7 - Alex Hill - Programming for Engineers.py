# -*- coding: utf-8 -*-
"""
Created on Thu Dec  5 16:02:25 2019

@author: alex
"""

#Assignment 7 - Practice_Problems

#%%

# 1. Plotting Data

import numpy as np
from matplotlib import pyplot as plt


# =============================================================================
# data1 = np.loadtxt('C:/Users/alex/Documents/KyotoU/Programming for Engineers/Assignments/sample_data/temperature_data.csv', dtype='float',skiprows=1, delimiter = ",",usecols = range(1,8))
# =============================================================================
data1 = np.loadtxt('sample_data/temperature_data.csv', dtype='float',skiprows=1, delimiter = ",",usecols = range(1,8))
data1 = np.array(data1)

months = np.arange(1,12,2)
months = np.append(months,12)
allmon = np.arange(1,13,1)
missmon = []

for i in range(0,len(allmon)):
    k=0
    for j in range(0,len(months)):
        if allmon[i] == months[j]:
            k = k+1
    if k == 0:
        missmon = np.append(missmon,allmon[i])

   
a1,b1,c1,d1 = np.polyfit(months,data1[0], deg=3)  
ylon = a1*missmon**3 + b1*missmon**2 + c1*missmon + d1
   
a2,b2,c2,d2= np.polyfit(months,data1[1], deg=3)  
yphi = a2*missmon**3 + b2*missmon**2 + c2*missmon + d2

a3,b3,c3,d3 = np.polyfit(months,data1[2], deg=3)  
yhon = a3*missmon**3 + b3*missmon**2 + c3*missmon + d3


plt.figure(figsize=(8,5))
plt.plot(months,data1[0],label = "London")
plt.plot(months,data1[1],label = "Philadelphia")
plt.plot(months,data1[2],label = "Hong Kong")
plt.scatter(missmon,ylon,label = "London missing months estimate")
plt.scatter(missmon,yphi,label = "Philidelphia missing months estimate")
plt.scatter(missmon,yhon,label = "Hong Kong missing months estimate")
plt.xlabel("Month")
plt.ylabel("Temperature °C")
plt.legend(loc = (0.3,0.05))
plt.title("Exercise 1 : Plotting Data")
plt.show()


#%%

# 2. Curve Fitting

data2 = np.loadtxt('C:/Users/alex/Documents/KyotoU/Programming for Engineers/Assignments/sample_data/air_temperature.dat')
time = np.arange(0,24,2)

#lets go for cubic eh

a,b,c,d = np.polyfit(time,data2, deg=3)  
x = np.linspace(0,22,40)
y = a*x**3+b*x**2+c*x+d


ytest = a*time**3+b*time**2+c*time+d

e = ytest - data2
RMSE = []
rmse = np.sqrt(np.sum(e**2)/ len(time))
RMSE.append(round(rmse, 3))


plt.figure()
plt.plot(time,data2, color ="k")
plt.scatter(x,y,alpha = 0.5, color = "blue")
plt.xlabel("Time (Hours)")
plt.ylabel("Temperature (°C)")
plt.legend()
plt.title("Exercise 2 : Curve Fitting (RMSE = %.3f)" %RMSE[0])
plt.show()

#%%

# 3. Functions and Libraries

import sys
# =============================================================================
# sys.path.append('C:/Users/alex/Documents/mypython/Kyoto/my_functions')
# =============================================================================

import myfunc as my

data3 = np.loadtxt('C:/Users/alex/Documents/KyotoU/Programming for Engineers/Assignments/sample_data/air_temperature.dat')
x = np.linspace(1,15,15)


plt.figure(figsize=(8,5))

for j in range(0,len(data3)):
    f =[]
    for i in range(0,15):
        f = np.append(f,(my.myfunc(x[i],data3[j])))
    plt.plot(x,f,label = "k = %.1f" %data3[j]) 
plt.title("Exercise 3 : Functions and Libraries")
plt.xlabel("x")
plt.ylabel("f")
plt.legend(loc = "best")
plt.show()

#%%

# 4. Systems of Equations

from sympy import Matrix, symbols, solve,cos,pi
import numpy as np
from matplotlib import pyplot as plt

a,b,c,d = symbols('a, b, c, d')

t1 = 0
t2 = 0.25
t3 = 0.5
t4 = 0.75

A = Matrix([[cos(pi*t1),cos(2*pi*t1),cos(3*pi*t1),cos(4*pi*t1)],
              [cos(pi*t2),cos(2*pi*t2),cos(3*pi*t2),cos(4*pi*t2)],
              [cos(pi*t3),cos(2*pi*t3),cos(3*pi*t3),cos(4*pi*t3)],
              [cos(pi*t4),cos(2*pi*t4),cos(3*pi*t4),cos(4*pi*t4)]])

B = Matrix([3,1,-3,1])

X = Matrix([[a],
            [b],
            [c],
            [d]])

sols = solve(A * X - B, [a, b, c, d])
a = float(sols[a])
b = float(sols[b])
c = float(sols[c])
d = float(sols[d])

x = np.linspace(0,1,100)
y = np.cos(np.pi*x) + np.cos(2*np.pi*x) + np.cos(3*np.pi*x) + np.cos(4*np.pi*x)
x1 = np.array([t1,t2,t3,t4])
y1 = np.cos(np.pi*x1) + np.cos(2*np.pi*x1) + np.cos(3*np.pi*x1) + np.cos(4*np.pi*x1)

plt.figure()
plt.plot(x,y, label ="Solved function")
plt.scatter(x1,y1,s=20,label = "Test data")
plt.title("Exercise 4 : Systems of Equations")
plt.legend()
plt.show()

#%%

# 5. Numerical Integration

import numpy as np
import sympy as sp
from sympy import integrate
import matplotlib.pyplot as plt


# a)

a, b, c, x = symbols('a, b, c, x')
f = sp.exp(-x)

sol = integrate(f, (x,1,5))
print(f'The solution to the integral is {sol}')

# b)

# We have a few methods to integrate numerically, I shall opt for the trapezium 
# method

def func5b(x):
    y = np.exp(-x)/x
    return y

x = np.linspace(1,5,10000)
y = func5b(x)
trap5b = round(np.trapz(y,x),6)

print(f'The trapezium method gives the area to be {trap5b} which is the desired result.')
    
#%%

# 6. Predator Prey

from scipy.integrate import odeint

def dU_dx(U, x):
    """
    U is a vector such that U[0] = x and U[1] = y.
    Returns [x', y']
    """  
    return [U[0]-U[0]*U[1], 
            U[1]*U[0]-U[1]]


U0 = [1,2]                   # the initial value(had no clue what to set for this)
ts = np.linspace(0, 12, 200)  # the value(s) of x at which to evaluate t


# odeint returns x,y at each value of t
Us = odeint(dU_dx, # user-defined function
            U0,    # initial value
            ts)    # value(s) of t at which to evaluate x


ys = Us[:,0]  # y is the first column of the solution U = [y, z]T


plt.xlabel("x")
plt.ylabel("y")
plt.title("Exercise 6 : Predator Prey")
plt.plot(ts,ys);




