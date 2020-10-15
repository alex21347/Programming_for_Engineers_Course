# -*- coding: utf-8 -*-
"""
Created on Sat Nov 30 21:25:27 2019

@author: alex
"""

# Assignment 6 - Programming for Engineers 


#%%

#Review Exercise 1 : Differentiation

import sympy as sp
from sympy import symbols
from sympy import diff

f,x,c,b=symbols('f,x,c,b')

f= sp.cos(x) + b*x + c

diff = diff(f,x,2)

print(diff)

#%%

#Review Exercise 2 : Integration

# 1) Symbolically
from sympy import integrate

f,x,c,b=symbols('f,x,c,b')

f= sp.cos(x) + b*x + c
print()
print(f'The integral of f in [0,1] is {integrate(f, (x,0,1))}')



# 2) Non-Symbolically
from scipy.integrate import quad

b=2
c=5

def ex2(x):
    return sp.cos(x) + b*x + c


ans, err = quad(ex2, 0, 1)

print()
print(f'The integral of f in [0,1] with b=2 and c=5 is {round(ans,5)} with error {err}')

#%%

#Review Exercise 3 : Estimating the integral of exerimental data

import matplotlib
from matplotlib import pyplot as plt


x = np.array([0, 10, 15, 30, 35, 50, 75, 80])
y = np.array([5, 6, 1, 1.5, 2, 5, 6, 1])

plt.figure()
plt.plot(x,y)
plt.xlabel("x")
plt.ylabel("y")
plt.show();

print()
print(f'The trapezoidal method gives the error under the graph to be {np.trapz(y, x)}')


#%%

#Review Exercise 4: Solving ordinary differntial equations

import numpy as np
import sympy as sp
from sympy import symbols, Symbol, Function, Eq, pprint,exp,dsolve
from sympy import integrate
from sympy import diff

t = Symbol('t')
v0,R,C1 =symbols('v0,R,C1')
v = Function('v')
D = Function('D')


veq = Eq(v,v0*exp(-t/R))
print()
pprint(veq)


equa = Eq(veq.rhs,D(t).diff())
equa2 = dsolve(equa)
pprint(equa2)


C1eq = equa2.subs([(t,0),(D(0),0)])
C1eq1 = Eq(solve(C1eq,R*v0)[0],R*v0)
print()
pprint(C1eq)
print()
pprint(C1eq1)


fullsol = equa2.subs([(C1,C1eq1.rhs)])
pprint(fullsol)

#%%

#Review Exercise 5 : Solving ordinary differntial equations

import numpy as np
import sympy as sp
from sympy import solve, symbols, Symbol, Function, Eq, pprint,exp,dsolve
from sympy import integrate
from sympy import diff
from scipy.integrate import odeint
import matplotlib.pyplot as plt

def dydx(y, x):
    c=x-y
    return c 

f0 = 0                    # the initial value
xspace = np.linspace(0,5,100) # the value(s) of t at which to evaluate x 


# odeint returns x at each value of t
y = odeint(dydx, # user-defined function
            f0,   # initial value
            xspace)   # value(s) of t at which to evaluate x


y = np.array(y).flatten() # flatten to 1D array

plt.figure()
plt.plot(y,xspace)
plt.xlabel('x')
plt.ylabel('y')
plt.title("Solution to ODE")
plt.show()







