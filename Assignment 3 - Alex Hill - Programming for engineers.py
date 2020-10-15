#!/usr/bin/env python
# coding: utf-8

# In[257]:


#get_ipython().run_line_magic('matplotlib', 'inline')
import numpy as np
from matplotlib import pyplot as plt
from scipy import stats
from scipy.optimize import curve_fit


# ### Review Exercise 1: Curve Fitting

# In[256]:


data= np.array(np.loadtxt("sample_data/signal_data.csv", delimiter=","))

def func1(x, a, b): # function name and inputs
    y = a * np.sin(b + x)   # function
    return y 

opt,cov = curve_fit(func1,data[0],data[1])

x=np.linspace(min(data[0]),max(data[0]),100)
y=func1(x,opt[0],opt[1])


e=func1(data[0],opt[0],opt[1])-data[1]

RMSE = []
  # rmse
rmse = np.sqrt(np.sum(e**2)/ len(data[1]))

    # RMSE
RMSE.append(round(rmse, 3))
    
f = plt.figure()
plt.scatter(data[0],data[1],color="k",alpha=0.5,s=15,label="Data");
plt.plot(x,y,color=(0.9,0.4,0),linewidth=2,label="Approximation")
plt.title("RMSE = %.3f" %RMSE[0])
plt.legend()
plt.show()

print(f'The equation of the fitted line is y = {round(opt[0],2)}*sin(x + {round(opt[1],2)})')

f.savefig("Sample_data/Exercise1.pdf", bbox_inches='tight')


# ### Review Exercise 2: Interpolation

# In[66]:


y=np.array([19.1,19.1,19.0,18.8,18.7,18.3,18.2,17.6,11.7,9.9,9.1])
x = np.linspace(0,-10,11)

a, b        = np.polyfit(x,y, deg=1)
x0          = min(x)  
x1          = max(x)  
y0          = a*x0 + b   
y1          = a*x1 + b

linearestimate=a*-7.5+b

c, d, e      = np.polyfit(x,y, deg=2)  
w=np.linspace(x0,x1,100)
z=c*w**2 + d*w + e

quadraticestimate=c*(-7.5)**2 + d*(-7.5) + e

plt.figure()
plt.title("Depth vs Temperature Regression Analysis")
plt.scatter(x,y)
plt.plot([x0,x1],[y0,y1],color="r",label="Linear Regression")
plt.plot(w,z,label="Quadratic Regression")
plt.scatter([-7.5],[linearestimate], label="Linear Estimate ( %.3f)"%linearestimate, color="r",marker="^",s=50)
plt.scatter([-7.5],[quadraticestimate], label="Quadratic Estimate ( %.3f)"%quadraticestimate,color="b",marker="^",s=50)
plt.legend()
plt.xlabel("Depth (m)")
plt.ylabel("Temperature (°C)")
plt.show()


# The quadratic curve fits the data better.

# ### Review Exercise 3: Importing .csv Data and Working with Arrays

# __Part A : Importing data__

# In[180]:


np.set_printoptions(suppress=True)
data = np.loadtxt("sample_data/douglas_data.csv", delimiter = ",",dtype='float', skiprows=2,usecols=range(1,9))
data=np.array(data)


# __Part B : Manipulating Data__

# In[179]:


b=np.array( [False]*data[:,0].size)                       #creating Boolean array

for i in range(0,10):                                     #setting first 10 rows to true
    b[i]=True
    
datanew=data[b]                                           #using boolean array to reduce the data to 10 rows

for i in range(0,10):                                     #converting units in the 8th column
    datanew[i,7]=datanew[i,7]*10**-6
    
newcol=np.ones((10,1))                                    # constructing new column
for i in range(0,10):
    newcol[i]=newcol[i]*0.01*datanew[i,4]*datanew[i,5]    #defining new column

datanew=np.hstack((datanew,newcol))                       #adding column to array


# __Part C : Displaying Data__

# In[178]:


print(f'The mass of the first beam is {datanew[0,8]}')
print()
for i in range(0,9):
    if i % 2 == 0:
        if i == 0:
            print(f'The data in the 5th row and {i+1}st column is {datanew[4,i]}')
        elif i == 1:
            print(f'The data in the 5th row and {i+1}nd column is {datanew[4,i]}')
        elif i == 2:
            print(f'The data in the 5th row and {i+1}rd column is {datanew[4,i]}')
        else:
            print(f'The data in the 5th row and {i+1}th column is {datanew[4,i]}')


# In[ ]:




