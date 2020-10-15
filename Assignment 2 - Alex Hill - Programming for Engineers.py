#!/usr/bin/env python
# coding: utf-8

# ## Review Exercise 1 : Combining Imported Functions

# In[25]:


import numpy as np

#This function determines the angle between two angles given by;
a = np.array([9, 2, 7])
b = np.array([4, 8, 10])

asize=np.linalg.norm(a)
bsize=np.linalg.norm(b)
adotb=0
for d in range(0,3):
    adotb += a[d]*b[d]

#print(asize)
#print(bsize)
#print(adotb)

theta=np.arccos(adotb/(asize*bsize))
print("The angle between a and b is",theta, "radians")


# ## Review Exercise 2 :  Classifer

# In[27]:


a = np.array([-1, 2, 6])
b = np.array([4, 3, 3])

adotb=0
for d in range(0,3):
    adotb += a[d]*b[d]
    
if adotb>0:
    print("The angle between a and b is acute")
elif adotb<0:
    print("The angle between a and b is obtuse")
else:
    print("The angle between a and b is a right-angle")
    


# ## Review Exercise 3: Numpy Package Functions. 

# In[76]:


#A

a = np.array([0.1, 0, 10])
b = np.array([0.0, 0.0, 0.0])

for d in range(0,3):
    b[d]=np.exp(a[d])

print(b)


# In[78]:


#B

theta=47
thetar=np.radians(theta)

print(theta,"degrees is equivalent to",thetar,"radians")


# In[79]:


#C

a = np.array([4, 16, 81])
b = np.array([0.0, 0.0, 0.0])

for d in range(0,3):
    b[d]=np.sqrt(a[d])

print(b)


# ## Review Exercise 4:  Using a single list with a `for` loop.

# In[92]:


months = ["January",
         "February",
         "March",
         "April",
         "May",
         "June",
         "July",
         "August",
         "September",
         "October",
         "November",
         "December"]

for n in range(0,12):
    print(months[n][0])

