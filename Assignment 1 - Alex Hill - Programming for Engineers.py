#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# Review Exercise 1: Temperature conversion


# In[41]:


Degrees_in_Fahrenheit=90
F=Degrees_in_Fahrenheit
C=5*(F-32)/9
print(F,"°F is",round(C,2),"°C")


# In[ ]:


# Review Exercise 2: `continue`


# In[62]:


for c in range(1,7):
    if c % 3 == 0:
        continue
    print(c)


# In[ ]:


# Review Exercise 3 : Currency Trading 両替


# In[106]:


JPY=100_000
marketrate=0.0091

if JPY<10_000:
    multiplier=0.9
    
elif JPY<100_000:
    multiplier=0.925
        
elif JPY<1_000_000:
    multiplier=0.95
                
elif JPY<10_000_000:
    multiplier=0.97
                        
else:
    multiplier=0.98
    
usdrecieved=JPY*marketrate*multiplier
print(f"JPY Paid: {JPY}¥")
print("----------")
print(f"USD Recieved: {round(usdrecieved,2)}$")
print("----------")
print("Effective rate:",round((usdrecieved/JPY),10))
print("----------")
print(f"Transaction Charge: {round((((JPY*marketrate)-round(usdrecieved,2))/marketrate))}¥ or {round((((JPY*marketrate)-round(usdrecieved,2))),2)}$")

