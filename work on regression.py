#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd 
import numpy as  np
import matplotlib.pyplot as plt 
from sklearn import linear_model


# In[3]:


#**********Linear Regression**********#


# In[42]:


df = pd.read_csv("C:/Users/subba/OneDrive/Desktop/movie/Linear Regression.csv  -  version 1.0. 26-04-2025 22.42.csv")
df


# In[ ]:





# In[43]:


#x = df.area
#y = df.price
#plt.scatter(x,y,marker="+",color="Red")
#                or 
get_ipython().run_line_magic('matplotlib', 'inline')
plt.scatter(df.area,df.price,marker="+",color="Red")


# In[44]:


reg = linear_model.LinearRegression()
reg.fit(df[['area']],df.price)


# In[45]:


reg.predict([[3500]])


# In[46]:


reg.coef_ #the value of "m" in formula y = mx+b


# In[47]:


reg.intercept_ #the value of "b" in formula y = mx+b


# In[48]:


135.78767123*3300+180616.43835616432
#  m*x+b


# In[49]:


plt.scatter(df.area,df.price,marker="+",color="Red")
plt.plot(df.area,reg.predict(df[["area"]]),color = 'black')


# In[ ]:





# In[50]:


#*************Polynomial Regression********************


# In[51]:


import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
from sklearn import linear_model


# In[52]:


df = pd.read_csv("C:/Users/subba/OneDrive/Desktop/movie/Polynomial_Regression.csv")
df


# In[53]:


reg = linear_model.LinearRegression()


# In[54]:


reg


# In[66]:


plt.scatter(df.area,df.price,marker="+",color = 'blue')
#plt.scatter(df.bedrooms,df.price,marker="+",color = 'black')
#plt.scatter(df.age,df.price,marker="+",color = 'red')


# In[56]:


reg.fit(df[['area','bedrooms','age']],df.price)


# In[57]:


reg.coef_


# In[58]:


reg.intercept_


# In[59]:


reg.predict([[3000,3,40]])

