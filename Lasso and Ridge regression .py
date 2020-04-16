#!/usr/bin/env python
# coding: utf-8

# In[1]:


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import matplotlib
matplotlib.rcParams.update({'font.size':12})


# In[3]:


from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge


# In[4]:


boston=load_boston()
boston_df=pd.DataFrame(boston.data,columns=boston.feature_names)


# In[6]:


print(boston_df.info())


# In[7]:


boston_df['Price']=boston.target


# In[8]:


print(boston_df.head(3))


# In[9]:


newX=boston_df.drop('Price',axis=1)


# In[10]:


print(newX[0:3])


# In[11]:


newY=boston_df['Price']


# In[17]:


x_train,x_test,y_train,y_test=train_test_split(newX,newY,test_size=0.3,random_state=3)


# In[18]:


len(x_test)


# In[19]:


len(y_test)


# In[20]:


lr=LinearRegression()


# In[21]:


lr.fit(x_train,y_train)


# In[22]:


rr=Ridge(alpha=0.01) #higher the alpha value, more restriction on the coefficients; low alpha > more generalization, coefficients are barely


# In[23]:


rr.fit(x_train,y_train)
rr100 = Ridge(alpha=100)
rr100.fit(x_train,y_train)


# In[25]:


train_score=lr.score(x_train, y_train)
test_score=lr.score(x_test, y_test)
Ridge_train_score = rr.score(x_train,y_train)
Ridge_test_score = rr.score(x_test, y_test)


# In[26]:


Ridge_train_score100 = rr100.score(x_train,y_train)
Ridge_test_score100 = rr100.score(x_test, y_test)


# In[30]:


print("linear regression train score:", train_score)
print("linear regression test score:", test_score)
print("ridge regression train score low alpha:", Ridge_train_score)
print("ridge regression test score low alpha:", Ridge_test_score)
print("ridge regression train score high alpha:", Ridge_train_score100)
print("ridge regression test score high alpha:", Ridge_test_score100 )


# In[31]:


plt.plot(rr.coef_,alpha=0.7,linestyle='none',marker='*',markersize=5,color='red',label=r'Ridge; $\alpha = 0.01$',zorder=7) # zorder for ordering the markers


# In[32]:


plt.plot(rr100.coef_,alpha=0.5,linestyle='none',marker='d',markersize=6,color='blue',label=r'Ridge; $\alpha = 100$') # alpha here is for transparency


# In[33]:


plt.plot(lr.coef_,alpha=0.4,linestyle='none',marker='o',markersize=7,color='green',label='Linear Regression')


# In[35]:


plt.plot(rr.coef_,alpha=0.7,linestyle='none',marker='*',markersize=5,color='red',label=r'Ridge; $\alpha = 0.01$',zorder=7) # zorder for ordering the markers
plt.plot(rr100.coef_,alpha=0.5,linestyle='none',marker='d',markersize=6,color='blue',label=r'Ridge; $\alpha = 100$') # alpha here is for transparency
plt.plot(lr.coef_,alpha=0.4,linestyle='none',marker='o',markersize=7,color='green',label='Linear Regression')
plt.xlabel('Coefficient Index',fontsize=16)
plt.ylabel('Coefficient Magnitude',fontsize=16)
plt.legend(fontsize=13,loc=4)
plt.show()


# Let’s understand the figure above.
# In X axis we plot the coefficient index and,
# for Boston data there are 13 features (for Python 0th index 
#                                        refers to 1st feature). 
# For low value of α (0.01), when the coefficients are less restricted,
# the magnitudes of the coefficients are almost same as of linear regression. For higher value of α (100), we see that for coefficient indices 3,4,5 the magnitudes are considerably less compared to linear regression case. This is an example of shrinking coefficient magnitude using Ridge regression.
