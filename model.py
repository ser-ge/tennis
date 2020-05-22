#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix,log_loss, r2_score
import datetime
import dateutil
from dateutil.relativedelta import relativedelta
from datetime import date
# get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


data = pd.read_csv('data.csv', index_col='index')
data.Date = pd.to_datetime(data.Date)
data.set_index('Date', inplace=True)
data.sort_index(inplace=True)

len(data)


# In[3]:


x_values = ['PtsDelta', 'IntervalDelta' , 'P1Age', 'P2Age']
y_values = ['P1Result']
odds = ['B365P1', 'B365P2']


# In[4]:


data['total_over']= 1/data[odds[0]]+1/data[odds[1]] -1
data['P1_implied'] = 1/data[odds[0]] - data['total_over']/2
data['P2_implied'] = 1/data[odds[1]]- data['total_over']/2
data['P1_implied_log'] =data['P1_implied'].apply(np.log)
data['P2_implied_log'] =data['P2_implied'].apply(np.log) 

implied_probs = ['P1_implied', 'P2_implied', 'P1_implied_log','P2_implied_log']


# In[5]:


model_data = data[x_values+y_values+odds+implied_probs].copy()
model_data.dropna(inplace=True)
print(len(model_data))


# In[6]:


train_to = model_data.index[int(len(model_data)/2)]
train_model = model_data[:train_to]
print(len(train_model))


# In[ ]:





# In[7]:


X_train, X_test, y_train, y_test = train_test_split(train_model[x_values],train_model['P1Result'], test_size=0.1)


# In[8]:


model = LogisticRegression(solver='liblinear', fit_intercept=True)
model.fit(X_train,y_train)
predictions = model.predict(X_test)


# In[9]:


print(classification_report(y_test, predictions))


# In[10]:


simple_model_probs = model.predict_proba(X_test)[:,1]
p1_implied_test = model_data.P1_implied[train_to:]

sns.distplot(simple_model_probs, rug=True)
# model.predict_proba(model_data[train_to:][x_values])
# sns.distplot(p1_implied_test)


# In[11]:


model_data['LogProbsP2'], model_data['LogProbsP1']  = model.predict_log_proba(model_data[x_values])[:,0] , model.predict_log_proba(model_data[x_values])[:,1] 
train_model_2 = model_data[train_to:]
print(len(train_model_2))


# In[12]:


x_values_2 = ['LogProbsP1',  'P1_implied_log']


# In[13]:


X_train2, X_test2, y_train2, y_test2 = train_test_split(train_model_2[['LogProbsP1',  'P1_implied_log']],train_model_2['P1Result'], test_size=0.1)
model_wp = LogisticRegression(solver='liblinear', fit_intercept=True)
model_wp.fit(X_train2,y_train2)
predictions2 = model_wp.predict(X_test2)
print(classification_report(y_test2, predictions2))

sns.distplot(model_wp.predict_proba(X_test2)[:,1], bins=100)
sns.distplot(model_wp.predict_proba(X_test2)[:,1], bins=100)

r2_score(y_test2, predictions2)


# In[14]:


print(1/np.exp(-model.intercept_))


# In[ ]:





# In[15]:


p1_probs = model_wp.predict_proba(X_test2)[...,1]
p2_probs = model_wp.predict_proba(X_test2)[...,0]
p1_implied_probs=np.exp(X_test2.values[...,1])
# sns.distplot(p1_probs, label='model outs')
sns.distplot(p1_implied_probs, label='bookies')



# In[16]:


class BetBot:
    
    def _init_(self,capital,predictor):
        self.capital = capital 
        self.predictor = predictor
        
    def bet(self, *info, odds):
        pass
        
        
        
        
        


# In[17]:


model_wp.predict_proba(X_test2)[...,0]


# In[18]:


days = 250
hours = 7
pay = 30000
per_hour = days*hours/pay


# In[19]:


print(1/per_hour)


# In[ ]:




