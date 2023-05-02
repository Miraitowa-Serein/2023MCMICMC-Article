#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
data=pd.read_excel("Data_Wordle.xlsx",sheet_name="Sheet1")
data


# In[2]:


data['Date']=pd.to_datetime(data['Date'])
data.sort_values(by='Date',inplace=True)
data=data.reset_index(drop=True)
data


# In[3]:


import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['Times New Roman']
plt.rcParams['axes.unicode_minus'] = False
ax=data.plot(x='Date', y=['Number of reported results', 'Number in hard mode'])
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.xlabel('Date',fontsize=14)
plt.ylabel('Person Quantity',fontsize=14)
plt.legend()
plt.tight_layout()
plt.savefig("figures\\报告结果每日变化.pdf")


# In[4]:


data['WordLength'] = data['Word'].apply(len)
data['SumRate']=data.loc[:,['1 try','2 tries','3 tries','4 tries','5 tries','6 tries','7 or more tries (X)']].sum(axis=1)
data['HardRate']=data['Number in hard mode']/data['Number of reported results']
data


# In[5]:


data[data['WordLength']!=5]


# In[6]:


ax=data.plot.scatter(x='Date', y='HardRate')
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.xlabel('Date',fontsize=14)
plt.ylabel('Hard mode frequency',fontsize=14)
plt.tight_layout()
plt.savefig('figures\\每日选择困难模式人数频率变化.pdf')


# In[7]:


data['HardRateDiff']=data['HardRate'].diff()


# In[8]:


data.plot.scatter(x='Date', y='HardRateDiff',color='g')
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.xlabel('Date',fontsize=14)
plt.ylabel('Hard mode frequency gradient',fontsize=14)
plt.tight_layout()
plt.savefig('figures\\每日选择困难模式人数频率变化率.pdf')


# In[9]:


data=data.fillna(0)
data


# In[10]:


data[abs(data['HardRateDiff'])>=0.02]

