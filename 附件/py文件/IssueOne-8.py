#!/usr/bin/env python
# coding: utf-8

# # 附件一预处理（不包括异常值剔除及标准化）

# In[1]:


import pandas as pd
import numpy as np


# In[2]:


data=pd.read_excel("Data\\First\\New\\201404101159.xlsx",sheet_name='201404101159')
data


# In[3]:


data.info()


# In[4]:


import missingno
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif']=['SimHei']
plt.rcParams['axes.unicode_minus']=False
missingno.bar(data, color=(190/255,190/255,190/255))
plt.tight_layout()
plt.savefig('Figures\\附件1航班8缺失值.pdf')


# In[5]:


data=data.fillna(0)
data


# In[6]:


data.drop(labels=['月','日','起飞机场','落地机场','飞机重量'],axis=1,inplace=True)
data


# # QAR异常判断，剔除

# In[7]:


dup_row = data.duplicated(subset=['具体时间'], keep=False)
data.insert(0, 'is_dup', dup_row)
data[data['is_dup'] == True]


# In[8]:


data=data.drop_duplicates(subset=['具体时间'],keep='first')
data


# In[9]:


dup_row = data.duplicated(subset=['具体时间'], keep=False)
data.insert(0, 'is_dup_N', dup_row)
data[data['is_dup_N'] == True]


# In[10]:


def function(a, b):
    if a == b:
        return 1
    else:
        return 0


data['bool'] = data.apply(lambda x : function(x['俯仰角率'],x['俯仰角率.1']),axis = 1)
data


# In[11]:


data[data['bool']==0]


# In[12]:


data=data.drop(labels=['is_dup','is_dup_N','bool','具体时间','俯仰角率.1'],axis=1)
data

