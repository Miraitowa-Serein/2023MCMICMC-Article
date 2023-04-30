#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# In[2]:


data1=pd.read_excel('Data\\Two\\Landing\\1Landing.xlsx',sheet_name='Sheet1')
data2=pd.read_excel('Data\\Two\\Landing\\2Landing.xlsx',sheet_name='Sheet1')
data3=pd.read_excel('Data\\Two\\Landing\\3Landing.xlsx',sheet_name='Sheet1')
data4=pd.read_excel('Data\\Two\\Landing\\4Landing.xlsx',sheet_name='Sheet1')
data5=pd.read_excel('Data\\Two\\Landing\\5Landing.xlsx',sheet_name='Sheet1')
data6=pd.read_excel('Data\\Two\\Landing\\6Landing.xlsx',sheet_name='Sheet1')
data7=pd.read_excel('Data\\Two\\Landing\\7Landing.xlsx',sheet_name='Sheet1')
data8=pd.read_excel('Data\\Two\\Landing\\8Landing.xlsx',sheet_name='Sheet1')


# In[3]:


def draw(data,i):
    data.fillna(method='ffill',axis = 0,inplace=True)
    x=data['时刻']
    y1=data['杆量']
    y2=data['盘量']
    y3=data['着陆G值']
    y4=data['姿态（俯仰角）']
    y5=data['坡度（左负右正）']
    plt.rcParams['font.sans-serif']=['SimHei']
    plt.rcParams['axes.unicode_minus']=False
    plt.figure(figsize=(12, 6))
    plt.plot(x, y1, ms=5, label='杆量', color="blue")
    plt.plot(x, y2, ms=5, label='盘量', color="red")
    plt.plot(x, y3, ms=5, label='着陆阶段G值', color="green")
    plt.plot(x, y4, ms=5, label='姿态（俯仰角）', color="black")
    plt.plot(x, y5, ms=5, label='坡度（左负右正）', color="orange")
    plt.xlabel('时刻',size=13)
    plt.ylabel('量值',size=13)
    plt.xticks(font='Times New Roman',size=12)
    plt.yticks(font='Times New Roman',size=12)
    plt.legend()
    plt.savefig('Figures\\1-{} Landing.pdf'.format(i))


# In[4]:


draw(data1,1)


# In[5]:


draw(data2,2)


# In[6]:


draw(data3,3)


# In[7]:


draw(data4,4)


# In[8]:


draw(data5,5)


# In[9]:


draw(data6,6)


# In[10]:


draw(data7,7)


# In[11]:


draw(data8,8)

