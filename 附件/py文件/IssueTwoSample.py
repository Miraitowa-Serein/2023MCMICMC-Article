#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np


# In[2]:


def Sample(data):
    data = data[['空地电门0.2秒','空地电门0.4秒','空地电门0.6秒','空地电门0.8秒','空地电门1秒',
                 '着陆G值0.1秒','着陆G值0.2秒','着陆G值0.3秒','着陆G值0.4秒','着陆G值0.5秒',
                 '着陆G值0.6秒','着陆G值0.7秒','着陆G值0.8秒','着陆G值0.9秒','着陆G值1秒',
                 '姿态（俯仰角）','姿态（俯仰角）.1','姿态（俯仰角）.2','姿态（俯仰角）.3','姿态（俯仰角）.4',
                 '杆量','杆量.1','杆量.2','杆量.3','杆量.4',
                 '坡度（左负右正）','坡度（左负右正）.1',
                 '盘量','盘量.1','盘量.2','盘量.3','盘量.4']]
    return data


# In[3]:


data1 = pd.read_excel('Data\\Two\\LandingData\\1.xlsx', sheet_name='Sheet1')
data2 = pd.read_excel('Data\\Two\\LandingData\\2.xlsx', sheet_name='Sheet1')
data3 = pd.read_excel('Data\\Two\\LandingData\\3.xlsx', sheet_name='Sheet1')
data4 = pd.read_excel('Data\\Two\\LandingData\\4.xlsx', sheet_name='Sheet1')
data5 = pd.read_excel('Data\\Two\\LandingData\\5.xlsx', sheet_name='Sheet1')
data6 = pd.read_excel('Data\\Two\\LandingData\\6.xlsx', sheet_name='Sheet1')
data7 = pd.read_excel('Data\\Two\\LandingData\\7.xlsx', sheet_name='Sheet1')
data8 = pd.read_excel('Data\\Two\\LandingData\\8.xlsx', sheet_name='Sheet1')


# In[4]:


data1N = Sample(data1)
data2N = Sample(data2)
data3N = Sample(data3)
data4N = Sample(data4)
data5N = Sample(data5)
data6N = Sample(data6)
data7N = Sample(data7)
data8N = Sample(data8)


# In[5]:


data1N.to_csv('Data\\Two\\LandingData\\1N.csv', index=False)
data2N.to_csv('Data\\Two\\LandingData\\2N.csv', index=False)
data3N.to_csv('Data\\Two\\LandingData\\3N.csv', index=False)
data4N.to_csv('Data\\Two\\LandingData\\4N.csv', index=False)
data5N.to_csv('Data\\Two\\LandingData\\5N.csv', index=False)
data6N.to_csv('Data\\Two\\LandingData\\6N.csv', index=False)
data7N.to_csv('Data\\Two\\LandingData\\7N.csv', index=False)
data8N.to_csv('Data\\Two\\LandingData\\8N.csv', index=False)

