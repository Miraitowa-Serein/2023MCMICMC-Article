#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np


# In[2]:


data=pd.read_csv('Data\\First\\NNew\\1-7D.csv')
data


# # 数据标准化

# In[3]:


import sklearn.preprocessing as sp
StandardTransform = data.loc[:,~data.columns.isin(['Unnamed: 0'])]
StandardTransform


# In[4]:


ListColumns=list(StandardTransform.columns)
ListColumns


# In[5]:


StandardTransformScaler = sp.StandardScaler()
StandardTransformScaler = StandardTransformScaler.fit(StandardTransform)
StandardTransform = StandardTransformScaler.transform(StandardTransform)
StandardTransform = pd.DataFrame(StandardTransform)
StandardTransform.columns = ListColumns
StandardTransform


# # 数据归一化

# In[6]:


import sklearn.preprocessing as sp
MinMaxTransform = data.loc[:,~data.columns.isin(['Unnamed: 0'])]
MinMaxTransform


# In[7]:


MinMaxTransformScaler=sp.MinMaxScaler(feature_range=(0.002, 1))
MinMaxTransformScaler=MinMaxTransformScaler.fit(MinMaxTransform)
MinMaxTransform=MinMaxTransformScaler.transform(MinMaxTransform)
MinMaxTransform=pd.DataFrame(MinMaxTransform)
MinMaxTransform.columns=ListColumns
MinMaxTransform


# # PCA&层次聚类

# In[8]:


import matplotlib.pyplot as plt


# In[9]:


from sklearn.decomposition import PCA

pca = PCA()
pca.fit(StandardTransform)
evr = pca.explained_variance_ratio_

plt.figure(figsize=(12, 5))
plt.plot(range(0, len(evr)), evr.cumsum(), marker="d", linestyle="-")
plt.xlabel("Number of components",font='Times New Roman',fontsize=13)
plt.ylabel("Cumulative explained variance",font='Times New Roman',fontsize=13)
plt.xticks(font='Times New Roman',fontsize=12)
plt.yticks(font='Times New Roman',fontsize=12)
plt.tight_layout()
plt.savefig("Figures\\1-7 PCA Cumulative explained variance.pdf")


# In[10]:


import scipy.cluster.hierarchy as sch
plt.figure(figsize=(12, 6))
dendrogram = sch.dendrogram(sch.linkage(StandardTransform.T, method = 'ward'))
plt.rcParams['font.sans-serif']=['SimHei']
plt.rcParams['axes.unicode_minus']=False
plt.xlabel('指标',fontsize=13)
plt.ylabel('离差平方和',fontsize=13)
plt.xticks(font='Times New Roman',fontsize=11)
plt.yticks(font='Times New Roman',fontsize=11)
plt.savefig("Figures\\1-7 层次聚类树状图.pdf")


# 环境：风向、风速
# 人为操作：下降率、地速、起落架、G值、杆量、盘量、RUDD位置
# 飞机状态：姿态（俯仰角）、坡度、发动机N1值、磁航向、油门杆位置、下滑道偏差、航向道偏差、俯仰角率

# In[11]:


IFactor=MinMaxTransform.loc[:,['风向','风速','下降率','地速','起落架',
                               '着陆G值0.1秒','着陆G值0.2秒','着陆G值0.3秒','着陆G值0.4秒','着陆G值0.5秒','着陆G值0.6秒','着陆G值0.7秒','着陆G值0.8秒','着陆G值0.9秒','着陆G值1秒',
                               '姿态（俯仰角）','姿态（俯仰角）.1','姿态（俯仰角）.2','姿态（俯仰角）.3','姿态（俯仰角）.4',
                               '杆量','杆量.1','杆量.2','杆量.3','杆量.4',
                               '坡度（左负右正）','坡度（左负右正）.1',
                               '盘量','盘量.1','盘量.2','盘量.3','盘量.4',
                               '左侧发动机油门N1值','右侧发动机油门N1值','磁航向',
                               '左发油门杆位置（角度）','左发油门杆位置（角度）.1','右发油门杆位置（角度）','右发油门杆位置（角度）.1',
                               'RUDD位置','下滑道偏差（C）','下滑道偏差（L）','下滑道偏差（R）',
                               '航向道偏差（C）','航向道偏差（L）','航向道偏差（R）',
                               '俯仰角率']]
IFactor


# In[12]:


IFactorEnv=IFactor.loc[:,['风向','风速']]
IFactorEnv


# In[13]:


IFactorPeo=IFactor.loc[:,['下降率','地速','起落架',
                          '着陆G值0.1秒','着陆G值0.2秒','着陆G值0.3秒','着陆G值0.4秒','着陆G值0.5秒','着陆G值0.6秒','着陆G值0.7秒','着陆G值0.8秒','着陆G值0.9秒','着陆G值1秒',
                          '杆量','杆量.1','杆量.2','杆量.3','杆量.4',
                          '盘量','盘量.1','盘量.2','盘量.3','盘量.4',
                          'RUDD位置',]]
IFactorPeo


# In[14]:


IFactorMac=IFactor.loc[:,~IFactor.columns.isin(['风向','风速','下降率','地速','起落架',
                          '着陆G值0.1秒','着陆G值0.2秒','着陆G值0.3秒','着陆G值0.4秒','着陆G值0.5秒','着陆G值0.6秒','着陆G值0.7秒','着陆G值0.8秒','着陆G值0.9秒','着陆G值1秒',
                          '杆量','杆量.1','杆量.2','杆量.3','杆量.4',
                          '盘量','盘量.1','盘量.2','盘量.3','盘量.4','RUDD位置',])]
IFactorMac


# In[15]:


import copy
def ewm(data):
    label_need = data.keys()[:]
    data1 = data[label_need].values
    data2 = data1
    [m, n] = data2.shape
    data3 = copy.deepcopy(data2)
    p = copy.deepcopy(data3)
    for j in range(0, n):
        p[:, j] = data3[:, j] / sum(data3[:, j])
    e = copy.deepcopy(data3[0, :])
    for j in range(0, n):
        e[j] = -1 / np.log(m) * sum(p[:, j] * np.log(p[:, j]))
    w = (1 - e) / sum(1 - e)
    total = 0
    for sum_w in range(0, len(w)):
        total = total + w[sum_w]
    print(f'权重为：{w}，权重之和为：{total}')


# In[16]:


ewm(IFactorEnv)


# In[17]:


ewm(IFactorPeo)


# In[18]:


ewm(IFactorMac)

