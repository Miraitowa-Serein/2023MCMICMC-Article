#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
from yellowbrick.features import PCA
import matplotlib.pyplot as plt


# In[2]:


data=pd.read_excel("WordleClass.xlsx",sheet_name='ALL')
data


# In[3]:


classes=['Medium','Very Easy','Hard','Very Hard','Easy']
features=['1 try','2 tries','3 tries','4 tries','5 tries','6 tries','7 or more tries (X)','w1','w2','w3','w4','w5','Vowel_fre','Consonant_fre','Speech','Same_letter_fre','w1_fre','w2_fre','w3_fre','w4_fre','w5_fre']

plt.rcParams['font.sans-serif'] = ['Times New Roman']
plt.rcParams['font.size'] = '16'
plt.figure(figsize=(10,6))
X=data[features]
y=data['Class']
visualizer = PCA(scale=True, projection=3, classes=classes)
visualizer.fit_transform(X, y)
visualizer.show(outpath="figures\\PCA-3.pdf")


# In[4]:


classes=['Medium','Very Easy','Hard','Very Hard','Easy']
features=['1 try','2 tries','3 tries','4 tries','5 tries','6 tries','7 or more tries (X)','w1','w2','w3','w4','w5','Vowel_fre','Consonant_fre','Speech','Same_letter_fre','w1_fre','w2_fre','w3_fre','w4_fre','w5_fre']

plt.rcParams['font.sans-serif'] = ['Times New Roman']
plt.rcParams['font.size'] = '16'
plt.figure(figsize=(8,10))
X=data[features]
y=data['Class']
visualizer = PCA(scale=True, proj_features=3, classes=classes,heatmap=True)
visualizer.fit_transform(X, y)
visualizer.show(outpath="figures\\PCA-2.pdf")

