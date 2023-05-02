#!/usr/bin/env python
# coding: utf-8

# In[1]:


import matplotlib.pyplot as plt
import pandas as pd
data=pd.read_excel("WordleClass.xlsx",sheet_name='ALL')
data


# In[2]:


from yellowbrick.features import RadViz
from sklearn.model_selection import train_test_split
classes=['Medium','Very Easy','Hard','Very Hard','Easy']
features=['1 try','2 tries','3 tries','4 tries','5 tries','6 tries','7 or more tries (X)','w1','w2','w3','w4','w5','Vowel_fre','Consonant_fre','Speech','Same_letter_fre','w1_fre','w2_fre','w3_fre','w4_fre','w5_fre']

plt.rcParams['font.sans-serif'] = ['Times New Roman']
plt.rcParams['font.size'] = '16'
plt.figure(figsize=(10,6))
X=data[features]
y=data['Class']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=20222023)
visualizer = RadViz(classes=classes, features=features)
visualizer.fit(X, y)
visualizer.transform(X)
visualizer.poof(outpath='figures\\RadViz.pdf',bbox_inches="tight")


# In[3]:


from yellowbrick.features import Rank1D
plt.figure(figsize=(10,5))
visualizer = Rank1D(features=features, algorithm='shapiro')
visualizer.fit(X, y)
visualizer.transform(X)
# plt.tight_layout()
visualizer.poof(outpath="figures\\Rank1D.pdf")


# In[4]:


from yellowbrick.features import Rank2D
visualizer = Rank2D(features=features, algorithm='pearson')
plt.figure(figsize=(8,6))
visualizer.fit(X, y)
visualizer.transform(X)
plt.tight_layout()
visualizer.poof(outpath='figures\\Rank2D.pdf')

