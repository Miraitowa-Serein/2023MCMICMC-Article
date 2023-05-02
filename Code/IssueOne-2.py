#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


data = pd.read_excel("Data_Wordle_All_Features.xlsx", sheet_name="Data_Wordle_All_Features")
data


# In[3]:


X=data[['1 try','2 tries','3 tries','4 tries','5 tries','6 tries','7 or more tries (X)','w1','w2','w3','w4','w5','Vowel_fre','Consonant_fre','Same_letter_fre','Speech','w1_fre','w2_fre','w3_fre','w4_fre','w5_fre']]
X


# In[4]:


plt.subplots(figsize = (21, 21))
sns.heatmap(X.corr(method='pearson'),linewidths=0.1,vmax=1.0,square=True,linecolor='white',annot=True,annot_kws={'size':12})
plt.rcParams['font.sans-serif']=['Times New Roman']
plt.rcParams['axes.unicode_minus']=False
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.title('Pearson\'s correlation coefficient', size=16)
plt.savefig('figures\\皮尔逊相关系数.pdf',bbox_inches='tight')

