#!/usr/bin/env python
# coding: utf-8

# In[1]:


from itertools import cycle
import matplotlib.pyplot as plt
import pandas as pd
import nltk
data=pd.read_excel("WordleClass.xlsx",sheet_name='ALL')
data


# In[2]:


plt.rcParams['font.sans-serif']=['Times New Roman']


# In[3]:


def dclass(s):
    if s==0:
        return 'Medium'
    elif s==1:
        return 'Very Easy'
    elif s==2:
        return 'Hard'
    elif s==3:
        return 'Very Hard'
    elif s==4:
        return 'Easy'


data['Class']=data['Class'].apply(dclass)

data['Average Tries']=(1*data['1 try']+2*data['2 tries']+3*data['3 tries']+4*data['4 tries']+5*data['5 tries']+6*data['6 tries']+7*data['7 or more tries (X)'])/100

pos_tags = nltk.pos_tag(list(data['Word']))
data['Speech']=pd.DataFrame(pos_tags)[1]

data['Letter average frequency']=(data['w1_fre']+data['w2_fre']+data['w3_fre']+data['w4_fre']+data['w5_fre'])/5

data


# In[4]:


data['Speech'].value_counts()


# In[5]:


groups = data.groupby('Class')
markers = ['1', 'x', '.','+','*']

fig, ax = plt.subplots()
for (name, group), marker in zip(groups, cycle(markers)):
    ax.plot(group.Date, group['Number of reported results'], marker=marker, linestyle='', ms=5, label=name)
ax.legend()
plt.xlabel('Date',fontsize=14)
plt.ylabel('Number of reported results',fontsize=14)
plt.savefig('figures\\WordleClass.pdf')


# In[6]:


data['Hard Rate']=data['Number in hard mode']/data['Number of reported results']
ax=data.plot.scatter(x='Date', y='Hard Rate')
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.xlabel('Date',fontsize=14)
plt.ylabel('Hard mode frequency',fontsize=14)
plt.tight_layout()
plt.savefig('figures\\选择困难游戏模式游戏的人数比例.pdf')


# In[7]:


import seaborn as sns
sns.set_style('ticks')
ax=sns.jointplot(x=data['Contest number'],y=data['Hard Rate'],kind='kde',height=4,shade=True).plot_joint(sns.regplot,scatter=True,color='#FF3333')
plt.tight_layout()
plt.savefig('figures\\选择困难游戏模式游戏的人数比例核密度.pdf')


# In[8]:


plt.rcParams['font.sans-serif']=['Times New Roman']
plt.style.use('ggplot')
sns.violinplot(x = "Class",
               y = "Letter average frequency",
               data = data,
               order = ['Very Easy','Easy','Medium','Hard','Very Hard'],
               split = True,)
plt.savefig("figures\\单词字母平均出现率与难度.pdf")


# In[9]:


sns.violinplot(x = "Class",
               y = "Hard Rate",
               data = data,
               order = ['Very Easy','Easy','Medium','Hard','Very Hard'],
               split = True,)
plt.savefig("figures\\选择困难模式占比与难度.pdf")


# In[10]:


sns.violinplot(x = "Speech",
               y = "Hard Rate",
               data = data,
               order = ['NN','JJ','VBP','VBD','RB','VB','NNS','VBN','IN','VBZ'],
               split = True,)
plt.savefig("figures\\选择困难模式占比与词性.pdf")


# In[11]:


sns.violinplot(x = "Class",
               y = "Average Tries",
               data = data,
               order = ['Very Easy','Easy','Medium','Hard','Very Hard'],
               split = True,)
plt.savefig("figures\\平均尝试次数与难度.pdf")


# In[12]:


sns.violinplot(x = "Speech",
               y = "Average Tries",
               data = data,
               order = ['NN','JJ','VBP','VBD','RB','VB','NNS','VBN','IN','VBZ'],
               split = True,)
plt.savefig("figures\\平均尝试次数与词性.pdf")


# In[13]:


sns.violinplot(x = "Vowel_fre",
               y = "Average Tries",
               data = data,
               order = [1,2,3],
               split = True,)
plt.savefig("figures\\平均尝试次数与单词中元音个数.pdf")


# In[14]:


sns.violinplot(x = "Same_letter_fre",
               y = "Average Tries",
               data = data,
               order = [0,2,3,4],
               split = True,)
plt.savefig("figures\\平均尝试次数与单词中相同字母个数.pdf")


# In[15]:


plt.style.use('ggplot')
plt.rcParams['font.sans-serif']=['Times New Roman']
data.plot(x='Contest number',y=['1 try','2 tries','3 tries','4 tries'],figsize=(10,6))
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.xlabel('Percentage',fontsize=14)
plt.ylabel('Contest number',fontsize=14)
plt.savefig('figures\\尝试次数分布1.pdf')


# In[16]:


data.plot(x='Contest number',y=['5 tries','6 tries','7 or more tries (X)'],figsize=(10,6))
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.xlabel('Percentage',fontsize=14)
plt.ylabel('Contest number',fontsize=14)
plt.savefig('figures\\尝试次数分布2.pdf')

