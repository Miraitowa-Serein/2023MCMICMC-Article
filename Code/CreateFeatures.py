#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import nltk


# In[2]:


data = pd.read_excel("Data_Wordle_All.xlsx", sheet_name="Sheet1")
data


# In[3]:


data['Letters']=data['Word'].apply(lambda x:str(list(x))[1:-1].replace("'","").replace(" ",""))
data['w1'],data['w2'],data['w3'],data['w4'],data['w5']=data['Letters'].str.split(',',n=4).str
letter = [str(chr(i)) for i in range(ord('a'),ord('z')+1)]
letter_map = dict(zip(letter,range(1,27)))
data['w1'] = data['w1'].map(letter_map)
data['w2'] = data['w2'].map(letter_map)
data['w3'] = data['w3'].map(letter_map)
data['w4'] = data['w4'].map(letter_map)
data['w5'] = data['w5'].map(letter_map)
data


# In[4]:


Vowel = ['a','e','i','o','u']
Consonant = list(set(letter).difference(set(Vowel)))


def count_vowel(s):
    c = 0
    for i in range(len(s)):
        if s[i] in Vowel:
            c+=1
    return c


def count_consonant(s):
    c = 0
    for i in range(len(s)):
        if s[i] in Consonant:
            c+=1
    return c


data['Vowel_fre'] = data['Word'].apply(lambda x:count_vowel(x))
data['Consonant_fre'] = data['Word'].apply(lambda x:count_consonant(x))
data


# In[5]:


pos_tags = nltk.pos_tag(list(data['Word']))
data['Speech']=pd.DataFrame(pos_tags)[1]
data


# In[6]:


def count_same_letter(s):
    d={}
    for char in set(s):
        d[char]=s.count(char)

    sum = 0
    for i in d:
        if d[i]>1:
            sum = sum + d[i]

    return sum


# In[7]:


data['Same_letter_fre'] = data['Word'].apply(lambda x:count_same_letter(x))
data


# In[8]:


Frequency=pd.read_excel("Letter_Frequency.xlsx",sheet_name="Sheet1")
Frequency_map=dict(zip(Frequency['N'],Frequency['Frequency']))
data['w1_fre']=data['w1']
data['w2_fre']=data['w2']
data['w3_fre']=data['w3']
data['w4_fre']=data['w4']
data['w5_fre']=data['w5']
data.replace({'w1_fre':Frequency_map,'w2_fre':Frequency_map,'w3_fre':Frequency_map,'w4_fre':Frequency_map,'w5_fre':Frequency_map},inplace=True)
data


# In[9]:


import sklearn.preprocessing as sp
le=sp.LabelEncoder()
data['Speech']=le.fit_transform(data['Speech'])
data

