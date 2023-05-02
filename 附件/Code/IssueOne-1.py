#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import seaborn as sns
data=pd.read_excel("Data_Wordle_New.xlsx",sheet_name="Sheet1")
data


# In[2]:


import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['Times New Roman']
plt.rcParams['axes.unicode_minus'] = False


# In[3]:


sns.set_style('ticks')
ax=sns.jointplot(x=data['Contest number'],y=data['Number of reported results'],kind='kde',height=4,shade=True).plot_joint(sns.regplot,scatter=True,color='#FF3333')
plt.tight_layout()
plt.savefig('figures\\核密度估计值.pdf')


# In[4]:


import scipy.stats as st
import matplotlib.pyplot as plt

plt.figure(figsize=(16, 6))
y = data['Number of reported results']
plt.subplot(121)
sns.distplot(y, kde=False, fit=st.johnsonsu, color='Red')
plt.xticks(font='Times New Roman',fontsize=12)
plt.yticks(font='Times New Roman',fontsize=12)
plt.xlabel('Number of reported results',font='Times New Roman',fontsize=14)

plt.subplot(122)
st.probplot(y, dist="norm", plot=plt)
plt.xticks(font='Times New Roman',fontsize=12)
plt.yticks(font='Times New Roman',fontsize=12)
plt.xlabel('Theoretical quantiles',font='Times New Roman',fontsize=14)
plt.ylabel('Ordered Values',font='Times New Roman',fontsize=14)
plt.title('Probability Plot',font='Times New Roman',fontsize=16)

plt.tight_layout()
plt.savefig("figures\\正态分布分析.pdf")


# In[5]:


data=pd.read_excel("Data_Wordle_All.xlsx",sheet_name='Sheet1')
data


# In[6]:


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


# In[7]:


X1=data['Contest number']
y1=data['Number of reported results']


# In[8]:


from sklearn.model_selection import train_test_split
X1_train,X1_test,y1_train,y1_test=train_test_split(X1,y1,test_size=0.1,random_state=20222023)


# In[9]:


from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.metrics import r2_score
import xgboost as xgb

XGB_All = xgb.XGBRegressor(objective='reg:squarederror',n_estimators=100,
                           max_depth=9,learning_rate=0.1,
                           subsample=0.8,reg_lambda= 0.5,
                           reg_alpha= 0,gamma= 0,
                           colsample_bytree=0.6,min_child_weight=5)
XGB_All.fit(X1_train,y1_train)
pre_All=XGB_All.predict(X1)


# In[10]:


mae = mean_absolute_error(data['Number of reported results'], pre_All)
mse = mean_squared_error(data['Number of reported results'], pre_All)
rmse = mse**(1/2)
r2=r2_score(data['Number of reported results'], pre_All)


# In[11]:


print(mae,mse,rmse,r2)


# In[12]:


plt.rcParams['font.sans-serif'] = ['Times New Roman']


# In[13]:


data['Number of reported results Pre']=pre_All
plt.figure(figsize=(8, 6))
data.plot(x='Contest number', y=['Number of reported results', 'Number of reported results Pre'])
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.xlabel('Contest number',fontsize=14)
plt.ylabel('Number of reported results',fontsize=14)
plt.tight_layout()
plt.savefig('figures\\XGBoost预测结果（总人数）.pdf')


# In[14]:


X2=data[['Contest number','Number of reported results']]
y2=data['Number in hard mode']
X2_train,X2_test,y2_train,y2_test=train_test_split(X2,y2,test_size=0.1,random_state=20222023)
XGB_Hard = xgb.XGBRegressor(objective='reg:squarederror',n_estimators=100,
                            max_depth=9,learning_rate=0.1,
                            subsample=0.8,reg_lambda= 0.5,
                            reg_alpha= 0,gamma= 0,
                            colsample_bytree=0.6,min_child_weight=5)
XGB_Hard.fit(X2_train,y2_train)
pre_Hard=XGB_Hard.predict(X2)


# In[15]:


mae = mean_absolute_error(data['Number in hard mode'], pre_Hard)
mse = mean_squared_error(data['Number in hard mode'], pre_Hard)
rmse = mse**(1/2)
r2=r2_score(data['Number in hard mode'], pre_Hard)
print(mae,mse,rmse,r2)


# In[16]:


data['Number in hard mode Pre']=pre_Hard
plt.figure(figsize=(8, 6))
data.plot(x='Contest number', y=['Number in hard mode', 'Number in hard mode Pre'])
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.xlabel('Contest number',fontsize=14)
plt.ylabel('Number in hard mode',fontsize=14)
plt.tight_layout()
plt.savefig('figures\\XGBoost预测结果（困难人数）.pdf')


# In[17]:


import yellowbrick
yellowbrick.style.rcmod.set_aesthetic(font='Times New Roman',font_scale=2)


# In[18]:


from yellowbrick.regressor import PredictionError
model = PredictionError(XGB_All)
model.fit(X1_train, y1_train)
model.score(X1_test, y1_test)
model.poof(outpath='figures\\XGBoost预测误差（总人数）.pdf')


# In[19]:


from yellowbrick.regressor import PredictionError
model = PredictionError(XGB_Hard)
model.fit(X2_train, y2_train)
model.score(X2_test, y2_test)
model.poof(outpath='figures\\XGBoost预测误差（困难人数）.pdf')


# In[20]:


EERIE_Pre_All=pd.read_excel("EERIE_Pre.xlsx",sheet_name='All')
EERIE_Pre_All


# In[21]:


X_pre_all=XGB_All.predict(EERIE_Pre_All['Contest number'])
X_pre_all


# In[22]:


EERIE_Pre_Hard=pd.read_excel("EERIE_Pre.xlsx",sheet_name='Hard')
EERIE_Pre_Hard


# In[23]:


X_pre_hard=XGB_Hard.predict(EERIE_Pre_Hard[['Contest number','Number of reported results']])
X_pre_hard

