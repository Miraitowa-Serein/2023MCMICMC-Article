#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np


# In[2]:


data = pd.read_excel('Data\\Third\\附件3：飞行参数测量数据（剔除C）.xlsx', sheet_name='数据')
data


# In[3]:


data.info()


# In[4]:


data.isnull().sum()


# In[5]:


import missingno
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif']=['SimHei']
plt.rcParams['axes.unicode_minus']=False
missingno.bar(data, color=(190/255,190/255,190/255))
plt.tight_layout()
plt.savefig('Figures\\附件3缺失值.pdf')


# In[6]:


data.drop(labels=['机型','落地主操控',' V1_Method',' Vr_Method',
                  ' EvtsEStoG2',' EvtsG2toG3',' EvtsTDG3toSD',' TO_V1',
                  ' TO_V2',' Max(CAS-Vfe)F1',' Max(CAS-Vfe)F4',' Max(CAS-Vfe)F5',
                  ' Max(CAS-Vfe)F6',' Max(CAS-Vfe)F7',' Max(CAS-Vfe)F8',' Max(CAS-Vfe)F9',
                  ' Max(CAS-Vfe)F10',' Max(CAS-Vfe)Slat',' CAS-V2atG1',' Min(CAS-V2)TOtoG1',
                  ' Max(CAS-V2)TOtoG1',' (CAS-V2)atG2',' Min(CAS-V2)G1toG2',' Max(CAS-V2)G1toG2',
                  ' TD_Vref',' Max(CAS-Vfe)F2-1',' Max(CAS-Vfe)F6-1',' Max(CAS-Vfe)F7-1',
                  ' Max(CAS-Vfe)F8-1',' Max(CAS-Vfe)F9-1',' Max(CAS-Vfe)F10-1',' Max(CAS-Vfe)Slat-1',
                  ' SPDBKarmdTime',' SPDBKarmdAAL',' CAS-Vref_G3',' CAS-Vref_G2',
                  ' Min(CAS-Vref)G3toG2',' Max(CAS-Vref)G3toG2',' Max(G S)TDG3toG2',' Min(G S)TDG3toG2',
                  ' CAS-Vref_G1',' RoD_G1',' Pitch_G1',' Ave(PWR)_G1',
                  ' Min(CAS-Vref)G2toG1',' Max(CAS-Vref)G2toG1',' Max(G S)TDG2toG1',' Min(G S)TDG2toG1',
                  ' CAS-VrefAt50',' CAS-VrefAtTD',' Min(CAS-Vref)G1to50',' Max(CAS-Vref)G1toTD',
                  ' MaxGlideSG1to100',' MinGlideSG1to100',
                  ' TO Gate 1',' TO Gate 2',' TD Gate 3',' TD Gate 2',' TD Gate 1-1', ' TO_Vr'],axis=1,inplace=True)
data


# In[7]:


data.isnull().sum()


# In[8]:


data.fillna(data.mode().iloc[0], inplace=True)
data


# In[9]:


data.isnull().sum()


# In[10]:


data


# In[11]:


#标签编码
import sklearn.preprocessing as sp
le=sp.LabelEncoder()

Competency=le.fit_transform(data["落地主操控人员资质"])
V2Method=le.fit_transform(data[" V2_Method"])
Vref_Method=le.fit_transform(data[" Vref_Method"])
RoDMethod=le.fit_transform(data[" RoD_Method"])
MACHMethod=le.fit_transform(data[" MACH_Method"])

data["落地主操控人员资质"]=pd.DataFrame(Competency)
data[" V2_Method"]=pd.DataFrame(V2Method)
data[" Vref_Method"]=pd.DataFrame(Vref_Method)
data[" RoD_Method"]=pd.DataFrame(RoDMethod)
data[" MACH_Method"]=pd.DataFrame(MACHMethod)
data


# In[12]:


#数据标准化
StandardTransform = data.loc[:,~data.columns.isin(['落地主操控人员资质',' V2_Method',' Vref_Method',' RoD_Method',' MACH_Method'])]
StandardTransform


# In[13]:


ListDataColumns=list(StandardTransform.columns)
ListDataColumns


# In[14]:


StandardTransformScaler = sp.StandardScaler()
StandardTransformScaler = StandardTransformScaler.fit(StandardTransform)
StandardTransform = StandardTransformScaler.transform(StandardTransform)
StandardTransform = pd.DataFrame(StandardTransform)
StandardTransform.columns = ListDataColumns
StandardTransform


# In[15]:


dataLeave=data[['落地主操控人员资质',' V2_Method',' Vref_Method',' RoD_Method',' MACH_Method']]
dataLeave


# In[16]:


dataNew=pd.concat([dataLeave, StandardTransform],axis=1)
dataNew


# In[17]:


y=dataNew["落地主操控人员资质"]
X=dataNew.loc[:,~dataNew.columns.isin(['落地主操控人员资质'])]


# In[18]:


#数据集划分
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.1,random_state=2023)


# In[19]:


from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier

DecisionTree = DecisionTreeClassifier(random_state=2023)
DecisionTree = DecisionTree.fit(X_train, y_train)

RandomForest = RandomForestClassifier(random_state=2023)
RandomForest = RandomForest.fit(X_train, y_train)
RandomForest_score = RandomForest.score(X_test, y_test)
RandomForest_score


# In[20]:


std = np.std([i.feature_importances_ for i in RandomForest.estimators_], axis=0)
importances = DecisionTree.feature_importances_
feat_with_importance = pd.Series(importances, X.columns)
fig, ax = plt.subplots(figsize=(24,10))
feat_with_importance.plot.bar(yerr=std, ax=ax, color=(5/255,126/255,215/255))
ax.set_ylabel("Mean decrease in impurity",font='Times New Roman',size=14)
plt.yticks(font='Times New Roman',size=13)
plt.xticks(font='Times New Roman',size=13)
plt.tight_layout()
plt.savefig('Figures\\feature_importance_RF.pdf')


# In[21]:


from xgboost import XGBClassifier

XGB = XGBClassifier()
XGB.fit(X_train, y_train)
XGB_score = XGB.score(X_test, y_test)
XGB_score


# In[22]:


from xgboost import plot_importance
fig, ax = plt.subplots(figsize=(10,24))
plot_importance(XGB, height=0.4, ax=ax)
plt.xticks(fontsize=12, font='Times New Roman')
plt.yticks(fontsize=12, font='Times New Roman')
plt.tight_layout()
plt.savefig('Figures\\feature_importance_XGB.pdf')


# In[23]:


from sklearn.decomposition import PCA

pca = PCA()
pca.fit(X)
evr = pca.explained_variance_ratio_

plt.figure(figsize=(12, 5))
plt.plot(range(0, len(evr)), evr.cumsum(), marker="d", linestyle="-")
plt.xlabel("Number of components",font='Times New Roman',size=14)
plt.ylabel("Cumulative explained variance",font='Times New Roman',size=14)
plt.xticks(font='Times New Roman',size=12)
plt.yticks(font='Times New Roman',size=12)
plt.tight_layout()
plt.savefig('Figures\\资质PCA累计方差解释.pdf')


# In[24]:


#降维
y=dataNew["落地主操控人员资质"]
X=dataNew.loc[:,~dataNew.columns.isin(['落地主操控人员资质',' CASgearselected',' MinPRWTDG2toG1',' MaxLeft(LOC)TDG2toG1',' V2_Method',' MaxRight(LOC)TDG3toG2'])]
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.1,random_state=2023)


# In[25]:


from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier

DecisionTree = DecisionTreeClassifier(random_state=2023)
DecisionTree = DecisionTree.fit(X_train, y_train)

RandomForest = RandomForestClassifier(random_state=2023)
RandomForest = RandomForest.fit(X_train, y_train)
RandomForest_score = RandomForest.score(X_test, y_test)
RandomForest_score


# In[26]:


from xgboost import XGBClassifier

XGB = XGBClassifier(random_state=2023,)
XGB.fit(X_train, y_train)
XGB_score = XGB.score(X_test, y_test)
XGB_score


# In[27]:


from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error


# In[28]:


print(f'XGBoost平均绝对误差：'
      f'{mean_absolute_error(y_test, XGB.predict(X_test), sample_weight=None, multioutput="uniform_average")}\n'
      f'XGBoost均方误差：'
      f'{mean_squared_error(y_test, XGB.predict(X_test), sample_weight=None, multioutput="uniform_average")}')


# In[29]:


from yellowbrick.classifier import ROCAUC
classes=['A','F','J','M','T']
visualizer = ROCAUC(XGB, classes=classes)
visualizer.fit(X_train, y_train)
visualizer.score(X_test, y_test)
plt.xticks(font='Times New Roman')
plt.yticks(font='Times New Roman')
visualizer.show(outpath='Figures\\ROCAUC.pdf')


# In[30]:


from yellowbrick.classifier import ConfusionMatrix
classes=['A','F','J','M','T']
confusion_matrix = ConfusionMatrix(XGB, classes=classes)
confusion_matrix.fit(X_train, y_train)
confusion_matrix.score(X_test, y_test)
plt.xticks(font='Times New Roman',rotation=0)
plt.yticks(font='Times New Roman')
confusion_matrix.show(outpath='Figures\\ConfusionMatrix.pdf')


# In[31]:


from yellowbrick.classifier import ClassificationReport
visualizer = ClassificationReport(XGB, classes=classes, support=True, cmap='Greens')
visualizer.fit(X_train, y_train)
visualizer.score(X_test, y_test)
plt.xticks(font='Times New Roman')
plt.yticks(font='Times New Roman')
visualizer.show(outpath='Figures\\ClassificationReport.pdf')


# In[32]:


from sklearn.model_selection import cross_val_score
XGB_5K = cross_val_score(estimator=XGB,X=X_train,y=y_train,cv=5)
print(XGB_5K, '\n', XGB_5K.mean())
print(XGB_5K.std())

