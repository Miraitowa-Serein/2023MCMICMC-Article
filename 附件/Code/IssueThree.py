#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
data=pd.read_excel('Data_Wordle_All_Features.xlsx',sheet_name='Data_Wordle_All_Features')
data


# In[2]:


dataNew=data[['1 try','2 tries','3 tries','4 tries','5 tries','6 tries','7 or more tries (X)','w1','w2','w3','w4','w5','Vowel_fre','Consonant_fre','Speech','Same_letter_fre','w1_fre','w2_fre','w3_fre','w4_fre','w5_fre']]


# In[3]:


import scipy.cluster.hierarchy as sch
import matplotlib.pyplot as plt
plt.figure(figsize=(24, 6))
dendrogram = sch.dendrogram(sch.linkage(dataNew, method = 'ward'))
plt.rcParams['font.sans-serif']=['Times New Roman']
plt.rcParams['axes.unicode_minus']=False
plt.xlabel('Contest number',fontsize=14)
plt.ylabel('Sum of squares of deviations',fontsize=14)
plt.xticks([],font='Times New Roman',fontsize=12)
plt.yticks(font='Times New Roman',fontsize=12)
plt.tight_layout()
plt.savefig("figures\\层次聚类树状图.pdf")


# In[4]:


from yellowbrick.cluster import KElbowVisualizer
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from yellowbrick.cluster import InterclusterDistance
from yellowbrick.cluster import SilhouetteVisualizer


# In[5]:


model = KElbowVisualizer(KMeans(random_state = 20222023), k=12)
model.fit(dataNew)
model.poof(outpath="figures\\肘部法则.pdf")


# In[6]:


n_clusters = 5
cluster = KMeans(n_clusters = n_clusters, random_state = 20222023).fit(dataNew)
y_pred = cluster.labels_
data['Class']=y_pred
data


# In[7]:


# 输出5个类别数据
def ClassDataOutPut(i):
    data[data['Class']==i].to_excel(f'Class\\Class {i}.xlsx',sheet_name='Class')


for i in range(5):
    ClassDataOutPut(i)


# In[8]:


silhouette_score(dataNew, y_pred)


# In[9]:


model = SilhouetteVisualizer(cluster)
model.fit(dataNew)
model.poof(outpath='figures\\轮廓系数.pdf')


# In[10]:


model = InterclusterDistance(cluster)
model.fit(dataNew)
model.poof(outpath='figures\\类间距离.pdf')


# In[11]:


from sklearn.decomposition import PCA
pca = PCA(n_components=2)
DataNewPCA = pca.fit_transform(dataNew)
x0, y0= [], []
x1, y1= [], []
x2, y2= [], []
x3, y3= [], []
x4, y4= [], []

plt.rcParams['font.sans-serif'] = ['Times New Roman']
plt.rcParams['axes.unicode_minus'] = False

for index, value in enumerate(y_pred):
    if value == 0:
        x0.append(DataNewPCA[index][0])
        y0.append(DataNewPCA[index][1])
    elif value == 1:
        x1.append(DataNewPCA[index][0])
        y1.append(DataNewPCA[index][1])
    elif value == 2:
        x2.append(DataNewPCA[index][0])
        y2.append(DataNewPCA[index][1])
    elif value == 3:
        x3.append(DataNewPCA[index][0])
        y3.append(DataNewPCA[index][1])
    elif value == 4:
        x4.append(DataNewPCA[index][0])
        y4.append(DataNewPCA[index][1])

plt.figure(figsize=(10, 10))

# #定义坐标轴
k = 200
plt.scatter(x0, y0,s=k)
plt.scatter(x1, y1,s=k)
plt.scatter(x2, y2,s=k)
plt.scatter(x3, y3,s=k)
plt.scatter(x4, y4,s=k)
plt.legend(['Medium','Very Easy','Hard','Very Hard','Easy'])
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.xlabel('PC2',fontsize=14)
plt.ylabel('PC1',fontsize=14)
plt.tight_layout()
plt.savefig('figures\\聚类散点图.pdf')


# In[12]:


EERIE=pd.read_excel("EERIE_Result(Initially+EERIE&EERIE).xlsx",sheet_name='EERIE_Result')
EERIE


# In[13]:


data


# In[14]:


X=data[['1 try','2 tries','3 tries','4 tries','5 tries','6 tries','7 or more tries (X)','w1','w2','w3','w4','w5','Vowel_fre','Consonant_fre','Speech','Same_letter_fre','w1_fre','w2_fre','w3_fre','w4_fre','w5_fre']]
X


# In[15]:


from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
y=data['Class']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=20222023)
SVM = SVC(random_state=20222023)
SVM.fit(X_train, y_train)
SVM_score = SVM.score(X_test, y_test)
SVM_score


# In[16]:


SVM.predict(EERIE[['1 try','2 tries','3 tries','4 tries','5 tries','6 tries','7 or more tries (X)','w1','w2','w3','w4','w5','Vowel_fre','Consonant_fre','Speech','Same_letter_fre','w1_fre','w2_fre','w3_fre','w4_fre','w5_fre']])


# In[17]:


from yellowbrick.classifier import ConfusionMatrix
classes=['Medium','Very Easy','Hard','Very Hard','Easy']
confusion_matrix = ConfusionMatrix(SVM, classes=classes, cmap='BuGn')
confusion_matrix.fit(X_train, y_train)
confusion_matrix.score(X_test, y_test)
plt.xticks(font='Times New Roman')
plt.yticks(font='Times New Roman')
confusion_matrix.show(outpath='figures\\混淆矩阵热力图.pdf')


# In[18]:


from yellowbrick.classifier import ClassificationReport
visualizer = ClassificationReport(SVM, classes=classes, support=True, cmap='Blues')
visualizer.fit(X_train, y_train)
visualizer.score(X_test, y_test)
plt.xticks(font='Times New Roman')
plt.yticks(font='Times New Roman')
visualizer.show(outpath='figures\\分类报告.pdf')


# In[19]:


from yellowbrick.classifier import ROCAUC
visualizer = ROCAUC(SVM)
visualizer.fit(X_train, y_train)
visualizer.score(X_test, y_test)
plt.xticks(font='Times New Roman')
plt.yticks(font='Times New Roman')
visualizer.show(outpath='figures\\ROCAUC曲线.pdf')


# In[20]:


from yellowbrick.classifier import ClassPredictionError
visualizer = ClassPredictionError(SVM, classes=classes)
visualizer.fit(X_train, y_train)
visualizer.score(X_test, y_test)
plt.xticks(font='Times New Roman')
plt.yticks(font='Times New Roman')
visualizer.show(outpath='figures\\分类预测结果.pdf')

