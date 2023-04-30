#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np


# In[2]:


data1=pd.read_excel("Data\\First\\New\\201404070532.xlsx",sheet_name="201404070532")
data2=pd.read_excel("Data\\First\\New\\201404071917.xlsx",sheet_name="201404071917")
data3=pd.read_excel("Data\\First\\New\\201404080617.xlsx",sheet_name="201404080617")
data4=pd.read_excel("Data\\First\\New\\201404081034.xlsx",sheet_name="201404081034")
data5=pd.read_excel("Data\\First\\New\\201404090110.xlsx",sheet_name="201404090110")
data6=pd.read_excel("Data\\First\\New\\201404091701.xlsx",sheet_name="201404091701")
data7=pd.read_excel("Data\\First\\New\\201404100843.xlsx",sheet_name="201404100843")
data8=pd.read_excel("Data\\First\\New\\201404101159.xlsx",sheet_name="201404101159")


# In[3]:


def MaxFactor(data):
    data['Max_G']=data[['着陆G值0.1秒','着陆G值0.2秒','着陆G值0.3秒','着陆G值0.4秒','着陆G值0.5秒',
                        '着陆G值0.6秒','着陆G值0.7秒','着陆G值0.8秒','着陆G值0.9秒','着陆G值1秒']].max(axis=1)
    data['Max_姿态']=data[['姿态（俯仰角）','姿态（俯仰角）.1','姿态（俯仰角）.2','姿态（俯仰角）.3','姿态（俯仰角）.4']].max(axis=1)
    data['Max_杆量']=data[['杆量','杆量.1','杆量.2','杆量.3','杆量.4']].max(axis=1)
    data['Max_坡度']=data[['坡度（左负右正）','坡度（左负右正）.1']].max(axis=1)
    data['Max_盘量']=data[['盘量','盘量.1','盘量.2','盘量.3','盘量.4']].max(axis=1)
    return data


# In[4]:


data1A=MaxFactor(data1)
data2A=MaxFactor(data2)
data3A=MaxFactor(data3)
data4A=MaxFactor(data4)
data5A=MaxFactor(data5)
data6A=MaxFactor(data6)
data7A=MaxFactor(data7)
data8A=MaxFactor(data8)


# In[5]:


def FactorSelect(data):
    New=data[['海拔高度','无线电高度',
              '计算空速','地速',
              'Max_G','Max_姿态','Max_杆量','Max_坡度','Max_盘量',
              '风向','风速',
              '下滑道偏差（C）','下滑道偏差（L）','下滑道偏差（R）',
              '航向道偏差（C）','航向道偏差（L）','航向道偏差（R）',
              '俯仰角率','下降率','飞机重量']]
    return New


# In[6]:


data1AN=FactorSelect(data1A)
data2AN=FactorSelect(data2A)
data3AN=FactorSelect(data3A)
data4AN=FactorSelect(data4A)
data5AN=FactorSelect(data5A)
data6AN=FactorSelect(data6A)
data7AN=FactorSelect(data7A)
data8AN=FactorSelect(data8A)


# In[7]:


import matplotlib.pyplot as plt


# In[8]:


def draw_HBWXD(data,i):
    y1=data['海拔高度']
    y2=data['无线电高度']
    plt.rcParams['font.sans-serif']=['SimHei']
    plt.rcParams['axes.unicode_minus']=False
    plt.figure(figsize=(14, 4))
    plt.plot(y1, ms=5, label='海拔高度', color="blue")
    plt.plot(y2, ms=5, label='无线电高度', color="red")
    plt.xlabel('时刻',size=13)
    plt.ylabel('量值',size=13)
    plt.xticks(font='Times New Roman',size=12)
    plt.yticks(font='Times New Roman',size=12)
    plt.legend()
    plt.savefig("Figures\\全航段\\第"+str(i)+"次航班海拔高度与无线电高度.pdf")


# In[9]:


def draw_KSDS(data,i):
    y1=data['计算空速']
    y2=data['地速']
    y3=data['风速']
    plt.rcParams['font.sans-serif']=['SimHei']
    plt.rcParams['axes.unicode_minus']=False
    plt.figure(figsize=(14, 4))
    plt.plot(y1, ms=5, label='计算空速', color="blue")
    plt.plot(y2, ms=5, label='地速', color="red")
    plt.plot(y3, ms=5, label='风速', color="green")
    plt.xlabel('时刻',size=13)
    plt.ylabel('量值',size=13)
    plt.xticks(font='Times New Roman',size=12)
    plt.yticks(font='Times New Roman',size=12)
    plt.legend()
    plt.savefig("Figures\\全航段\\第"+str(i)+"次航班空速、地速、风速.pdf")


# In[10]:


def draw_FV(data,i):
    y1=data['Max_G']
    y2=data['Max_姿态']
    y3=data['Max_杆量']
    y4=data['Max_坡度']
    y5=data['Max_盘量']
    plt.rcParams['font.sans-serif']=['SimHei']
    plt.rcParams['axes.unicode_minus']=False
    plt.figure(figsize=(14, 5))
    plt.plot(y1, ms=5, label='G', color="green")
    plt.plot(y2, ms=5, label='姿态', color="black")
    plt.plot(y3, ms=5, label='杆量', color="blue")
    plt.plot(y4, ms=5, label='坡度', color="orange")
    plt.plot(y5, ms=5, label='盘量', color="red")
    plt.xlabel('时刻',size=13)
    plt.ylabel('量值',size=13)
    plt.xticks(font='Times New Roman',size=12)
    plt.yticks(font='Times New Roman',size=12)
    plt.legend()
    plt.savefig("Figures\\全航段\\第"+str(i)+"次航班飞行参量.pdf")


# In[11]:


def draw_XHDPC(data,i):
    y1=data['下滑道偏差（C）']
    y2=data['下滑道偏差（L）']
    y3=data['下滑道偏差（R）']
    plt.rcParams['font.sans-serif']=['SimHei']
    plt.rcParams['axes.unicode_minus']=False
    plt.figure(figsize=(14, 4))
    plt.plot(y1, ms=5, label='下滑道偏差（C）', color="blue")
    plt.plot(y2, ms=5, label='下滑道偏差（L）', color="red")
    plt.plot(y3, ms=5, label='下滑道偏差（R）', color="orange")
    plt.xlabel('时刻',size=13)
    plt.ylabel('量值',size=13)
    plt.xticks(font='Times New Roman',size=12)
    plt.yticks(font='Times New Roman',size=12)
    plt.legend()
    plt.savefig("Figures\\全航段\\第"+str(i)+"次航班下滑道偏差.pdf")


# In[12]:


def draw_HXDPC(data,i):
    y1=data['航向道偏差（C）']
    y2=data['航向道偏差（L）']
    y3=data['航向道偏差（R）']
    plt.rcParams['font.sans-serif']=['SimHei']
    plt.rcParams['axes.unicode_minus']=False
    plt.figure(figsize=(14, 4))
    plt.plot(y1, ms=5, label='航向道偏差（C）', color="green")
    plt.plot(y2, ms=5, label='航向道偏差（L）', color="red")
    plt.plot(y3, ms=5, label='航向道偏差（R）', color="orange")
    plt.xlabel('时刻',size=13)
    plt.ylabel('量值',size=13)
    plt.xticks(font='Times New Roman',size=12)
    plt.yticks(font='Times New Roman',size=12)
    plt.legend()
    plt.savefig("Figures\\全航段\\第"+str(i)+"次航班航向道偏差.pdf")


# In[13]:


def draw_FYJXJ(data,i):
    y1=data['俯仰角率']
    y2=data['下降率']
    plt.rcParams['font.sans-serif']=['SimHei']
    plt.rcParams['axes.unicode_minus']=False
    plt.figure(figsize=(14, 4))
    plt.plot(y1, ms=5, label='俯仰角率', color="green")
    plt.plot(y2, ms=5, label='下降率', color="red")
    plt.xlabel('时刻',size=13)
    plt.ylabel('量值',size=13)
    plt.xticks(font='Times New Roman',size=12)
    plt.yticks(font='Times New Roman',size=12)
    plt.legend()
    plt.savefig("Figures\\全航段\\第"+str(i)+"次航班俯仰角率、下降率.pdf")


# In[14]:


def draw_M(data,i):
    y=data['飞机重量']
    plt.rcParams['font.sans-serif']=['SimHei']
    plt.rcParams['axes.unicode_minus']=False
    plt.figure(figsize=(14, 4))
    plt.plot(y, ms=5, label='飞机重量', color="green")
    plt.xlabel('时刻',size=13)
    plt.ylabel('量值',size=13)
    plt.xticks(font='Times New Roman',size=12)
    plt.yticks(font='Times New Roman',size=12)
    plt.legend()
    plt.savefig("Figures\\全航段\\第"+str(i)+"次航班飞机重量.pdf")


# # 飞机重量

# In[15]:


draw_M(data1AN,1)
draw_M(data2AN,2)
draw_M(data3AN,3)
draw_M(data4AN,4)
draw_M(data5AN,5)
draw_M(data6AN,6)
draw_M(data7AN,7)
draw_M(data8AN,8)


# # 下降率、俯仰角率

# In[16]:


draw_FYJXJ(data1AN,1)
draw_FYJXJ(data2AN,2)
draw_FYJXJ(data3AN,3)
draw_FYJXJ(data4AN,4)
draw_FYJXJ(data5AN,5)
draw_FYJXJ(data6AN,6)
draw_FYJXJ(data7AN,7)
draw_FYJXJ(data8AN,8)


# # 航向道偏差

# In[17]:


draw_HXDPC(data1AN,1)
draw_HXDPC(data2AN,2)
draw_HXDPC(data3AN,3)
draw_HXDPC(data4AN,4)
draw_HXDPC(data5AN,5)
draw_HXDPC(data6AN,6)
draw_HXDPC(data7AN,7)
draw_HXDPC(data8AN,8)


# # 下滑道偏差

# In[18]:


draw_XHDPC(data1AN,1)
draw_XHDPC(data2AN,2)
draw_XHDPC(data3AN,3)
draw_XHDPC(data4AN,4)
draw_XHDPC(data5AN,5)
draw_XHDPC(data6AN,6)
draw_XHDPC(data7AN,7)
draw_XHDPC(data8AN,8)


# # 五因素

# In[19]:


draw_FV(data1AN,1)
draw_FV(data2AN,2)
draw_FV(data3AN,3)
draw_FV(data4AN,4)
draw_FV(data5AN,5)
draw_FV(data6AN,6)
draw_FV(data7AN,7)
draw_FV(data8AN,8)


# # 速度

# In[20]:


draw_KSDS(data1AN,1)
draw_KSDS(data2AN,2)
draw_KSDS(data3AN,3)
draw_KSDS(data4AN,4)
draw_KSDS(data5AN,5)
draw_KSDS(data6AN,6)
draw_KSDS(data7AN,7)
draw_KSDS(data8AN,8)


# # 海拔、无线电高度

# In[21]:


draw_HBWXD(data1AN,1)
draw_HBWXD(data2AN,2)
draw_HBWXD(data3AN,3)
draw_HBWXD(data4AN,4)
draw_HBWXD(data5AN,5)
draw_HBWXD(data6AN,6)
draw_HBWXD(data7AN,7)
draw_HBWXD(data8AN,8)

