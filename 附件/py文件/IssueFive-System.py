#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd


# In[2]:


data1=pd.read_excel("Data\\First\\New\\201404070532.xlsx",sheet_name="201404070532")
data2=pd.read_excel("Data\\First\\New\\201404071917.xlsx",sheet_name="201404071917")
data3=pd.read_excel("Data\\First\\New\\201404080617.xlsx",sheet_name="201404080617")
data4=pd.read_excel("Data\\First\\New\\201404081034.xlsx",sheet_name="201404081034")
data5=pd.read_excel("Data\\First\\New\\201404090110.xlsx",sheet_name="201404090110")
data6=pd.read_excel("Data\\First\\New\\201404091701.xlsx",sheet_name="201404091701")
data7=pd.read_excel("Data\\First\\New\\201404100843.xlsx",sheet_name="201404100843")
data8=pd.read_excel("Data\\First\\New\\201404101159.xlsx",sheet_name="201404101159")
dataAll=pd.read_excel("Data\\First\\New\\汇总.xlsx",sheet_name="Sheet1")


# In[3]:


def MaxFactor(data):
    data['G值']=data[['着陆G值0.1秒','着陆G值0.2秒','着陆G值0.3秒','着陆G值0.4秒','着陆G值0.5秒',
                      '着陆G值0.6秒','着陆G值0.7秒','着陆G值0.8秒','着陆G值0.9秒','着陆G值1秒']].max(axis=1)
    data['姿态']=data[['姿态（俯仰角）','姿态（俯仰角）.1','姿态（俯仰角）.2','姿态（俯仰角）.3','姿态（俯仰角）.4']].max(axis=1)
    data['杆量']=data[['杆量','杆量.1','杆量.2','杆量.3','杆量.4']].max(axis=1)
    data['坡度']=data[['坡度（左负右正）','坡度（左负右正）.1']].max(axis=1)
    data['盘量']=data[['盘量','盘量.1','盘量.2','盘量.3','盘量.4']].max(axis=1)
    data['左油门杆位置']=data[['左发油门杆位置（角度）','左发油门杆位置（角度）.1']].max(axis=1)
    data['右油门杆位置']=data[['右发油门杆位置（角度）','右发油门杆位置（角度）.1']].max(axis=1)
    return data


# In[4]:


def FactorSelect(data):
    New=data[['具体时间',
              '计算空速','G值','姿态','杆量','坡度','盘量',
              '下滑道偏差（C）','下滑道偏差（L）','下滑道偏差（R）',
              '航向道偏差（C）','航向道偏差（L）','航向道偏差（R）',
              '俯仰角率','下降率','RUDD位置','左油门杆位置','右油门杆位置']]
    return New


# In[5]:


dataAI=MaxFactor(dataAll)


# In[6]:


dataAF=FactorSelect(dataAI)


# In[7]:


dataAF


# In[8]:


import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif']=['SimHei']
plt.rcParams['axes.unicode_minus']=False


# In[9]:


def drawHist(x):
    plt.figure(figsize=(10,8))
    dataAF.hist(column=x,bins=150,color='#9ACD32',grid=False)
    plt.xticks(font='Times New Roman',fontsize=12)
    plt.yticks(font='Times New Roman',fontsize=12)
    plt.savefig('Figures\\{}.pdf'.format(x))


# In[10]:


drawHist('计算空速')
drawHist('G值')
drawHist('姿态')
drawHist('杆量')
drawHist('坡度')
drawHist('盘量')
drawHist('下滑道偏差（C）')
drawHist('下滑道偏差（L）')
drawHist('下滑道偏差（R）')
drawHist('航向道偏差（C）')
drawHist('航向道偏差（L）')
drawHist('航向道偏差（R）')
drawHist('俯仰角率')
drawHist('下降率')
drawHist('RUDD位置')
drawHist('左油门杆位置')
drawHist('右油门杆位置')


# In[11]:


def calculateRateTh(x):
    t=dataAF[x].diff()
    mean=t.mean()
    std=t.std()
    max=mean+3*std
    min=mean-3*std
    return min,max


# In[12]:


def calculateTh(x):
    mean=dataAF[x].mean()
    std=dataAF[x].std()
    max=mean+3*std
    min=mean-3*std
    return min,max


# In[13]:


RateTh=['姿态','坡度','RUDD位置','左油门杆位置','右油门杆位置']
Th=['G值','杆量','盘量','下滑道偏差（C）','下滑道偏差（L）','下滑道偏差（R）',
    '航向道偏差（C）','航向道偏差（L）','航向道偏差（R）','俯仰角率','下降率']


# In[14]:


for i in RateTh:
    min,max=calculateRateTh(i)
    print(i,min,max)


# In[15]:


for i in Th:
    min,max=calculateTh(i)
    print(i,min,max)


# 姿态 -0.1823728785192135 0.18236945078921915
# 坡度 -0.6673918000300177 0.6673918000300177
# RUDD位置 -0.06688924609154025 0.06688922658809789
# 左油门杆位置 -0.7397604532999682 0.7397604532999682
# 右油门杆位置 -0.7364677711308804 0.7364677711308804
# 
# G值 0.9537159362101948 1.0635978346157227
# 杆量 -0.045490727710496254 0.044095951682060375
# 盘量 -0.061697333362055275 0.054986543622766275
# 下滑道偏差（C） -3.272516829802762 3.346957205632264
# 下滑道偏差（L） -3.088455218634214 3.147513401994933
# 下滑道偏差（R） -3.0607728912464105 3.1096427210309474
# 航向道偏差（C） -4.9660114773697135 4.740072025511289
# 航向道偏差（L） -4.952541392349814 4.725793916814326
# 航向道偏差（R） -4.961188798370152 4.756167466584085
# 俯仰角率 -0.269455167818758 0.2101911997652408
# 下降率 -1660.3728475810756 1660.7590236085364

# In[16]:


def zitai(x):
    if x<=-0.1823728785192135 or x>=0.18236945078921915:
        return '异常'
    else:
        return '正常'


def podu(x):
    if x<=-0.6673918000300177 or x>=0.6673918000300177:
        return '异常'
    else:
        return '正常'


def rudd(x):
    if x<=-0.06688924609154025 or x>=0.06688922658809789:
        return '异常'
    else:
        return '正常'


def zuoyoumengan(x):
    if x<=-0.7397604532999682 or x>=0.7397604532999682:
        return '异常'
    else:
        return '正常'


def youyoumengan(x):
    if x<=-0.7364677711308804 or x>=0.7364677711308804:
        return '异常'
    else:
        return '正常'


def g(x):
    if x<=0.9537159362101948:
        return '失重'
    elif x>=1.0635978346157227:
        return '超重'
    else:
        return '正常'


def gangliang(x):
    if x<=-0.045490727710496254:
        return '拉杆过度'
    elif x>=0.044095951682060375:
        return '松杆（推杆）过度'
    else:
        return '正常'


def panliang(x):
    if x<=-0.061697333362055275:
        return '左旋过度'
    elif x>=0.054986543622766275:
        return '右旋过度'
    else:
        return '正常'


def xiahuadaoC(x):
    if x<=-3.272516829802762 or x>=3.346957205632264:
        return '异常C'
    else:
        return '正常'


def xiahuadaoL(x):
    if x<=-3.088455218634214 or x>=3.147513401994933:
        return '异常L'
    else:
        return '正常'


def xiahuadaoR(x):
    if x<=-3.0607728912464105 or x>=3.1096427210309474:
        return '异常R'
    else:
        return '正常'


def hangxiangdaoC(x):
    if x<=-4.9660114773697135 or x>=4.740072025511289:
        return '异常C'
    else:
        return '正常'


def hangxiangdaoL(x):
    if x<=-4.952541392349814 or x>=4.725793916814326:
        return '异常L'
    else:
        return '正常'


def hangxiangdaoR(x):
    if x<=-4.961188798370152 or x>=4.756167466584085:
        return '异常R'
    else:
        return '正常'


def fuyangjiaolv(x):
    if x<=-0.269455167818758 or x>=0.2101911997652408:
        return '异常'
    else:
        return '正常'


def xiajianglv(x):
    if x<=-1660.3728475810756 or x>=1660.7590236085364:
        return '异常'
    else:
        return '正常'


# In[17]:


def System(data):
    data=FactorSelect(MaxFactor(data))
    data['飞行姿态变化']=data['姿态'].diff().apply(lambda x:zitai(x))
    data['坡度变化']=data['坡度'].diff().apply(lambda x:podu(x))
    data['RUDD位置变化']=data['RUDD位置'].diff().apply(lambda x:rudd(x))
    data['左油门杆位置变化']=data['左油门杆位置'].diff().apply(lambda x:zuoyoumengan(x))
    data['右油门杆位置变化']=data['右油门杆位置'].diff().apply(lambda x:youyoumengan(x))
    data['失重超重情况']=data['G值'].apply(lambda x:g(x))
    data['杆量情况']=data['杆量'].apply(lambda x:gangliang(x))
    data['盘量情况']=data['盘量'].apply(lambda x:panliang(x))
    data['下滑道偏差（C）情况']=data['下滑道偏差（C）'].apply(lambda x:xiahuadaoC(x))
    data['下滑道偏差（L）情况']=data['下滑道偏差（L）'].apply(lambda x:xiahuadaoL(x))
    data['下滑道偏差（R）情况']=data['下滑道偏差（R）'].apply(lambda x:xiahuadaoR(x))
    data['航向道偏差（C）情况']=data['航向道偏差（C）'].apply(lambda x:hangxiangdaoC(x))
    data['航向道偏差（L）情况']=data['航向道偏差（L）'].apply(lambda x:hangxiangdaoL(x))
    data['航向道偏差（R）情况']=data['航向道偏差（R）'].apply(lambda x:hangxiangdaoR(x))
    data['俯仰角率情况']=data['俯仰角率'].apply(lambda x:fuyangjiaolv(x))
    data['下降率情况']=data['下降率'].apply(lambda x:xiajianglv(x))
    return data


# In[18]:


data1V=System(data1)
data2V=System(data2)
data3V=System(data3)
data4V=System(data4)
data5V=System(data5)
data6V=System(data6)
data7V=System(data7)
data8V=System(data8)


# In[19]:


data1V


# In[20]:


data2V


# In[21]:


data3V


# In[22]:


data4V


# In[23]:


data5V


# In[24]:


data6V


# In[25]:


data7V


# In[26]:


data8V

