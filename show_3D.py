import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pandas.api.types import CategoricalDtype
from io import StringIO

# data_e = pd.read_csv('E://run//企业数据.csv',dtype=object)
# data_b = pd.read_csv('E://run//银行数据.csv',dtype=object)
comsume_M = []
product_M = []
comsume_X = []
product_X = []
comsume_NP = []
product_NP = []
base_batch = 1000
# data_e = pd.read_csv('E://run//企业数据测试_2022_06_15_20_26_44.csv', dtype=object) # 降低贷款额度
# data_e = pd.read_csv('E://run//企业数据测试_2022_06_16_00_08_05.csv', dtype=object) # 固定1
data_e = pd.read_csv('E://run//企业数据测试_2022_06_19_23_51_13.csv', dtype=object) # 银行便百分比


def getXYZ(target_X,target_Y):
    XX = []
    YY = []
    ZZ = []
    COLOR = []
    for i in range(len(target_X) // base_batch):
        color = 0.001 * i
        for j in range(i * base_batch,(i+1)*base_batch):
            ZZ.append(i)
            COLOR.append(color)
            XX.append(target_X[j])
            YY.append(target_Y[j])
    i = (len(target_X) // base_batch) + 1
    color = 0.001 * i
    for j in range(i * base_batch, len(target_X)):
        ZZ.append(i)
        COLOR.append(color)
        XX.append(target_X[j])
        YY.append(target_Y[j])
    return np.array(XX),np.array(YY),np.array(ZZ),np.array(COLOR)

M0 = np.array(data_e['消费企业现金'])
for i in M0:
    comsume_M.append(float(i))
M1 = np.array(data_e['生产企业现金'])
for i in M1:
    product_M.append(float(i))
X0 = np.array(data_e['消费企业产量'])
for i in X0:
    comsume_X.append(float(i))
X1 = np.array(data_e['生产企业产量'])
for i in X1:
    product_X.append(float(i))
NP0 = np.array(data_e['消费企业定价'])
for i in NP0:
    comsume_NP.append(float(i))
NP1 = np.array(data_e['生产企业定价'])
for i in NP1:
    product_NP.append(float(i))



#
total_M = np.array(comsume_M) + np.array(product_M)
total_X = np.array(comsume_X) + np.array(product_X)
average_NP = (np.array(comsume_NP) + np.array(product_NP))/2
#
# plt.scatter(total_M,total_X)

fig4 = plt.figure()
ax4 = plt.axes(projection='3d')
ax4.set_xlabel("M")
ax4.set_ylabel("X")
ax4.set_zlabel("episode/1000")

# xx = np.random.random(20)*10-5   #取100个随机数，范围在5~5之间
# yy = np.random.random(20)*10-5
# X, Y = np.meshgrid(xx, yy)
# Z = np.sin(np.sqrt(X**2+Y**2))
#
# #作图
# ax4.scatter(X,Y,Z,alpha=0.3,c=np.random.random(400))     #生成散点.利用c控制颜色序列,s控制大小
tx,ty,tz,tc = getXYZ(total_M,total_X)

ax4.scatter(tx,ty,tz,alpha=0.3,c=tc)

plt.show()
