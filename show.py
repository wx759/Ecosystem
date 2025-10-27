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
# data_e = pd.read_csv('E://run//企业数据测试_2022_06_15_22_07_15.csv', dtype=object)
data_e = pd.read_csv('E://run//FINALTD3;reward收入支出利息2：1：1;二层网络128：32;__2023_02_03_08_03_14//数据.csv', dtype=object) # 银行便百分比

M0 = np.array(data_e['消费企业现金'])[-10000:-1]
for i in M0:
    comsume_M.append(float(i))
M1 = np.array(data_e['生产企业现金'])[-10000:-1]
for i in M1:
    product_M.append(float(i))
X0 = np.array(data_e['消费企业产量'])[-10000:-1]
for i in X0:
    comsume_X.append(float(i))
X1 = np.array(data_e['生产企业产量'])[-10000:-1]
for i in X1:
    product_X.append(float(i))
NP0 = np.array(data_e['消费企业定价'])[-10000:-1]
for i in NP0:
    comsume_NP.append(float(i))
NP1 = np.array(data_e['生产企业定价'])[-10000:-1]
for i in NP1:
    product_NP.append(float(i))

total_M = np.array(comsume_M) + np.array(product_M)
total_X = np.array(comsume_X) + np.array(product_X)
average_NP = (np.array(comsume_NP) + np.array(product_NP))/2

plt.figure('M_X')
plt.title('M_X')
plt.scatter(total_M,total_X)
plt.xlabel('M')
plt.ylabel('X')

plt.figure('M_NP')
plt.title('M_NP')
plt.scatter(total_M,average_NP)
plt.xlabel('M')
plt.ylabel('NP')


plt.show()
