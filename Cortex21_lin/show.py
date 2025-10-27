import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pandas.api.types import CategoricalDtype
from io import StringIO

data_e = pd.read_csv('E://run//企业数据.csv',dtype=object)
data_b = pd.read_csv('E://run//银行数据.csv',dtype=object)
print(data_e)
print(data_b)
e_num=3
b_num=3
e_r = [[] for i in range(e_num)]
e_d = [[] for i in range(e_num)]
b_r = [[] for i in range(b_num)]
b_d = [[] for i in range(b_num)]
for i in range(e_num):
    for j in np.array(data_e[str(i+1)+'号企业累积奖励']):
        e_r[i].append(float(j))
for i in range(e_num):
    for j in np.array(data_e[str(i+1)+'号企业破产数']):
        e_d[i].append(float(j))

for i in range(b_num):
    for j in np.array(data_b[str(i+1)+'号银行累积奖励']):
        b_r[i].append(float(j))
for i in range(b_num):
    for j in np.array(data_b[str(i+1)+'号银行破产数']):
        b_d[i].append(float(j))

plt.figure('企业累积奖励')
e_r_ave,e_d_ave,b_r_ave,b_d_ave=[],[],[],[]
for i in range(len(e_r[0])):
    e_r_ave.append((e_r[0][i]+e_r[1][i]+e_r[2][i])/3)
for i in range(len(e_d[0])):
    e_d_ave.append((e_d[0][i]+e_d[1][i]+e_d[2][i])/3)
for i in range(len(b_r[0])):
    b_r_ave.append((b_r[0][i]+b_r[1][i]+b_r[2][i])/3)
for i in range(len(b_d[0])):
    b_d_ave.append((b_d[0][i]+b_d[1][i]+b_d[2][i])/3)
plt.plot(e_r_ave, c='b', label='平均')
# plt.plot(e_r[0], c='b', label='1号企业')
# plt.plot(e_r[1], c='y', label='2号企业')
# plt.plot(e_r[2], c='r', label='3号企业')

#
plt.figure('企业破产数')
plt.plot(e_d_ave, c='b', label='平均')
# plt.plot(e_d[0], c='b', label='1号企业')
# plt.plot(e_d[1], c='y', label='2号企业')
# plt.plot(e_d[2], c='r', label='3号企业')

plt.figure('银行累积奖励')
plt.plot(b_r_ave, c='b', label='平均')

# plt.plot(b_r[0], c='b', label='1号银行')
# plt.plot(b_r[1], c='y', label='2号银行')
# plt.plot(b_r[2], c='r', label='3号银行')

#
plt.figure('银行破产数')
plt.plot(b_d_ave, c='b', label='平均')
# plt.plot(b_d[0], c='b', label='1号银行')
# plt.plot(b_d[1], c='y', label='2号银行')
# plt.plot(b_d[2], c='r', label='3号银行')

plt.show()
