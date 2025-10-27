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
bank_M = []
comsume_M_limit = []
product_M_limit = []
comsume_X_limit = []
product_X_limit = []
comsume_NP_limit = []
product_NP_limit = []
bank_M_limit = []
total_π1 = []
total_π2 = []
total_πb = []
loss1 = []
loss2 = []
lossb = []
XXX = []

path_name = 'FINALTD3三方定价600;reward收入支出利息2：1：1;二层网络128：32;__2023_02_17_06_21_02'

data_e = pd.read_csv('E://run//' + path_name + '//数据.csv', dtype=object)






def cal_corr(label1,data1,label2,data2):
    corr_list = []
    windows_num = 7000
    windows_1 = []
    windows_2 = []
    for i in range(min(windows_num,len(data1))):
        windows_1.append(data1[i])
        windows_2.append(data2[i])
    for i in range(windows_num, len(data1)):
        if i%1000 == 0:
            print(label1,label2,i,"/",len(data1))
        data = pd.DataFrame({label1: windows_1, label2: windows_2})
        # corr = data.corr(method='pearson')
        corr = data.corr(method='spearman')
        # corr = data.corr(method='kendall')

        corr_list.append(corr[label1][label2])
        windows_1[i%windows_num] = data1[i]
        windows_2[i%windows_num] = data2[i]
    return corr_list
begin = 20000
end = 1
for i in range(max(0,len(data_e)-begin),len(data_e)-end):
    comsume_M.append(float(data_e['消费企业_现金'][i]))
    product_M.append(float(data_e['生产企业_现金'][i]))
    bank_M.append(float(data_e['银行_现金'][i]))
    comsume_X.append(float(data_e['消费企业_本回合售出'][i]))
    product_X.append(float(data_e['生产企业_本回合售出'][i]))
    # comsume_X.append(float(data_e['消费企业_存货'][i]))
    # product_X.append(float(data_e['生产企业_存货'][i]))
    comsume_NP.append(float(data_e['消费企业_今日定价'][i]))
    product_NP.append(float(data_e['生产企业_今日定价'][i]))
    total_π1.append(  (total_π1[len(total_π1)-1] if len(total_π1)>0 else 0) + float(data_e['生产企业_总利润'][i]))
    total_π2.append(  (total_π2[len(total_π2)-1] if len(total_π2)>0 else 0) + float(data_e['消费企业_总利润'][i]))
    total_πb.append(  (total_πb[len(total_πb)-1] if len(total_πb)>0 else 0) + float(data_e['银行_总利润'][i]))





total_M = np.array(comsume_M) + np.array(product_M) + np.array(bank_M)
total_X = np.array(comsume_X) + np.array(product_X)
average_NP = (np.array(comsume_NP) + np.array(product_NP))/2

list_M_X = cal_corr('M',total_M,'X',total_X)
list_M_NP = cal_corr('M',total_M,'NP',average_NP)




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

plt.figure('corr_M_X')
plt.plot(np.array(list_M_X), c='r', label='1')
plt.ylabel('corr_M_X')
plt.xlabel('游戏回合')
plt.legend(loc='best')

plt.figure('corr_M_NP')
plt.plot(np.array(list_M_NP), c='r', label='1')
plt.ylabel('corr_M_NP')
plt.xlabel('游戏回合')
plt.legend(loc='best')

plt.figure('π')
plt.plot(np.array(total_π1), c='r', label='生产')
plt.plot(np.array(total_π2), c='b', label='消费')
plt.ylabel('π')
plt.xlabel('游戏天数')
plt.legend(loc='best')

plt.figure('产量')
plt.plot(np.array(comsume_X), c='r', label='消费')
plt.plot(np.array(product_X), c='b', label='生产')
plt.ylabel('X')
plt.xlabel('游戏天数')
plt.legend(loc='best')


plt.figure('πb')
plt.plot(np.array(total_πb), c='r', label='1')
plt.ylabel('π')
plt.xlabel('游戏天数')
plt.legend(loc='best')





plt.show()


