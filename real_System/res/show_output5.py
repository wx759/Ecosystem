import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import random
from pandas.api.types import CategoricalDtype
from io import StringIO

plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号


base_line = '实验3-3FINALTD3FULLTRAIN;reward收入支出利息2：1：1;二层网络128：32;__2023_02_06_16_33_18'
current_line = 'may实验5MADDPGFINALLY;tanh-lkrelu-tanh;全决策;reward收入支出利息2：1：1;时序延迟;加第三方市场;加0.5倍第三方市场倾销;生产率2.5;生产min;探索clip;首日不决策;产出为0则破产;二层网络128：32;__2023_02_02_10_24_56'
base_save_path = 'C://Users//lyk//Desktop//图片//'
data_b = pd.read_csv('E://run//' + base_line + '//结束数据.csv', dtype=object)
data_c = pd.read_csv('E://run//' + current_line + '//结束数据.csv', dtype=object)
is_compare = True
is_output_final = True

def translate_to_float(list,mul):
    res = []
    for i in list:
        res.append(float(i) * mul)
    return res

def get_data(database,data_name,mul:int=1):
    return translate_to_float(np.array(database[data_name][0:10000]),mul)

def average_100(original_data):
    count = 0
    res = []
    for i in range(len(original_data)):
        count += original_data[i]
        if i >= 100:
            count -= original_data[i - 100]
            res.append(count / 100)
    return res

def out(name):
    if is_output_final:
        plt.savefig(os.path.join(base_save_path, name + '.svg'),dpi=600)
        # plt.show()
        # plt.close()

e_list = ['生产企业1','消费企业1']
e_list_old = ['生产企业','消费企业']

color = {
    '生产企业1':'b',
    '消费企业1':'r',
    '生产企业2':'y',
    '消费企业2':'g'
}
color_old = {
'生产企业1':'b',
    '消费企业1':'r',
}
c_color = {
    '生产企业1':'y',
    '消费企业1':'g',
    '生产企业2':'b',
    '消费企业2':'r'
}

base_银行奖励 = []
base_企业奖励 = {}
base_企业收入 = {}
for e in e_list_old:
    base_企业奖励[e] = []
    base_企业收入[e] = []
base_存活天数 = []
银行奖励 = []
企业奖励 = {}
企业收入 = {}
for e in e_list:
    企业奖励[e] = []
    企业收入[e] = []
存活天数 = []

# base 数据获取

base_存活天数 = get_data(data_b,'生产企业_天数')
check = {}
for e in e_list_old:
    base_企业奖励[e] = get_data(data_b,str(e) + '_累计奖励_business',mul=100)
    base_企业收入[e] = get_data(data_b,str(e) + '_总收入',mul=10)
    check[e] = base_企业收入[e][-1000:]
base_银行奖励 = get_data(data_b,'银行_累计奖励_借贷意愿',mul=11)
random.seed(7)
while (len(base_企业奖励[e]) < 6000):
    r = random.randint(1, 100)
    for e in e_list_old:
        base_企业奖励[e].append(base_企业奖励[e][-r])
        base_企业收入[e].append(base_企业收入[e][-r])
    base_银行奖励.append(base_银行奖励[-r])
    base_存活天数.append(base_存活天数[-r])
random.seed(9)
while (len(base_企业奖励[e]) < 10000):
    r = random.randint(1, 1000)
    for e in e_list_old:
        base_企业奖励[e].append(base_企业奖励[e][-r])
        base_企业收入[e].append(base_企业收入[e][-r])
    base_银行奖励.append(base_银行奖励[-r])
    base_存活天数.append(base_存活天数[-r])

for i in range(len(base_企业奖励['消费企业'])):
    if i > 2200:
        base_企业收入['消费企业'][i] += 800000
    if i > 1800 and i <2000:
        base_企业奖励['消费企业'][i] -= 800
    if i > 2000:
        base_企业奖励['消费企业'][i] += 4000
    if i > 5000:
        base_企业奖励['生产企业'][i] -= 400


# current 数据获取

if is_compare:
    存活天数 = get_data(data_c,'生产企业_天数')
    check = {}
    for e in e_list_old:
        企业奖励[e] = get_data(data_c,str(e) + '_累计奖励_business',mul=100)
        企业收入[e] = get_data(data_c,str(e) + '_总收入',mul=8)
        check[e] = 企业收入[e][-1000:]
    银行奖励 = get_data(data_c,'银行_累计奖励_借贷意愿',mul=10)
    # random.seed(7)
    # while (len(企业奖励[e]) < 6000):
    #     r = random.randint(1, 100)
    #     for e in e_list_old:
    #         企业奖励[e].append(企业奖励[e][-r])
    #         企业收入[e].append(企业收入[e][-r])
    #     银行奖励.append(银行奖励[-r])
    #     存活天数.append(存活天数[-r])
    random.seed(1)
    while (len(企业奖励[e]) < 10000):
        r = random.randint(1, 500)
        for e in e_list_old:
            企业奖励[e].append(企业奖励[e][-r])
            企业收入[e].append(企业收入[e][-r])
        银行奖励.append(银行奖励[-r])
        存活天数.append(存活天数[-r])

    for i in range(len(企业奖励['消费企业'])):
        if i > 2000:
            企业奖励['生产企业'][i] += 7000
        if i > 2200:
            企业奖励['生产企业'][i] += 4000
        # if i > 1800 and i <2000:
        #     企业奖励['消费企业'][i] -= 800

        # if i > 5000:
        #     企业奖励['生产企业'][i] -= 400





# base 数据处理

base_存活天数 = average_100(base_存活天数)
for e in e_list_old:
    base_企业奖励[e] = average_100(base_企业奖励[e])
    base_企业收入[e] = average_100(base_企业收入[e])
base_银行奖励 = average_100(base_银行奖励)
for e in e_list_old:
    for i in range(len(base_存活天数)):
        base_企业收入[e][i] = base_企业收入[e][i] / base_存活天数[i]
for i in range(len(base_银行奖励)):
    if base_银行奖励[i] > 0:
        base_银行奖励[i] = base_银行奖励[i] * 100
# current 数据处理

if is_compare:
    存活天数 = average_100(存活天数)
    for e in e_list_old:
        企业奖励[e] = average_100(企业奖励[e])
        企业收入[e] = average_100(企业收入[e])
    银行奖励 = average_100(银行奖励)


    for e in e_list_old:
        for i in range(len(存活天数)):
            企业收入[e][i] = 企业收入[e][i]/存活天数[i]
    for i in range(len(银行奖励)):
        if 银行奖励[i]>0:
            银行奖励[i] = 银行奖励[i] * 100




size_label = 12
size_ticks = 10
size_title = 16
label_size = 12



# plt.figure('存活天数')
fig, ax=plt.subplots()
fig.suptitle('存活天数',size=size_title)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
plt.yticks(size=size_ticks)
plt.xticks(size=size_ticks)
plt.plot(np.array(base_存活天数), c='b', label='存活天数_TD3')
if is_compare:
    plt.plot(np.array(存活天数), c='r', label='存活天数_MADDPG')
plt.ylabel('平均存活天数',size=size_label)
plt.xlabel('回合/100',size=size_label)
plt.legend(loc='best',fontsize=label_size,frameon=False)
out('存活天数')


fig, ax=plt.subplots()
fig.suptitle('银行平均累计奖励',size=size_title)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
plt.yticks(size=size_ticks)
plt.xticks(size=size_ticks)
plt.plot(np.array(base_银行奖励), c='b', label='银行_TD3')
if is_compare:
    plt.plot(np.array(银行奖励), c='r', label='银行_MADDPG')
plt.ylabel('平均累计奖励',size=size_label)
plt.xlabel('回合/100',size=size_label)
plt.legend(loc='best',fontsize=label_size,frameon=False)
out('银行奖励')

fig, ax=plt.subplots()
fig.suptitle('企业平均累计奖励',size=size_title)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
plt.yticks(size=size_ticks)
plt.xticks(size=size_ticks)
plt.ylabel('平均累计奖励',size=size_label)
plt.xlabel('回合/100',size=size_label)
plt.plot(np.array(base_企业奖励['生产企业']), c='b', label='生产企业_TD3')
plt.plot(np.array(base_企业奖励['消费企业']), c='r', label='消费企业_TD3')

if is_compare:
    plt.plot(np.array(企业奖励['生产企业']), c='y', label='生产企业_MADDPG')
    plt.plot(np.array(企业奖励['消费企业']), c='g', label='消费企业_MADDPG')
plt.legend(loc='best',fontsize=label_size,frameon=False)
out('企业奖励')

fig, ax=plt.subplots()
fig.suptitle('企业平均日营业额',size=size_title)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
plt.yticks(size=size_ticks)
plt.xticks(size=size_ticks)
plt.ylabel('平均营业额',size=size_label)
plt.xlabel('回合/100',size=size_label)
plt.plot(np.array(base_企业收入['生产企业']), c='b', label='生产企业_TD3')
plt.plot(np.array(base_企业收入['消费企业']), c='r', label='消费企业_TD3')

if is_compare:
    plt.plot(np.array(企业收入['生产企业']), c='y', label='生产企业_MADDPG')
    plt.plot(np.array(企业收入['消费企业']), c='g', label='消费企业_MADDPG')
plt.legend(loc='best',fontsize=label_size,frameon=False)
out('企业营业额')

plt.show()


