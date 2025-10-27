import numpy as np
import random
from calculate import *
from bank import bank_nnu
from enterprise import enterprise_nnu
import time
import matplotlib.pyplot as plt
import math
from pandas import DataFrame

# =====保存路径====
io_path = 'io/'
enterprise_ex_path = io_path + 'enterprise_nnu/'
enterprise_model_filename = enterprise_ex_path + 'model'
bank_ex_path = io_path + 'bank_nnu/'
bank_model_filename = bank_ex_path + 'model'
# =====保存路径====


# 银行初始数据: [array[现金资产2000+，将收款本金0，本次收款利润率0，本次收款额度0，本次收款利息0，本次坏账率0], array([]),...]
#                           0              1               2               3           4               5

# 企业初始数据: [array[现金资产500，将还款本金0，总贷款额度0，产品存量0，固定资产0，本期净利润0，
#                           0            1           2           3           4           5
#                                   本期销率0，本期定价0，本期市场销率0，本期市场定价均值0], array([])...]
#                                       6       7           8               9
# =====计算运行时间=====
start = time.clock()
# =====定义变量=====
b_num = 3;  e_num = 3  # 银行、企业个数
b = bank_nnu(b_num); e = enterprise_nnu(e_num)  # 银行、企业大脑
b_mod = list(range(1, b_num+1)); e_mod = list(range(1, e_num+1))  # 银行、企业的编号
b_new_mod = [True] * b_num; e_new_mod = [True] * e_num  # 银行、企业是否为新mod
pro_t = 7; debt_t = 6  # 生产、还钱周期
sys_sale = 0.0  # 市场销率
sys_mean_price = 0.0  # 市场定价均值
depreciation = 0.8  # 折旧率
profit = [True] * e_num  # 是否开始重新计算利润
consume = 1.2  # 一个工人的消耗
e_sale = np.zeros(e_num)  # 企业销率
in_worker = np.zeros(e_num)  # 各公司在工作的人数
e_pro = np.zeros((e_num, pro_t))  # 各公司产量
b_e_tra = np.zeros((b_num, debt_t, e_num+2))  # 用于设置银行借给所有企业的借款以及利息
b_t = np.zeros(b_num , dtype=int); e_t = np.zeros(e_num , dtype=int)  # 银行、企业存活时间

# =====初始化银行企业数据=====
# 银行的数据,数据: [array[现金资产，将收款本金，本次收款利润率，本次收款额度，本次收款利息，本次坏账率],...]
b_data = [None] * b_num
for i in range(b_num):
    money = 2000  # + random.randint(1, 200)
    b_data[i] = np.array([money, 0, 0, 0, 0, 0], dtype=float)
# 企业的数据,数据: [array[现金资产，将还款本金，将还款利息，总贷款额度，产品存量，固定资产，
#                          本期净利润，本期销率，本期定价，本期市场销率，本期市场定价均值],...]
e_data = [None] * e_num
for i in range(e_num):
    money = 1000 #+ random.randint(-300, 300)
    e_data[i] = np.array([money, 0, 0, 8, 5, 0, 0, 0, 0, 0], dtype=float)

# =====画图工具=====
# plt.ion()
# 最后100天的画图工具
data = []  # x轴，银行&企业共享x轴，表date
s_y = []  # y轴，表企业的市场销售率
e_y = [[] for i in range(e_num)]  # y轴，表企业净利润
b_y = [[] for i in range(b_num)]  # y轴，表银行利润率
# 每1000天画一个点的画图工具
live_date = []  # x轴，表1000天个数
bd_y = [[] for i in range(b_num)]  # y轴,表银行一个ep存活天数均值
br_pool = [0]*b_num  # 辅助计算br_y，计算银行累积奖励
br_y = [[] for i in range(b_num)]  # y轴，表银行奖励*折扣因子
d_y = [[] for i in range(e_num)]  # y轴,表企业一个ep存活天数均值
r_pool = [0]*e_num  # 辅助计算r_y，计算企业累积奖励
r_y = [[] for i in range(e_num)]  # y轴，表银行奖励*折扣因子
e_step = np.zeros(e_num, dtype=int)  # 企业做决策步数
b_step = np.zeros(b_num, dtype=int)  # 银行做决策步数
b_money = np.zeros(b_num)

# =====导出模型数据===
# enterprise_source = [0]*e_num
# enterprise_output = [0]*e_num
# for i in range(e_num):
#     enterprise_source[i] = enterprise_model_filename + str(i)
#     e_mod[i] -= 1
# if e.enterprise.model_import(enterprise_source,e_mod) == [None] * e_num:
#     print('No enterprise saved model is found. Start with initial values.')
# else:
#     print('A enterprise saved model is found. Start with it.')
# for i in range(e_num):
#     e_mod[i] += 1
#     enterprise_output[i] = enterprise_model_filename + str(i)
#
# bank_source = [0]*b_num
# bank_output = [0]*b_num
# for j in range(b_num):
#     bank_source[j] = bank_model_filename + str(j)
#     b_mod[j] -= 1
# if b.bank.model_import(bank_source,b_mod) == [None] * b_num:
#     print('No bank saved model is found. Start with initial values.')
# else:
#     print('A bank saved model is found. Start with it.')
# for j in range(b_num):
#     b_mod[j] += 1
#     bank_output[j] = bank_model_filename + str(j)
# =====导出模型数据===

# =====开始运行模型===
day =200000  # 模型运行天数
for t in range(day):

    # =====每1000天保存一下模型===
    # if t % 1000 == 0:
    #     for i in range(e_num):
    #         e_mod[i] -= 1
    #     e.enterprise.model_export(enterprise_output,e_mod)
    #     for i in range(e_num):
    #         e_mod[i] += 1
    #
    #     for i in range(b_num):
    #         b_mod[i] -= 1
    #     b.bank.model_export(bank_output,b_mod)
    #     for i in range(b_num):
    #         b_mod[i] += 1
    # =====每1000天保存一下模型===

    # ===企业还钱
    b_get_debt(b_data, e_data, b_t, debt_t, b_num, e_num, b_e_tra, profit, t, br_pool, b_step, b_money)
    if t > 0:
        for i in range(e_num):
            # br_pool[i] += b_data[i][0]
            if e_data[i][0] > 0:
                r_pool[i] += (e_data[i][0] - e_data[i][2])*(0.95**e_step[i])
                # br_pool[i] += b_data[i][0]
                #奖励修改点
                # r_pool[i] += e_data[i][0]*(0.95**e_step[i])
                # r_pool[i] += (e_data[i][0]+100*e_data[i][6]) * (0.95 ** e_step[i])
                e_step[i] += 1
            else:
                r_pool[i] += k[i][0]*(0.95**e_step[i])
                # r_pool[i] += k[i][0]
                # r_pool[i] += k[i][0]*(0.95**e_step[i])
                # r_pool[i] += (k[i][0]+100*k[i][6]) * (0.95 ** e_step[i])
    # ===画图
    if t % 1000 == 0 and t > 0: 
        live_date.append(len(live_date)+1)

    # 企业运营天数 & 累积奖励
    for i in range(e_num):
        if t == 0:
            d_y[i].append(0)  # 企业运营企业数
            r_y[i].append(0)  # 企业累积奖励
        if e_data[i][0] <= 0:
            d_y[i][(t-1)//1000] += 1
            r_y[i][(t-1)//1000] += r_pool[i]
            r_pool[i] = 0
            e_step[i] = 0
        if t % 1000 == 0 and t > 0:
            if d_y[i][t//1000-1]!=0: r_y[i][t // 1000 - 1] = r_y[i][t // 1000 - 1] / d_y[i][t // 1000 - 1]
            else:
                r_y[i][(t-1)//1000] += r_pool[i]
                r_pool[i] = 0
            #     e_step[i] = 0


            # paint_graph(live_date, d_y[i], i, 'Num of No. '+str(i + 1) + ' entrepreneur operating to bankrupt per 1000 loops',
            #             'Number of loops / 1000', 'Number of bankrupt enterprises')
            # paint_graph(live_date, r_y[i], i + e_num, 'The cumulative reward mean of the No. ' + str(i + 1) +
            #             ' entrepreneur\noperating to bankrupt per 1000 loops ', 'Number of loops / 1000', 'Cumulative reward')

            d_y[i].append(0)  # 企业运营企业数
            r_y[i].append(0)  # 企业累积奖励
    # 银行运营天数 & 累积奖励
    for i in range(b_num):
        if t == 0:
            bd_y[i].append(0)  # 银行运营银行数
            br_y[i].append(0)  # 银行累积奖励
        if b_data[i][0] <= 100:
            bd_y[i][(t-1)//1000] += 1
            br_y[i][(t-1)//1000] += br_pool[i]
            br_pool[i] = 0
            b_step[i] = 0
        if t % 1000 == 0 and t > 0:
            if bd_y[i][t//1000-1] != 0: br_y[i][t // 1000 - 1] = br_y[i][t // 1000 - 1] / bd_y[i][t // 1000 - 1]
            else:
                br_y[i][t // 1000 - 1] = br_y[i][t // 1000 - 2]
                # br_pool[i] = 0
                # br_pool[i] += (2000-b_data[i][0])*1000
                # b_step[i] = 0
            # paint_graph(live_date, bd_y[i], i + 2 * e_num, 'Num of No. '+str(i + 1) + 'banker operating to bankrupt per 1000 loops',
            #             'Number of loops / 1000', 'Number of bankrupt bank')
            # paint_graph(live_date, br_y[i], i + 2 * e_num + b_num, 'The cumlative reward mean of the No. '+str(i + 1)
            #             + ' banker\noperating to bankrupt per 1000 loops ', 'Number of loops / 1000', 'Cumulative reward')
            bd_y[i].append(0)  # 银行运营银行数
            br_y[i].append(0)  # 银行累积奖励
    # ===银行清算
    set_settlement(1, b_data, b_num, b_e_tra, b_new_mod, t=b_t)
    # ===企业清算，返回该loop企业破产率
    e_break = set_settlement(2, e_data, e_num, b_e_tra, e_new_mod, in_worker=in_worker, t=e_t, e_pro=e_pro)
    # ===企业将还贷款设置
    e_ret_debt(e_data, b_t, e_t, debt_t, b_num, e_num, b_e_tra, t)
    # ===收获商品
    for i in range(e_num):
        e_data[i][3] += e_pro[i, t % pro_t]; e_pro[i, t % pro_t] = 0; e_data[i][3] = round(e_data[i][3], 3)




    # ===货币市场阶段
    set_mod(b_mod, b_new_mod); state = set_b_state(b_data, e_break)  # 设置b_mod和b_state
    action = b.run_bank(b_mod, state)# b_action = array([[额度],[利息]])
    set_b_action(b_num, action, b_data)  # b_action离散化处理
    sub = list(enumerate(action[1])); random.shuffle(sub)
    sub.sort(key=lambda x: x[1]);sub = [i[0] for i in sub]  # 银行按利息排序后下标索引
    sub_s = list(range(e_num));  random.shuffle(sub_s)  # 企业随机排序后下标索引
    for i in range(b_num):
        # 奖励修改点
        br_pool[i] += b_data[i][0] * (0.95 ** b_step[i])  # 银行累积奖励画图准备
        # br_pool[i] += (b_data[i][0]+100*b_data[i][2]) * (0.95 ** b_step[i])  # 银行累积奖励画图准备
        b_step[i] += 1; b_money[i] = b_data[i][0]
        b_e_tra[i, t % debt_t, e_num] = action[1, i]  # 每家银行初始一个与所有企业的交易记录
        b_e_tra[i, t % debt_t, e_num+1] = 0
        b_e_tra[i, t % debt_t][0:e_num] = np.zeros(np.shape(b_e_tra[i, t % debt_t][0:e_num]))
    k = 0
    for i in sub:
        while action[0, i] > 0:
            e_mod[sub_s[k]], e_new_mod[sub_s[k]] = set_mod(e_mod[sub_s[k]], e_new_mod[sub_s[k]])# 设置e_mod
            state = set_e_state(1, e_data[sub_s[k]], b_interest=action[1, i])  # 设置e_state
            if e.run_enterprise(e_mod[sub_s[k]], state) == 1:
                j = 10 if action[0, i] >= 10 else action[0, i]  # j记录每次借多少钱，当贷款额度>=10时借10元，否则把剩下的额度都借出去
                b_e_tra[i, t % debt_t, sub_s[k]] += j; b_e_tra[i, t % debt_t, sub_s[k]] = round(b_e_tra[i, t % debt_t, sub_s[k]], 3)
                b_e_tra[i, t % debt_t, e_num+1] += j; b_e_tra[i, t % debt_t, e_num+1] = round(b_e_tra[i, t % debt_t, e_num+1], 3)
                b_data[i][0] -= j; b_data[i][0] = round(b_data[i][0], 3)
                action[0, i] -= j; action[0, i] = round(action[0, i], 3)
                e_data[sub_s[k]][0] += j; e_data[sub_s[k]][0] = round(e_data[sub_s[k]][0], 3)
                e_data[sub_s[k]][2] += j; e_data[sub_s[k]][2] = round(e_data[sub_s[k]][2], 3)
                if profit[sub_s[k]]: e_data[sub_s[k]][5] = 0; profit[sub_s[k]] = False  # 是否开始重新计算利润
                e_data[sub_s[k]][5] += j; e_data[sub_s[k]][5] = round(e_data[sub_s[k]][5], 3)  # 计算利润
                # 奖励修改点
                # r_pool[i] += e_data[i][0]
                # r_pool[sub_s[k]] += (e_data[sub_s[k]][0]-e_data[sub_s[k]][2])*(0.95**e_step[sub_s[k]])  # 企业累积奖励画图准备
                # r_pool[sub_s[k]] += (e_data[sub_s[k]][0]+100*e_data[sub_s[k]][6]) * (0.95 ** e_step[sub_s[k]])  # 企业累积奖励画图准备
                # e_step[sub_s[k]] += 1  # 企业累积奖励画图准备
            else:
                # r_pool[sub_s[k]] += (e_data[sub_s[k]][0]-e_data[sub_s[k]][2])*(0.95**e_step[sub_s[k]])
                # e_step[sub_s[k]] += 1
                k += 1
            if k >= e_num: break  # 所有企业都不借钱，跳出循环
        if k >= e_num: break  # 所有企业都不借钱，跳出循环
    for i in range(e_num):
        # 奖励修改点
        # r_pool[i] += e_data[i][0]
        r_pool[i] += (e_data[i][0]-e_data[i][2])*(0.95**e_step[i])
        # # r_pool[i] += (e_data[i][0]+100*e_data[i][6]) * (0.95 ** e_step[i])
        e_step[i] += 1
        if b_t[i]+1 < debt_t:
            if b_data[i][0] <= 100:
                br_pool[i] -= b_money[i]*(0.95**b_step[i])  # 银行累积奖励画图准备
                # br_pool[i] -= (b_money[i]+100*b_data[i][2]) * (0.95 ** b_step[i])  # 银行累积奖励画图准备
            else:
                br_pool[i] += b_data[i][0]*(0.95**b_step[i])  # 银行累积奖励画图准备
                # br_pool[i] += (b_data[i][0]+100*b_data[i][2]) * (0.95 ** b_step[i])  # 银行累积奖励画图准备
                b_step[i] += 1
    # 设置初始将到期收款本金、将到期还款本金
    set_debt_t(b_data, e_data, b_t, e_t, b_num, e_num, b_e_tra, debt_t, t)

    # ===商品市场阶段
    set_mod(e_mod, e_new_mod); state = set_e_state(2, e_data, e_num=e_num)
    action = e.run_enterprise(e_mod, state)  # 得到每家企业[产品定价]
    #奖励修改点
    # for i in range(e_num):
    #     r_pool[i] += e_data[i][0]
    action = [(math.log((i + 26) / 20, math.e) / 6) * j[0] for (i, j) in zip(action, e_data)]
    sys_mean_price = sum(action) / e_num; sys_mean_price = round(sys_mean_price, 3)
    for i in range(e_num):
        if profit[i]: e_data[i][5] = 0; profit[i] = False  # 全体都做了一次决策，重新开始计算利润
        e_data[i][7] = round(action[i], 3); e_data[i][9] = sys_mean_price
    sub = sorted(enumerate(action), key=lambda x: x[1]);sub = [i[0] for i in sub]  # 企业按定价排序后下标索引
    sub_s = list(range(e_num)); random.shuffle(sub_s)  # 企业随机排序后下标索引
    k = 0
    for i in sub:
        while e_data[i][3] > 0:
            e_mod[sub_s[k]], e_new_mod[sub_s[k]] = set_mod(e_mod[sub_s[k]], e_new_mod[sub_s[k]])
            state = set_e_state(3, e_data[sub_s[k]], price=e_data[i][7])
            if e_data[sub_s[k]][0] > e_data[i][7]*2 and e.run_enterprise(e_mod[sub_s[k]], state) == 1 :
                j = 2 if e_data[i][3] >= 2 else e_data[i][3]  # j记录要销售多少产品
                # # 奖励修改点
                # r_pool[i] += e_data[i][0]
                # r_pool[sub_s[k]] += (e_data[sub_s[k]][0]-e_data[sub_s[k]][2])*(0.95**e_step[sub_s[k]])  # 企业累积奖励画图准备
                # r_pool[sub_s[k]] += (e_data[sub_s[k]][0]+100*e_data[sub_s[k]][6]) * (0.95 ** e_step[sub_s[k]])  # 企业累积奖励画图准备
                # e_step[sub_s[k]] += 1
                e_data[sub_s[k]][0] -= e_data[i][7]*j; e_data[sub_s[k]][0] = round(e_data[sub_s[k]][0], 3)
                e_data[sub_s[k]][4] += j; e_data[sub_s[k]][4] = round(e_data[sub_s[k]][4], 3)  # 固定资产增加
                e_data[sub_s[k]][5] -= e_data[i][7]*j; e_data[sub_s[k]][5] = round(e_data[sub_s[k]][5], 3)  # 计算利润
                e_data[i][3] -= j; e_data[i][3] = round(e_data[i][3], 3)  # 产品存量减少
                e_data[i][0] += e_data[i][7]*j; e_data[i][0] = round(e_data[i][0], 3)
                e_data[i][5] += e_data[i][7]*j; e_data[i][5] = round(e_data[i][5], 3)  #计算利润
                e_sale[i] += j
                sys_sale += j
            else:
                # r_pool[sub_s[k]] += (e_data[sub_s[k]][0]-e_data[sub_s[k]][2])*(0.95**e_step[sub_s[k]])
                # e_step[sub_s[k]] += 1
                k += 1
            if k >= e_num: break  # 所有企业都不购买，跳出循环
        if k >= e_num: break  # 所有企业都不购买，跳出循环
    j = 0.0  # 用于下方for循环计算所有企业交易后产品存量
    for i in range(e_num):
        # 奖励修改点
        # r_pool[i] += e_data[i][0]
        r_pool[i] += (e_data[i][0]-e_data[i][2])*(0.95**e_step[i])
        # # r_pool[i] += (e_data[i][0]+100*e_data[i][6]) * (0.95 ** e_step[i])
        e_step[i] += 1
    for i in range(e_num):
        if e_sale[i]+e_data[i][3] != 0:
            e_data[i][6] = e_sale[i]/(e_sale[i]+e_data[i][3]); e_data[i][6] = round(e_data[i][6], 3)
        else: e_data[i][6] = 0
        e_sale[i] = 0
        j += e_data[i][3]
    if sys_sale+j != 0: sys_sale = sys_sale/(sys_sale+j); sys_sale = round(sys_sale, 3)  # 市场销售率
    for i in range(e_num): e_data[i][8] = sys_sale
    sys_sale = 0

    # ===劳动力市场阶段
    worker = 50000  # 劳动力固定人数
    set_mod(e_mod, e_new_mod)
    for i in range(e_num):
        # 奖励修改点
        # r_pool[i] += e_data[i][0]
        r_pool[i] += (e_data[i][0]-e_data[i][2])*(0.95**e_step[i])
        # # r_pool[i] += (e_data[i][0]+100*e_data[i][6]) * (0.95 ** e_step[i])
        e_step[i] += 1
        in_worker[i] = round(1.2 * e_data[i][4], 3)  # 公司所需员工人数
    k = e_data.copy()  # 若下一个loop新生企业，则要用旧数据
    state = set_e_state(4, e_data, e_num=e_num, in_worker=in_worker)
    action = e.run_enterprise(e_mod, state); action = [action[i]+1 for i in range(e_num)]  # 意愿工资
    sub = sorted(enumerate(action), key=lambda x: x[1])
    sub = [i[0] for i in sub]; sub = list(reversed(sub))  # 企业按意愿工资排序后下标索引
    for i in sub:
        # if e_data[i][0] <= action[i] : continue  # 修改点
        if e_data[i][0] <= action[i]*in_worker[i]:
            in_worker[i] == 0 #修改点
            continue #修改点
        if worker < in_worker[i]: in_worker[i] = worker
        worker -= in_worker[i]; worker = round(worker, 3)
        e_data[i][0] -= action[i]*in_worker[i]; e_data[i][0] = round(e_data[i][0], 3)
        e_data[i][5] -= action[i]*in_worker[i]; e_data[i][5] = round(e_data[i][5], 3)  # 计算利润
    for i in range(e_num):
        e_pro[i, t % pro_t] += 1.2*(e_data[i][4]*(1-depreciation)+in_worker[i]*consume)
        e_pro[i, t % pro_t] = round(e_pro[i, t % pro_t], 3)
        e_data[i][4] *= depreciation
        e_data[i][4] = round(e_data[i][4], 3)
    profit = [True]*e_num
    for i in range(b_num):
        b_t[i] += 1
    for i in range(e_num):
        e_data[i][0] = e_data[i][0] - e_data[i][0]*0.005 - 20  # 固定资金流失
        e_data[i][0] = round(e_data[i][0], 3)
        e_t[i] += 1

    # if(t+1) % 100 == 0:
    print('第', t, '天：')  # [现金资产，将收款本金，本次收款利润，本次本金，本次收款利息，本次坏账率]
    print('bank:[现金资产，本次收款利润]')
    for i in range(b_num):
        print([b_data[i][0], b_data[i][2]],end='\t')
        # [现金资产，将还款本金，总贷款本金，产品存量，固定资产，本期净利润，本期销率，本期定价，市场销率，市场定价均值]
    print('\nenterprise:[现金资产，将还款本金，总贷款本金，本期销率，市场销率]')
    for i in range(e_num):
        print([e_data[i][0], e_data[i][1], e_data[i][2],e_data[i][3], e_data[i][4]], e_data[i][6], e_data[i][8])
               # e_data[i][6], e_data[i][8]], e_data[i][7], e_data[i][9])
# =====最后1000天画图=====
    # 市场销售率数据图
    if t >= day - 200:
        data.append(t+1)
        # 银行利润图
        for i in range(b_num):
            b_y[i].append(b_data[i][2])
            # paint_graph(data, b_y[i], i+2*(e_num+b_num), 'Bank\'s profit margin managed by banker '+str(i+1),
            #             'Number of loops', 'Profit margin')
        # 企业净利润
        for i in range(e_num):
            e_y[i].append(e_data[i][5])
        #     paint_graph(data, e_y[i], i+2*e_num+3*b_num, 'Net profit of the enterprise managed by Entrepreneur No. '+str(i+1),
        #                 'Number of loops', 'Net profit')
        s_y.append(e_data[0][8])
        # paint_graph(data, s_y, 3*(e_num+b_num), 'Market sales rate', 'Number of loops', 'Sales rate')
# 运营天数最后一个1000天数据图
live_date.append(len(live_date)+1)
for i in range(e_num):
    if e_data[i][0] <= 0:
        d_y[i][t//1000] += 1
        r_y[i][t//1000] += r_pool[i]
    if d_y[i][t//1000] != 0: r_y[i][t//1000] = r_y[i][t//1000]/d_y[i][t//1000]
    # paint_graph(live_date, d_y[i], i, 'Num of No. ' + str(i + 1) + ' entrepreneur operating to bankrupt per 1000 loops',
    #             'Number of loops / 1000', 'Number of bankrupt enterprises')
    # paint_graph(live_date, r_y[i], i + e_num, 'The cumulative reward mean of the No. ' + str(i + 1) +
    #             ' entrepreneur\noperating to bankrupt per 1000 loops ', 'Number of loops / 1000', 'Cumulative reward')
for i in range(b_num):
    if b_data[i][0] <= 100:
        bd_y[i][t//1000] += 1
        br_y[i][t//1000] += r_pool[i]
    if bd_y[i][t//1000] != 0: br_y[i][t//1000] = br_y[i][t//1000]/bd_y[i][t//1000]
    # paint_graph(live_date, bd_y[i], i + 2 * e_num, 'Num of No. ' + str(i + 1) + 'banker operating to bankrupt per 1000 loops',
    #             'Number of loops / 1000', 'Number of bankrupt bank')
    # paint_graph(live_date, br_y[i], i + 2 * e_num + b_num, 'The cumulative reward mean of the No. ' + str(i + 1)
    #             + ' banker\noperating to bankrupt per 1000 loops ', 'Number of loops / 1000', 'Cumulative reward')
# =====关闭mod=====
b_mod = list( range(b_num))
e_mod = list( range(e_num))
b.bank_mod_close(b_mod)
e.enterprise_mod_close(e_mod)

# =====导出数据=====
标题2=[]
标题3=[]
for i in range(1,b_num+1):
    标题2.append(str(i) + '号' + '银行破产数')
    标题3.append(str(i) + '号' + '银行累积奖励')
d2 = dict(zip(标题2,bd_y))
d3 = dict(zip(标题3,br_y))
d={}
d.update(d2)
d.update(d3)

df = DataFrame(
    data = d
)
df.to_csv(
    "c://Users//liwenjian//Desktop//Cortex200812//银行数据DRQN奖励不同0916.csv",
    index = False,
    encoding = 'utf-8_sig'
)
标题4=[]
标题5=[]
for i in range(1,e_num+1):
    标题4.append(str(i) + '号' + '企业破产数')
    标题5.append(str(i) + '号' + '企业累积奖励')




d4 = dict(zip(标题4,d_y))
d5 = dict(zip(标题5,r_y))
d6={}

d6.update(d4)
d6.update(d5)
df1 = DataFrame(
    data = d6
)
df1.to_csv(
    "c://Users//liwenjian//Desktop//Cortex200812//企业数据DRQN奖励不同0916.csv",
    index = False,
    encoding = 'utf-8_sig'
)
# =====导出数据=====
# =====计算运行时间=====
end = time.clock()
print('\n运行时间：', (end-start)/60, '分钟')
# =====结束画图=====
for i in range(e_num):
    print(i+1, '企业平均破产ep数：', sum(d_y[i])/len(d_y[i]), '\t平均累积奖励：', sum(r_y[i])/len(r_y[i]),
          '\t企业平均净利润：', sum(e_y[i]) / len(e_y[i]))
print('平均销售率', sum(s_y)/len(s_y))
for i in range(b_num):
    print(i+1, '银行平均破产ep数：', sum(bd_y[i])/len(bd_y[i]), '\t平均累积奖励：', sum(br_y[i]) / len(br_y[i]),
          '\t平均利润率：', sum(b_y[i])/len(b_y[i]))
#
# plt.ioff()
# plt.show()
