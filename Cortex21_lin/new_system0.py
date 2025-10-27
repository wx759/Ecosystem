import numpy as np
import random
from new_bank import bank_nnu
from new_bank import bank_nnu_without_intelligent as sb_bank_nnu
from new_enterprise import enterprise_nnu
from new_enterprise import enterprise_nnu_without_intelligent as sb_enterprise_nnu
import time
from new_calculate import *
import matplotlib.pyplot as plt
import math
from pandas import DataFrame

# =====保存路径=====
io_path = 'io/'
enterprise_ex_path = io_path + 'enterprise_nnu/'
enterprise_model_filename = enterprise_ex_path + 'model'
bank_ex_path = io_path + 'bank_nnu/'
bank_model_filename = bank_ex_path + 'model'
# =====保存路径=====

# 银行初始数据: [array[现金资产2000+，将收贷款本金0，本次收款利润率0，本次应收贷款本金0，本次应收贷款利息0，本次坏账率0], array([]),...]
#                           0       1               2               3           4               5
# 企业初始数据: [array[现金资产1000，将还贷款本金0，总贷款本金0，产品存量0，固定资产0，本期净利润0，本期销售率0，本期产品定价0，本期市场销售率0，本期市场产品定价均值0],
#                          0          1          2          3        4        5           6       7               8               9
#              array([])...]


# =====计算运行时间=====
start = time.clock()
# =====定义变量=====
b_num = 1
e_product_num = 1
e_consume_num = 1
e_num = e_product_num + e_consume_num # 银行、一类企业（生成生产资料）、二类企业（生成生活资料）个数
b_agent = bank_nnu(b_num)
e_agent = enterprise_nnu(e_num)
b_agent_sb = sb_bank_nnu(b_num, "银行")
e_product_agent_sb = sb_enterprise_nnu(e_product_num, "生产机器的企业")
e_consume_agent_sb = sb_enterprise_nnu(e_consume_num, "生产粮食的企业")
b_mod = list(range(0, b_num))
e_product_mod = list(range(0, e_product_num))
e_consume_mod = list(range(e_product_num, e_num)) # 银行、企业的编号,对应数组下标
e_mod = list(range(0, e_num))
debt_T = 3  # 还钱周期
debt_record = [[0]*debt_T] * e_num
sys_sale = 0.0  # 市场销率
sys_mean_price = 0.0  # 市场定价均值
depreciation = 0.8  # 折旧率
f_product = 1  #销售额系数
f_consume = 1
Debt_i = 0.001  #利率

# profit = [True] * e_num  # 是否开始重新计算利润

b_data = [None] * b_num
e_data = [None] * e_num



episode = 10000  # 回合数
day = 1000  # 结束日

# =====画图工具=====
# plt.ion()
# 最后100天的画图工具
e_y = [[] for i in range(e_num)]  # y轴，表企业净利润
b_y = [[] for i in range(b_num)]
live_day = []                                             # 记录存活天数
K_X = []
L_X = []
K_desired = [[] for i in range(e_num)]
L_desired = [[] for i in range(e_num)]
total_R = [[] for i in range(e_num)]
total_C = [[] for i in range(e_num)]
total_iD = [[] for i in range(e_num)]
final_D = [[] for i in range(e_num)]
final_M = [[] for i in range(e_num)]
WNDF1 = []
WNDF2 = []
K_NP = []
L_NP = []
# =====开始运行模型=====


for t in range(episode):

    # =====初始化银行企业数据=====
    # 企业的数据,数据: [array[现金M，存货X，负债D,收入R,将还款利息iD,成本C,利润π,市场价P,追加贷款意愿WNDF,机器K,劳动力L,实际机器getK,实际劳动力getL,次日定价NP],...]
    for i in e_product_mod:
        data = []
        for j in range(len(E_DATA)):
            data.append(0)
        data[E_DATA.M.value] = 0.0  # + random.randint(-300, 300)
        # data[E_DATA.WNDF.value] = 100.0
        data[E_DATA.X.value] = 10
        e_data[i] = np.array(data)
    for i in e_consume_mod:
        data = []
        for j in range(len(E_DATA)):
            data.append(0)
        data[E_DATA.M.value] = 0.0  # + random.randint(-300, 300)
        # data[E_DATA.WNDF.value] = 100.0
        data[E_DATA.X.value] = 10
        e_data[i] = np.array(data)
    # 银行的数据,数据: [array[现金，企业债券Ω,欠企业的钱D，贷款意向WNDB，企业观察OB,...]
    for i in b_mod:
        base_money = 0.0  # + random.randint(1, 200)
        b_data[i] = np.array(
            [base_money, np.array([0.0] * e_num), np.array([0.0] * e_num), np.array([0.0] * e_num), np.array(e_data)])
    # 初始化债务表
    debt_record = [[0] * debt_T, [0] * debt_T]
    e_temp_mod = [-x for x in range(1,e_num+1)]
    b_temp_mod = [-1] * b_num
    print("第"+str(t)+"回合开始")

    for d in range(day):
        # =================将前一天的成本作为当天的负利润=================
        # 如果将每天当天的成本与收入计入利润，两家企业一定是零和，扣除利息后两家企业利润之和必为负数
        # 任何时刻，企业利润总和＝这个时刻银行新增的贷款总量，所以计入利润时需要时间差
        e_state = [None] * e_num
        e_state = set_e_state(e_data, e_mod)
        set_init(e_data)
# =================清算还债=================
        if set_debt(b_data[0], [e_product_mod, e_consume_mod], e_data, d, debt_T, debt_record, Debt_i) or d == day-1 or StoreOver(e_data):
            print("第"+str(t)+"回合结束")
            live_day.append(d)
            for i in e_product_mod:
                e_y[i].append(e_data[i][E_DATA.total_π.value])
                total_R[i].append(e_data[i][E_DATA.total_R.value])
                total_C[i].append(e_data[i][E_DATA.total_C.value])
                total_iD[i].append(e_data[i][E_DATA.total_iD.value])
                final_D[i].append(e_data[i][E_DATA.D.value])
                final_M[i].append(e_data[i][E_DATA.M.value])
            for i in e_consume_mod:
                e_y[i].append(e_data[i][E_DATA.total_π.value])
                total_R[i].append(e_data[i][E_DATA.total_R.value])
                total_C[i].append(e_data[i][E_DATA.total_C.value])
                total_iD[i].append(e_data[i][E_DATA.total_iD.value])
                final_D[i].append(e_data[i][E_DATA.D.value])
                final_M[i].append(e_data[i][E_DATA.M.value])
            for i in b_mod:
                b_y[i].append(b_data[i][B_DATA.M.value])


            break  # 有人破产清算 回合结束

# =====各部门根据前一天数据决策当日行动=====
#         show_enterprise(e_data[0], "第一类企业")
#         show_enterprise(e_data[1], "第二类企业")
# =================银行===================
        b_state = [None] * 1
        b_state[0] = np.array(set_b_state(b_data[0]))  # 根据前日结算重新整理b的数据
        # b_action = [b_agent.run_bank(b_temp_mod, b_state, 0), b_agent.run_bank(b_temp_mod, b_state, 1)]
        # b_action = b_agent.run_bank(b_mod, b_state)  # action应为[ WNDB1,WNDB2 ]
        b_action = b_agent_sb.run_bank(b_mod, b_state, d)  # 手动输入
# =================企业===================
        e_action = [e_agent.run_enterprise(e_temp_mod, set_e_action_type(e_state, 0), d), e_agent.run_enterprise(e_temp_mod, set_e_action_type(e_state, 1), d),
                    e_agent.run_enterprise(e_temp_mod, set_e_action_type(e_state, 2), d), e_agent.run_enterprise(e_temp_mod, set_e_action_type(e_state, 3), d, is_train=True)]

# =================对决策动作数据进行处理并赋值data===================
        set_b_action(b_data[0], b_action)
        set_e_action(e_data, e_product_mod, e_action, e_action[3][e_consume_mod[0]])
        set_e_action(e_data, e_consume_mod, e_action, e_action[3][e_product_mod[0]])

# =====企业向银行贷款===== 此时银行现金不会减少
        rent_money(b_data[0], [e_product_mod, e_consume_mod], e_data, d, debt_T, debt_record,)


# =====企业开始交易=====
# =====开始交易机器===== product[0]作为卖家，pro[0]和con[0] 共同作为买家 商品为机器（K) 结算在get(K)上
        trade(b_data[0], e_product_mod[0], [e_product_mod[0], e_consume_mod[0]], e_data, E_DATA.K, E_DATA.getK)
# =====开始交易工人===== consume[0]作为卖家，pro[0]和con[0] 共同作为买家 商品为工人（L) 结算在get(L)上
        trade(b_data[0], e_consume_mod[0], [e_product_mod[0], e_consume_mod[0]], e_data, E_DATA.L, E_DATA.getL)

# =====企业进行生产=====
        product(e_data, [e_product_mod, e_consume_mod])

# =====当日数据进行结算=====
        daily_settlement(b_data[0], e_data, t, d)
        WNDF1.append(e_data[0][E_DATA.WNDF.value])
        WNDF2.append(e_data[1][E_DATA.WNDF.value])
        K_X.append(e_data[0][E_DATA.X.value])
        L_X.append(e_data[1][E_DATA.X.value])
        K_desired[0].append(e_data[0][E_DATA.K.value])
        K_desired[1].append(e_data[1][E_DATA.K.value])
        L_desired[0].append(e_data[0][E_DATA.L.value])
        L_desired[1].append(e_data[1][E_DATA.L.value])
        K_NP.append(e_data[0][E_DATA.NP.value])
        L_NP.append(e_data[1][E_DATA.NP.value])
        print(debt_record)

# =====关闭模型=====

b_mod = list(range(b_num))
e_product_mod = list(range(e_product_num))
e_consume_mod = list(range(e_consume_num))

# b_agent.bank_mod_close(b_mod)
# e_product_agent.enterprise_mod_close(e_product_mod)
# e_consume_agent.enterprise_mod_close(e_consume_mod)

# # =====导出数据=====
# # b_path = 'E://run//' + 'DQN银行20w eps_0.95_8w_0.02 lr=0.008 (本金-贷款)无折扣.csv'
# # e_path = 'E://run//' + 'DQN企业20w eps_0.95_8w_0.02 lr=0.008 (本金-贷款)无折扣.csv'
# b_path = 'E://run//' + 'DRQN银行20w eps=0.02 lr=0.008 (本金-贷款)无折扣.csv'
# e_path = 'E://run//' + 'DRQN企业20w eps=0.02 lr=0.008 (本金-贷款)无折扣.csv'
# # b_path = 'E://run//' + 'NN-DQN银行20w eps=0.02 lr=0.008 (本金-贷款)无折扣.csv'
# # e_path = 'E://run//' + 'NN-DQN企业20w eps=0.02 lr=0.008 (本金-贷款)无折扣.csv'
# 标题2 = []
# 标题3 = []
# for i in range(1, b_num + 1):
#     标题2.append(str(i) + '号' + '银行破产数')
#     标题3.append(str(i) + '号' + '银行累积奖励')
# d2 = dict(zip(标题2, bd_y))
# d3 = dict(zip(标题3, br_y))
# d = {}
# d.update(d2)
# d.update(d3)
# df = DataFrame(
#     data=d
# )
# df.to_csv(
#     b_path,
#     index=False,
#     encoding='utf-8_sig'
# )
#
# 标题4 = []
# 标题5 = []
# for i in range(1, e_num + 1):
#     标题4.append(str(i) + '号' + '企业破产数')
#     标题5.append(str(i) + '号' + '企业累积奖励')
# d4 = dict(zip(标题4, d_y))
# d5 = dict(zip(标题5, r_y))
# d6 = {}
# d6.update(d4)
# d6.update(d5)
# df1 = DataFrame(
#     data=d6
# )
# df1.to_csv(
#     e_path,
#     index=False,
#     encoding='utf-8_sig'
# )
#
#
# # =====导出数据=====
# # =====计算运行时间=====
# end = time.clock()
# print('\n运行时间：', (end - start) / 60, '分钟')
# # =====结束画图=====
# for i in range(e_num):
#     print(i + 1, '企业平均破产ep数：', sum(d_y[i]) / len(d_y[i]), '\t平均累积奖励：', sum(r_y[i]) / len(r_y[i]),
#           '\t企业平均净利润：', sum(e_y[i]) / len(e_y[i]))
# print('平均销售率', sum(s_y) / len(s_y))
# for i in range(b_num):
#     print(i + 1, '银行平均破产ep数：', sum(bd_y[i]) / len(bd_y[i]), '\t平均累积奖励：', sum(br_y[i]) / len(br_y[i]),
#           '\t平均利润率：', sum(b_y[i]) / len(b_y[i]))
# # plt.ioff()
# # plt.show()

end = time.clock()
print('\n运行时间：', (end - start) / 60, '分钟')
print("均值 中位数 方差 标准差 最大值 最小值")
e_r_res=[[] for i in range(e_num)]


for i in range(e_num):
    e_r_res[i].append(e_y[i][0])
    for j in range(1, len(e_y[i])):
        e_r_res[i].append(e_r_res[i][j-1]+e_y[i][j])

print("WNDF1", set_data_round([np.mean(WNDF1), np.median(WNDF1), np.var(WNDF1), np.std(WNDF1), max(WNDF1), min(WNDF1)]))
print("WNDF2", set_data_round([np.mean(WNDF2), np.median(WNDF2), np.var(WNDF2), np.std(WNDF2), max(WNDF2), min(WNDF2)]))
print("总利润1", set_data_round([np.mean(e_y[0]), np.median(e_y[0]), np.var(e_y[0]), np.std(e_y[0]), max(e_y[0]), min(e_y[0])]))
print("总利润2", set_data_round([np.mean(e_y[1]), np.median(e_y[1]), np.var(e_y[1]), np.std(e_y[1]), max(e_y[1]), min(e_y[1])]))
print("游戏结束时余额1", set_data_round([np.mean(final_M[0]), np.median(final_M[0]), np.var(final_M[0]), np.std(final_M[0]), max(final_M[0]), min(final_M[0])]))
print("游戏结束时余额2", set_data_round([np.mean(final_M[1]), np.median(final_M[1]), np.var(final_M[1]), np.std(final_M[1]), max(final_M[1]), min(final_M[1])]))
print("游戏结束时支付总利息1", set_data_round([np.mean(total_iD[0]), np.median(total_iD[0]), np.var(total_iD[0]), np.std(total_iD[0]), max(total_iD[0]), min(total_iD[0])]))
print("游戏结束时支付总利息2", set_data_round([np.mean(total_iD[1]), np.median(total_iD[1]), np.var(total_iD[1]), np.std(total_iD[1]), max(total_iD[1]), min(total_iD[1])]))
print("银行利润", set_data_round([np.mean(b_y[0]), np.median(b_y[0]), np.var(b_y[0]), np.std(b_y[0]), max(b_y[0]), min(b_y[0])]))
print("结束天数", set_data_round([np.mean(live_day), np.median(live_day), np.var(live_day), np.std(live_day), max(live_day), min(live_day)]))
print("机器存货", set_data_round([np.mean(K_X), np.median(K_X), np.var(K_X), np.std(K_X), max(K_X), min(K_X)]))
print("粮食存货", set_data_round([np.mean(L_X), np.median(L_X), np.var(L_X), np.std(L_X), max(L_X), min(L_X)]))
print("机器定价", set_data_round([np.mean(K_NP), np.median(K_NP), np.var(K_NP), np.std(K_NP), max(K_NP), min(K_NP)]))
print("粮食定价", set_data_round([np.mean(L_NP), np.median(L_NP), np.var(L_NP), np.std(L_NP), max(L_NP), min(L_NP)]))
print("K1决策", set_data_round([np.mean(K_desired[0]), np.median(K_desired[0]), np.var(K_desired[0]), np.std(K_desired[0]), max(K_desired[0]), min(K_desired[0])]))
print("K2决策", set_data_round([np.mean(K_desired[1]), np.median(K_desired[1]), np.var(K_desired[1]), np.std(K_desired[1]), max(K_desired[1]), min(K_desired[1])]))
print("L1决策", set_data_round([np.mean(L_desired[0]), np.median(L_desired[0]), np.var(L_desired[0]), np.std(L_desired[0]), max(L_desired[0]), min(L_desired[0])]))
print("L2决策", set_data_round([np.mean(L_desired[1]), np.median(L_desired[1]), np.var(L_desired[1]), np.std(L_desired[1]), max(L_desired[1]), min(L_desired[1])]))


plt.figure('企业单回合总利润')
plt.plot(np.array(e_y[0]), c='r', label='1')
plt.plot(np.array(e_y[1]), c='b', label='2')
plt.ylabel('累积奖励')
plt.xlabel('游戏回合')
plt.legend(loc='best')

plt.figure('企业单回合总利润之和')
plt.plot(np.array(e_r_res[0]), c='r', label='1')
plt.plot(np.array(e_r_res[1]), c='b', label='2')
# plt.plot(np.array(e_r_res[0]+e_r_res[1]), c='y', label='3')
plt.ylabel('累积奖励')
plt.xlabel('游戏回合')
plt.legend(loc='best')

plt.figure('游戏结束天数')
plt.plot(np.array(live_day), c='r', label='1')
plt.ylabel('天数')
plt.xlabel('游戏回合')
plt.legend(loc='best')

plt.figure('银行利润')
plt.plot(np.array(b_y[0]), c='r', label='1')
plt.ylabel('累积奖励')
plt.xlabel('游戏回合')
plt.legend(loc='best')

plt.figure('游戏结束时机器存货')
plt.plot(np.array(K_X), c='r', label='1')
plt.ylabel('累积奖励')
plt.xlabel('游戏回合')
plt.legend(loc='best')

plt.figure('游戏结束时粮食存货')
plt.plot(np.array(L_X), c='r', label='1')
plt.ylabel('累积奖励')
plt.xlabel('游戏回合')
plt.legend(loc='best')

plt.figure('K1')
plt.plot(np.array(K_desired[0]), c='r', label='1')
plt.ylabel('累积奖励')
plt.xlabel('游戏回合')
plt.legend(loc='best')

plt.figure('K2')
plt.plot(np.array(K_desired[1]), c='r', label='1')
plt.ylabel('累积奖励')
plt.xlabel('游戏回合')
plt.legend(loc='best')

plt.figure('L1')
plt.plot(np.array(L_desired[0]), c='r', label='1')
plt.ylabel('累积奖励')
plt.xlabel('游戏回合')
plt.legend(loc='best')

plt.figure('L2')
plt.plot(np.array(L_desired[1]), c='r', label='1')
plt.ylabel('累积奖励')
plt.xlabel('游戏回合')
plt.legend(loc='best')

plt.figure('WNDF1')
plt.plot(np.array(WNDF1), c='r', label='1')
plt.ylabel('累积奖励')
plt.xlabel('游戏回合')
plt.legend(loc='best')

plt.figure('WNDF2')
plt.plot(np.array(WNDF2), c='r', label='1')
plt.ylabel('累积奖励')
plt.xlabel('游戏回合')
plt.legend(loc='best')

plt.figure('游戏结束时余额')
plt.plot(np.array(final_M[0]), c='r', label='1')
plt.plot(np.array(final_M[1]), c='b', label='2')
plt.ylabel('累积奖励')
plt.xlabel('游戏回合')
plt.legend(loc='best')

plt.figure('游戏结束时欠款')
plt.plot(np.array(final_D[0]), c='r', label='1')
plt.plot(np.array(final_D[1]), c='b', label='2')
plt.ylabel('累积奖励')
plt.xlabel('游戏回合')
plt.legend(loc='best')

plt.figure('游戏结束时支付总利息')
plt.plot(np.array(total_iD[0]), c='r', label='1')
plt.plot(np.array(total_iD[1]), c='b', label='2')
plt.ylabel('累积奖励')
plt.xlabel('游戏回合')
plt.legend(loc='best')

plt.figure('游戏结束时总收入')
plt.plot(np.array(total_R[0]), c='r', label='1')
plt.plot(np.array(total_R[1]), c='b', label='2')
plt.ylabel('累积奖励')
plt.xlabel('游戏回合')
plt.legend(loc='best')

plt.figure('游戏结束时总支出')
plt.plot(np.array(total_C[0]), c='r', label='1')
plt.plot(np.array(total_C[1]), c='b', label='2')
plt.ylabel('累积奖励')
plt.xlabel('游戏回合')
plt.legend(loc='best')

plt.show()
