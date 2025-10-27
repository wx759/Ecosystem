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
e_product_agent = enterprise_nnu(e_product_num)
e_consume_agent = enterprise_nnu(e_consume_num)  # 银行、企业大脑
b_agent_sb = sb_bank_nnu(b_num, "银行")
e_product_agent_sb = sb_enterprise_nnu(e_product_num, "生产机器的企业")
e_consume_agent_sb = sb_enterprise_nnu(e_consume_num, "生产粮食的企业")
b_mod = list(range(0, b_num))
e_product_mod = list(range(0, e_product_num))
e_consume_mod = list(range(e_product_num, e_num)) # 银行、企业的编号,对应数组下标
debt_T = 3  # 还钱周期
debt_record = [[0]*debt_T] * e_num
sys_sale = 0.0  # 市场销率
sys_mean_price = 0.0  # 市场定价均值
depreciation = 0.8  # 折旧率
f_product = 1 #销售额系数
f_consume = 1
Debt_i = 0.001 #利率

# profit = [True] * e_num  # 是否开始重新计算利润

b_data = [None] * b_num
e_data = [None] * e_num

# =====画图工具=====
# plt.ion()
# 最后100天的画图工具
data = []  # x轴，银行&企业共享x轴，表date
s_y = []  # y轴，表企业的市场销售率
e_y = [[] for i in range(e_product_num + e_consume_num)]  # y轴，表企业净利润
b_y = [[] for i in range(b_num)]  # y轴，表银行利润率
# 每1000天画一个点的画图工具
live_date = []  # x轴，表1000天个数
bd_y = [[] for i in range(b_num)]  # y轴,表银行一个ep存活天数均值
br_pool = [0] * b_num  # 辅助计算br_y，计算银行累积奖励
br_y = [[] for i in range(b_num)]  # y轴，表银行奖励*折扣因子
d_y = [[] for i in range(e_product_num + e_consume_num)]  # y轴,表企业一个ep存活天数均值
r_pool = [0] * (e_product_num + e_consume_num)  # 辅助计算r_y，计算企业累积奖励
r_y = [[] for i in range(e_product_num + e_consume_num)]  # y轴，表银行奖励*折扣因子
e_step = np.zeros(e_product_num + e_consume_num, dtype=int)  # 企业做决策步数
b_step = np.zeros(b_num, dtype=int)  # 银行做决策步数


# =====开始运行模型=====

episode = 10000 # 回合数
day = 50  # 结束日
for t in range(episode):

    # =====初始化银行企业数据=====
    # 企业的数据,数据: [array[现金M，存货X，负债D,收入R,将还款利息iD,成本C,利润π,市场价P,追加贷款意愿WNDF,机器K,劳动力L,实际机器getK,实际劳动力getL,次日定价NP],...]
    for i in e_product_mod:
        data = []
        for j in range(len(E_DATA)):
            data.append(0)
        data[E_DATA.M.value] = 0.0  # + random.randint(-300, 300)
        data[E_DATA.WNDF.value] = 100.0
        data[E_DATA.X.value] = random.randint(5, 10)
        e_data[i] = np.array(data)
    for i in e_consume_mod:
        data = []
        for j in range(len(E_DATA)):
            data.append(0)
        data[E_DATA.M.value] = 0.0  # + random.randint(-300, 300)
        data[E_DATA.WNDF.value] = 100.0
        data[E_DATA.X.value] = random.randint(5, 10)
        e_data[i] = np.array(data)
    # 银行的数据,数据: [array[现金，企业债券Ω,欠企业的钱D，贷款意向WNDB，企业观察OB,...]
    for i in b_mod:
        base_money = 0.0  # + random.randint(1, 200)
        b_data[i] = np.array(
            [base_money, np.array([0.0] * e_num), np.array([0.0] * e_num), np.array([0.0] * e_num), np.array(e_data)])
    # 初始化债务表
    debt_record = [[0] * debt_T] * e_num
    print("第"+str(t)+"回合开始")

    for d in range(day):
        # =================将前一天的成本作为当天的负利润=================
        # 如果将每天当天的成本与收入计入利润，两家企业一定是零和，扣除利息后两家企业利润之和必为负数
        # 任何时刻，企业利润总和＝这个时刻银行新增的贷款总量，所以计入利润时需要时间差
        set_init(e_data)
# =================清算还债=================
        if set_debt(b_data[0], [e_product_mod, e_consume_mod], e_data, d, debt_T, debt_record, Debt_i):
            print("第"+str(t)+"回合结束")
            break  # 有人破产 回合结束

# =====各部门根据前一天数据决策当日行动=====
# =================银行===================
        b_state = set_b_state(b_data[0])  # 根据前日结算重新整理b的数据
        # b_action = b_agent.run_bank(b_mod, b_state)  # action应为[ WNDB1,WNDB2 ]
        b_action = b_agent_sb.run_bank(b_mod, b_state, d)  # 手动输入
        # print(b_action)
# =================企业===================
        e_product_state = [None] * 1
        e_consume_state = [None] * 1
        e_product_temp_mod = [-1]
        e_consume_temp_mod = [-1]
        e_product_state[0] = np.array(set_e_state(e_data, e_product_mod[0]))  # 根据前日结算重新整理e的数据
        e_consume_state[0] = np.array(set_e_state(e_data, e_consume_mod[0]))

        e_product_action = [e_product_agent.run_enterprise(e_product_temp_mod, e_product_state, 0), e_product_agent.run_enterprise(e_product_temp_mod, e_product_state, 1),
                            e_product_agent.run_enterprise(e_product_temp_mod, e_product_state, 2), e_product_agent.run_enterprise(e_product_temp_mod, e_product_state, 3)]
        # e_product_agent.run_enterprise(e_product_mod, e_product_state)  # action应为[WNDF, K, L, NP]
        e_consume_action = [e_consume_agent.run_enterprise(e_consume_temp_mod, e_consume_state, 0), e_consume_agent.run_enterprise(e_consume_temp_mod, e_consume_state, 1),
                            e_consume_agent.run_enterprise(e_consume_temp_mod, e_consume_state, 2), e_consume_agent.run_enterprise(e_consume_temp_mod, e_consume_state, 3)]
        # e_consume_action = e_consume_agent.run_enterprise(e_consume_mod, e_consume_state)
        # e_product_action = e_product_agent_sb.run_enterprise(e_product_mod, e_product_state, d)  # 手动输入
        print('e_product_action', e_product_action)
        # e_consume_action = e_consume_agent_sb.run_enterprise(e_consume_mod, e_consume_state, d, True)
        print('e_consume_action', e_consume_action)

# =================对决策动作数据进行处理并赋值data===================
        set_b_action(b_data[0], b_action)
        set_e_action(e_data, e_product_mod, e_product_action, e_consume_action)
        set_e_action(e_data, e_consume_mod, e_consume_action, e_product_action)

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
        daily_settlement(b_data[0], e_data)
# =====关闭模型=====

b_mod = list(range(b_num))
e_product_mod = list(range(e_product_num))
e_consume_mod = list(range(e_consume_num))

b_agent.bank_mod_close(b_mod)
e_product_agent.enterprise_mod_close(e_product_mod)
e_consume_agent.enterprise_mod_close(e_consume_mod)

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

