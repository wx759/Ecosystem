import numpy as np
import random
import matplotlib.pyplot as plt
import math
from  enum import  IntEnum

# class E_DATA(IntEnum):
#     # 输入数据
#     M = 0         # 现金
#     X = 1         # 存货
#     D = 2         # 负债本金
#     R = 3         # 收入
#     iD = 4        # 负债利息
#     C = 5         # 成本支出
#     π = 6         # 利润
#     P = 7         # 市场价，即另一企业的NP值
#     # 输出数据
#     WNDF = 8      # 追加贷款意愿，即第二日希望追加多少的贷款
#     K = 9         # 当日购买机器
#     L = 10        # 当日雇佣劳动力
#     NP = 11       # 次日定价
#
# class B_DATA(IntEnum):
#     M = 0         # 现金
#     OM1 = 1        # 分别为一、二类企业债券
#     OM2 = 2
#     WNDB1 = 2      # 分别为一、二类企业贷款意愿
#     WNDB2 = 3
#     OB1 = 4        # 分别为一、二类企业观察，对应数据项应为列表，是企业观察
#     OB2 = 5


# 银行初始数据: [array[现金资产2000+，将收贷款本金0，本次收款利润率0，本次应收贷款本金0，本次应收贷款利息0，本次坏账率0], array([]),...]
#                           0              1               2               3                 4               5

# 企业初始数据: [array[现金资产500，将还贷款本金0，总贷款本金0，产品存量0，固定资产0，本期净利润0，
#                           0            1            2           3         4          5
#                    本期销售率0，本期产品定价0，本期市场销售率0，本期市场产品定价均值0], array([])...]
#                           6           7               8               9


# 设置mod
# 输入参数编号mod、是否为新企业is_new_mod，返回编号、是否为新企业
def set_mod(mod, is_new_mod):  # 以mod为准
    if type(mod) != list:  # 设置e_mod
        if is_new_mod:
            mod = -abs(mod); is_new_mod = False
        else:
            mod = abs(mod)
        return mod, is_new_mod
    for i in range(len(mod)):
        if is_new_mod[i]:  # 若企业i是新企业，把企业i的编号mod[i]置为负数，后面根据mod[i]的正负来判断是旧/新企业
            mod[i] = -abs(mod[i]); is_new_mod[i] = False
        else:
            mod[i] = abs(mod[i])


# 设置b_state
# 输入参数银行数据b_data、该loop企业破产率e_break，返回b_state
def set_b_state(b_data, e_break):
    b_state = [None] * len(b_data)
    for i in range(len(b_data)):
        b_state[i] = list(b_data[i]); b_state[i].append(e_break)
        b_state[i] = np.array(b_state[i])
    return b_state


# 离散化b_action
# 输入参数银行个数、银行行为b_action、银行数据b_data，返回具体银行行为
def set_b_action(b_num, b_action, b_data):
        for i in range(b_num):
            b_action[0, i] = (math.log((b_action[0, i] + 28)/20, math.e) / 2) * b_data[i][0]  # 比论文多了/2，因为范围太大，缩小
            b_action[0, i] = round(b_action[0, i], 3)
            b_action[1, i] = math.log((b_action[1, i] + 21)/20, math.e) / 2
            b_action[1, i] = round(b_action[1, i], 3)



# 设置e_state
# 输入参数决策指示符flag，企业数据e_data，银行利息b_interest
# def set_e_state(flag, e_data, b_interest=None, e_num=None, price=None, in_worker=None):
#     cmp = [1000, 100, 500, 20, 5, 0, 0.2, 100]  # ？？
#     if flag == 1:
#         e_state = [0] * len(e_data)  # e_state = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
#         for i in range(len(cmp)):  # 0-7
#             if e_data[i] <= cmp[i]:
#                 e_state[i] = 0  # ？？
#             else:
#                 e_state[i] = 1
#         for i in range(8, len(e_data)):
#             if e_data[i] <= e_data[i-2]:  # 本期市场销售率<市场销售率; 本期市场产品定价均值<本期产品定价
#                 e_state[i] = 0  # ？？
#             else:
#                 e_state[i] = 1
#         e_state.append(1 if b_interest >= 0.5 else 0); e_state.insert(0, flag)  # ？？
#         return e_state
#     elif flag == 2:
#         e_state = [None] * e_num
#         for i in range(e_num):
#             e_state[i] = list(e_data[i])
#             for j in range(len(cmp)):
#                 if e_data[i][j] <= cmp[j]:
#                     e_state[i][j] = 0
#                 else:
#                     e_state[i][j] = 1
#             for j in range(8, len(e_data)):
#                 if e_data[i][j] <= e_data[i - 2][j]:  # e_data[i][j] <= e_data[i][j-2]
#                     e_state[i][j] = 0
#                 else:
#                     e_state[i][j] = 1
#         for i in range(e_num):
#             e_state[i].append(0); e_state[i].insert(0, flag);e_state[i] = np.array(e_state[i],dtype=float)
#         return e_state
#     elif flag == 3:
#         e_state = [0] * len(e_data)
#         for i in range(len(cmp)):
#             if e_data[i] <= cmp[i]:
#                 e_state[i] = 0
#             else:
#                 e_state[i] = 1
#         for i in range(8, len(e_data)):
#             if e_data[i] <= e_data[i - 2]:
#                 e_state[i] = 0
#             else:
#                 e_state[i] = 1
#         e_state.append(0 if price >= 100 else 1);e_state.insert(0, flag)
#         return e_state
#     else:
#         e_state = [None] * e_num
#         for i in range(e_num):
#             e_state[i] = list(e_data[i])
#             for j in range(len(cmp)):
#                 if e_data[i][j] <= cmp[j]:
#                     e_state[i][j] = 0
#                 else:
#                     e_state[i][j] = 1
#             for j in range(8, len(e_data)):
#                 if e_data[i][j] <= e_data[i - 2][j]:  # e_data[i][j] <= e_data[i][j-2]
#                     e_state[i][j] = 0
#                 else:
#                     e_state[i][j] = 1
#         for i in range(e_num):
#             e_state[i].append(0 if in_worker[i] <= 10 else 1);e_state[i].insert(0, flag);e_state[i] = np.array(e_state[i], dtype=float)
#         return e_state


def set_e_state(flag, e_data, b_interest=None, e_num=None, price=None, in_worker=None):
    if flag == 1:  # 1表示企业决策是否借钱
        e_state = list(e_data); e_state.append(b_interest); e_state.insert(0, flag)
        return e_state
    elif flag == 2:  # 2表示决策产品定价
        e_state = [None] * e_num
        for i in range(e_num):
            e_state[i] = list(e_data[i]); e_state[i].append(0); e_state[i].insert(0, flag)
            e_state[i] = np.array(e_state[i])
        # print(e_state)
        return e_state
    elif flag == 3:  # 3表示决策是否购买产品
        e_state = list(e_data); e_state.append(price); e_state.insert(0, flag)
        return e_state
    else:  # 4表示决策员工工资
        e_state = [None] * e_num
        for i in range(e_num):
            e_state[i] = list(e_data[i]); e_state[i].append(in_worker[i]); e_state[i].insert(0, flag)
            e_state[i] = np.array(e_state[i])
        return e_state


# 返回清算结果
# 输入参数主体类型指示符flag，公司数据data，公司个数num
def set_settlement(flag, data, num, b_e_tra, is_new_mod, in_worker=None, t=None, e_pro=None):
    if flag == 1:  # 银行清算
        for i in range(num):
            if data[i][0] <= 100:  # 破产
                # 数据初始化
                money = 2000  # + random.randint(1, 200)
                data[i] = np.array([money, 0, 0, 0, 0, 0], dtype=float)
                # t初始化
                t[i] = 0
                # 银行和企业交易数据初始化
                b_e_tra[i] = np.zeros((np.shape(b_e_tra[i])))
                is_new_mod[i] = True
    else:  # 企业清算
        e_break = 0.0
        for i in range(num):
            if data[i][0] <= 0:  # 破产
                # 数据初始化
                money = 1000  # + random.randint(-300, 300)
                data[i] = np.array([money, 0, 0, 8, 5, 0, 0, 0, 0, 0], dtype=float)
                # 企业员工初始化
                in_worker[i] = 0
                # t初始化
                t[i] = 0
                e_pro[i] = np.zeros(np.shape(e_pro[i]))
                # 计数破产企业数
                e_break += 1
                # 银行和企业交易数据初始化
                for m in range(np.shape(b_e_tra)[0]):
                    for n in range(np.shape(b_e_tra)[1]):
                        # 修改点
                        if b_e_tra[m, n, i] > 0:
                            b_e_tra[m, n, i] = -b_e_tra[m, n, i]  # 还没到还款时间，但企业破产了，标为负数
                            b_e_tra[m, n, 4] += b_e_tra[m, n, i]  # 加了下期应还本金减掉
                is_new_mod[i] = True
        e_break = e_break / num
        return e_break


# 银行收账（企业还钱）
def b_get_debt(b_data, e_data, b_t, debt_t, b_num, e_num, b_e_tra, profit, t, br_pool, b_step, b_money):
    # 银行
    for i in range(b_num):
        if b_t[i] >= debt_t:  # 小于的话银行还没到收帐的时间
            b_get = 0; b_bad_debt = 0; num = 0  # b_get银行实际收回的总钱数 b_bad_debt坏账数 num总次数 debt银行收回的钱
            for j in range(e_num):
                if b_e_tra[i, t % debt_t, j] < 0:  # <0说明前一个企业破产了
                    b_bad_debt += 1; num += 1  # 银行坏账+1
                elif b_e_tra[i, t % debt_t, j] > 0:
                    num += 1
                    debt = b_e_tra[i, t % debt_t, j] * (1 + b_e_tra[i, t % debt_t, e_num])
                    if e_data[j][0] < debt:
                        debt = e_data[j][0] if e_data[j][0] > 0 else 0
                        b_bad_debt += 1
                    # 企业现金资产 > 应还的钱
                    b_data[i][0] += debt; b_data[i][0] = round(b_data[i][0], 3)
                    e_data[j][0] -= debt; e_data[j][0] = round(e_data[j][0], 3)
                    e_data[j][2] -= b_e_tra[i, t % debt_t, j]; e_data[j][2] = round(e_data[j][2], 3)
                    b_get += debt; b_get = round(b_get, 3)
                    if profit[j]:  # 如果企业j要重新计算利润
                        e_data[j][5] = 0; profit[j] = False
                    e_data[j][5] -= debt; e_data[j][5] = round(e_data[j][5], 3)  # 计算利润

            if b_data[i][0] > 100:
                b_data[i][3] = b_data[i][1]  # 本期应收贷款本金 = 将收贷款本金
                b_data[i][1] = b_e_tra[i, (t+1) % debt_t, e_num+1]  # 将收贷款本金 = 银行i下一天的借出总数
                if b_data[i][3] != 0:
                    b_data[i][2] = b_get/b_data[i][3]; b_data[i][2] = round(b_data[i][2], 3)  # 本次收款利润率=收回的钱/应收贷款本金
                else: b_data[i][2] = 0  # 本次收款利润率
                b_data[i][4] = b_e_tra[i, t % debt_t, e_num]  # 本次应收贷款利息
                if num != 0: b_data[i][5] = b_bad_debt/num  # 本次坏账率
                else: b_data[i][5] = 0  # 本次坏账率
                # ===== 银行累积奖励 ===== #
                br_pool[i] += b_data[i][0] #可以注掉
                # br_pool[i] += b_data[i][0] * (0.95 ** b_step[i])  # 银行累积奖励画图准备
                # br_pool[i] += (b_data[i][0]+100*b_data[i][2]) * (0.95 ** b_step[i])  # 银行累积奖励画图准备
                b_step[i] += 1
            else:
                # ===== 银行累积奖励 ===== #
                br_pool[i] -= b_money[i] #可以注掉
                # br_pool[i] -= b_money[i] * (0.95 ** b_step[i])  # 银行累积奖励画图准备
            #   br_pool[i] -= (b_money[i]+100*b_data[i][2]) * (0.95 ** b_step[i])  # 银行累积奖励画图准备


# 设置企业将还款本金
def e_ret_debt(e_data, b_t, e_t, debt_t, b_num, e_num, b_e_tra, t):
    for i in range(e_num):
        if e_t[i] >= debt_t:  # 企业的存活时间大于还钱周期
            debt = 0
            for j in range(b_num):
                if b_e_tra[j, (t + 1) % debt_t, i] < 0:  # 修改点
                    continue
                else:
                    debt += b_e_tra[j, (t + 1) % debt_t, i]  # 下个loop企业i向所有银行贷款的总额
            e_data[i][1] = round(debt, 3)  # 将还贷款本金


# 设置初始将到期收款本金、将到期还款本金
def set_debt_t(b_data, e_data, b_t, e_t, b_num, e_num, b_e_tra, debt_t, t):
    # 银行
    for i in range(b_num):
        if b_t[i] == 0:
            b_data[i][1] = b_e_tra[i, t % debt_t, e_num+1]  # 将收贷款本金 = 借出总数
    # 企业
    for i in range(e_num):
        if e_t[i] == 0:
            debt = 0
            for j in range(b_num):
                debt += b_e_tra[j, t % debt_t, i]; debt = round(debt, 3)
            e_data[i][1] = debt


# 画图
def paint_graph(x, y, flag, title_str, xlab, ylab,
                y1=None, title_str1=None):
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 防止中文标签乱码
    plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
    plt.figure(flag)
    plt.clf()
    plt.suptitle(title_str, fontsize=12)
    plt.xlabel(xlab, fontsize=10)
    plt.ylabel(ylab, fontsize=10)
    plt.plot(x, y, c='y', ls='-', marker='h', mec='k')
    plt.pause(0.009)
