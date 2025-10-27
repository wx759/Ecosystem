import numpy as np
import random
from ddpg_new_bank3 import bank_nnu
from ddpg_new_bank3 import bank_nnu_without_intelligent as sb_bank_nnu
from ddpg_new_enterprise3 import enterprise_nnu
from ddpg_new_enterprise3 import enterprise_nnu_without_intelligent as sb_enterprise_nnu
import time
from ddpg_new_calculate3 import *
import matplotlib.pyplot as plt
import copy
import math
from pandas import DataFrame

plt.rcParams["font.sans-serif"]=["SimHei"] #设置字体
plt.rcParams["axes.unicode_minus"]=False #该语句解决图像中的“-”负号的乱码问题
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
b_agent = bank_nnu(b_num,'bank')
e_product_agent = enterprise_nnu(e_product_num,'product')
e_consume_agent = enterprise_nnu(e_consume_num,'consume')  # 银行、企业大脑
# e_product_agent_sb = sb_enterprise_nnu(e_product_num, "生产机器的企业")
# e_consume_agent_sb = sb_enterprise_nnu(e_consume_num, "生产粮食的企业")
b_mod = list(range(0, b_num))
e_product_mod = list(range(0, e_product_num))
e_consume_mod = list(range(e_product_num, e_num))  # 银行、企业的编号,对应数组下标
e_mod = list(range(0, e_num))
debt_T = 6  # 还钱周期
debt_record = [[0] * (debt_T + 1)] * e_num # 多的一个存放当日新增借款，偿还上个周期的d%debt_t的借款后将其替换
f_product = 1  #销售额系数
f_consume = 1
Debt_i = 0.005  #利率

# profit = [True] * e_num  # 是否开始重新计算利润

b_data = [None] * b_num
e_data = [None] * e_num
#
action_count = [[[] for i in range(4)] for i in range(2)]
b_action_count = [[] for i in range(2)]


episode = 999   # 回合数
day = 100  # 结束日
is_percent = True # 选择百分比还是固定值
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
WNDB1 = []
WNDB2 = []
real_WNDB1 = []
real_WNDB2 = []
K_NP = []
L_NP = []
e_td_err1 = []
e_td_err2 = []
b_td_err = []

csv_output_data = []  # [t d M0 NP0 deltaX0 M1 NP1 deltaX1 ] * (episode * day)


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
            [base_money, np.array([0.0] * e_num), np.array([0.0] * e_num), np.array([0.0] * e_num), np.array([0.0] * e_num), np.array(e_data),0])
    # 初始化债务表
    debt_record = [[0] * (debt_T+1), [0] * (debt_T+1)]

    e_temp_mod = [-x for x in range(1,e_num+1)]
    e_product_temp_mod = [-1] * e_product_num
    e_consume_temp_mod = [-1] * e_consume_num
    print("第"+str(t)+"回合开始")
    if t == episode - 1000:
        K_X = [0]
        L_X = [0]
        K_desired = [[0] for i in range(e_num)]
        L_desired = [[0] for i in range(e_num)]
        total_R = [[0] for i in range(e_num)]
        total_C = [[0] for i in range(e_num)]
        total_iD = [[0] for i in range(e_num)]
        final_D = [[0] for i in range(e_num)]
        final_M = [[0] for i in range(e_num)]
        WNDF1 = [0]
        WNDF2 = [0]
        WNDB1 = [0]
        WNDB2 = [0]
        real_WNDB1 = [0]
        real_WNDB2 = [0]
        K_NP = [0]
        L_NP = [0]
        action_count = [[[] for i in range(4)] for i in range(2)]
        b_action_count = [[] for i in range(2)]
    b_temp_mod = [-1] * b_num
    e_product_state = [None] * e_product_num
    e_consume_state = [None] * e_consume_num
    e_product_state = set_e_state(e_data, e_product_mod)  # 初始化
    e_consume_state = set_e_state(e_data, e_consume_mod)
    b_state = set_b_state(b_data[0], b_mod)  # 初始化
    reset_bank_current(b_data[0])

    is_End = False
    is_Fall = [False,False]
    for d in range(day):
        if is_End:
            break;
        # =================将前一天的成本作为当天的负利润=================
        # 如果将每天当天的成本与收入计入利润，两家企业一定是零和，扣除利息后两家企业利润之和必为负数
        # 任何时刻，企业利润总和＝这个时刻银行新增的贷款总量，所以计入利润时需要时间差

        set_init(e_data, b_data[0])
# =====各部门根据前一天数据决策当日行动=====
        # =================企业===================
        e_consume_action = e_consume_agent.run_enterprise(e_consume_temp_mod, e_consume_state)
        e_product_action = e_product_agent.run_enterprise(e_product_temp_mod, e_product_state)
        e_consume_action = e_consume_action[0].reshape(-1, 4)
        e_product_action = e_product_action[0].reshape(-1, 4)
        for i in range(4):
            action_count[0][i].append(e_consume_action[0][i])
            action_count[0][i].append(e_product_action[0][i])

        # =====================================================================统计量输出=============================================================================
        print("consume_action  ", e_consume_action, "product_action  ", e_product_action)
        # ==================================================================================================================================================

        if is_percent:
            set_e_action_percent(e_data, e_product_mod, e_product_action, d is 0)
            set_e_action_percent(e_data, e_consume_mod, e_consume_action, d is 0)
        else:
            set_e_action(e_data, e_product_mod, e_product_action, d is 0)
            set_e_action(e_data, e_consume_mod, e_consume_action, d is 0)
        # =================银行===================  企业必须先行动且赋完值 银行再根据决策给出额度
        update_B_after_E_action(b_data[0],e_data) # 更新企业决策后的OB 以此作为b_agent的S*的一部分
        b_state = np.array(set_b_state(b_data[0],b_mod))  # 填入
        if d>0:
            td_err = b_agent.env_upd(b_mod,b_state,d,True,is_End=is_End)
            if td_err != None :
                b_td_err.append(td_err)
                csv_output_data[len(csv_output_data) - 1].append(td_err)
        b_action = b_agent.run_bank(b_temp_mod, b_state)  # action应为[ WNDB ]
        for i in range(2):
            b_action_count[i].append(b_action[0][0][i])

# =================对决策动作数据进行处理并赋值data===================
        print("b_action", b_action)
        if is_percent:
            set_b_action_percent(b_data[0], e_data, b_action[0], e_product_mod, e_consume_mod, d)
        else:
            set_b_action(b_data[0], e_data, b_action[0], e_product_mod, e_consume_mod, d)
        csv_kuaizhao(b_data[0],e_data,t,d,debt_T,e_product_mod,e_consume_mod,csv_output_data)


        # =====企业向银行贷款===== 此时银行现金不会减少
        rent_money(b_data[0], [e_product_mod, e_consume_mod], e_data, d, debt_T, debt_record)


# =====企业开始交易=====
# =====交易前先检测是否能够买得起=========
        trade_check(e_data,e_product_mod,E_DATA.K,E_DATA.L,E_DATA.k,E_DATA.l)
        trade_check(e_data,e_consume_mod,E_DATA.L,E_DATA.K,E_DATA.l,E_DATA.k)

# =====开始交易机器===== product[0]作为卖家，pro[0]和con[0] 共同作为买家 商品为机器（K) 结算在get(K)上
        trade(b_data[0], e_product_mod[0], [e_product_mod[0], e_consume_mod[0]], e_data, E_DATA.k, E_DATA.getK)
# =====开始交易工人===== consume[0]作为卖家，pro[0]和con[0] 共同作为买家 商品为工人（L) 结算在get(L)上
        trade(b_data[0], e_consume_mod[0], [e_product_mod[0], e_consume_mod[0]], e_data, E_DATA.l, E_DATA.getL)

# =====企业进行生产=====
        product(e_data, [e_product_mod, e_consume_mod])
# =====修改次日定价=====
        print(e_consume_action, e_product_action)

        if is_percent:
            set_e_P_percent(e_data, e_product_mod, e_product_action, d is 0)
            set_e_P_percent(e_data, e_consume_mod, e_consume_action, d is 0)
        else:
            set_e_P(e_data, e_product_mod, e_product_action, d is 0)
            set_e_P(e_data, e_consume_mod, e_consume_action, d is 0)
        daily_settlement(b_data[0], e_data, t, d)
        # =====当日数据进行结算，并指定次日价格=====
        # =================清算还债=================
        is_Fall,is_Down = set_debt(b_data[0], e_mod, e_data, d, debt_T, debt_record, Debt_i)

        show_daily(b_data[0], e_data, t, d)
        csv_update(b_data[0], e_data, t, d,debt_T,e_product_mod,e_consume_mod,csv_output_data)

        if (is_Down or d == day - 1 or StoreOver(e_data)):
            print("第" + str(t) + "回合结束")
            if t % 100 == 0:
                live_day.append(d)
            else:
                live_day[t // 100] += d
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
            if d is not 0:
                K_desired[0][len(K_desired[0]) - 1] /= d
                K_desired[1][len(K_desired[1]) - 1] /= d
                L_desired[0][len(L_desired[0]) - 1] /= d
                L_desired[1][len(L_desired[1]) - 1] /= d
                K_NP[len(K_NP) - 1] /= d
                L_NP[len(L_NP) - 1] /= d
                WNDF1[len(WNDF1) - 1] /= d
                WNDF2[len(WNDF2) - 1] /= d
                WNDB1[len(WNDB1) - 1] /= d
                WNDB2[len(WNDB2) - 1] /= d
                real_WNDB1[len(real_WNDB1) - 1] /= d
                real_WNDB2[len(real_WNDB2) - 1] /= d

            is_End = True  # 有人破产清算 回合结束
        e_product_state = set_e_state(e_data, e_product_mod)  # 根据今日表现得出S* 并作为次日S
        e_consume_state = set_e_state(e_data, e_consume_mod)

        td_err1 = e_product_agent.env_upd(e_product_temp_mod,e_product_state,d,True,is_End=is_End,is_Fall=is_Fall[e_product_mod[0]]) # 根据S*得出reward 并存入经验池
        td_err2 = e_consume_agent.env_upd(e_consume_temp_mod,e_consume_state,d,True,is_End=is_End,is_Fall=is_Fall[e_consume_mod[0]])
        if td_err1 != None:
            e_td_err1.append(td_err1)
            csv_output_data[len(csv_output_data)-1].append(td_err1)
        if td_err2 != None:
            e_td_err2.append(td_err2)
            csv_output_data[len(csv_output_data)-1].append(td_err2)
        if is_End: # 企业破产则直接更新S*和reward，否则S*中的OB部分应为次日完成决策后的企业动作
            b_state = np.array(set_b_state(b_data[0], b_mod))  # 根据结算(主要是偿还贷款)重新整理b的数据
            td_err = b_agent.env_upd(b_mod,b_state,d,True,is_End=is_End)
            if td_err != None:
                b_td_err.append(td_err)
                csv_output_data[len(csv_output_data) - 1].append(td_err)
        if d == 0:
            WNDF1.append(e_data[0][E_DATA.WNDF.value])
            WNDF2.append(e_data[1][E_DATA.WNDF.value])
            WNDB1.append(b_data[0][B_DATA.WNDB.value][0])
            WNDB2.append(b_data[0][B_DATA.WNDB.value][1])
            real_WNDB1.append(b_data[0][B_DATA.real_WNDB.value][0])
            real_WNDB2.append(b_data[0][B_DATA.real_WNDB.value][1])
            K_X.append(e_data[0][E_DATA.X.value])
            L_X.append(e_data[1][E_DATA.X.value])
            K_desired[0].append(e_data[0][E_DATA.K.value])
            K_desired[1].append(e_data[1][E_DATA.K.value])
            L_desired[0].append(e_data[0][E_DATA.L.value])
            L_desired[1].append(e_data[1][E_DATA.L.value])
            K_NP.append(e_data[0][E_DATA.NP.value])
            L_NP.append(e_data[1][E_DATA.NP.value])
        else:
            WNDF1[len(WNDF1)-1] += (e_data[0][E_DATA.WNDF.value])
            WNDF2[len(WNDF2)-1] += (e_data[1][E_DATA.WNDF.value])
            WNDB1[len(WNDB1) - 1] += (b_data[0][B_DATA.WNDB.value][0])
            WNDB2[len(WNDB2) - 1] += (b_data[0][B_DATA.WNDB.value][1])
            real_WNDB1[len(real_WNDB1) - 1] += (b_data[0][B_DATA.real_WNDB.value][0])
            real_WNDB2[len(real_WNDB2) - 1] += (b_data[0][B_DATA.real_WNDB.value][1])
            K_X[len(K_X)-1] += (e_data[0][E_DATA.X.value])
            L_X[len(L_X)-1] += (e_data[1][E_DATA.X.value])
            K_desired[0][len(K_desired[0])-1] += (e_data[0][E_DATA.K.value])
            K_desired[1][len(K_desired[1])-1] += (e_data[1][E_DATA.K.value])
            L_desired[0][len(L_desired[0])-1] += (e_data[0][E_DATA.L.value])
            L_desired[1][len(L_desired[1])-1] += (e_data[1][E_DATA.L.value])
            K_NP[len(K_NP)-1] += (e_data[0][E_DATA.NP.value])
            L_NP[len(L_NP)-1] += (e_data[1][E_DATA.NP.value])
        print(debt_record)

# =====关闭模型=====

b_mod = list(range(b_num))
e_product_mod = list(range(e_product_num))
e_consume_mod = list(range(e_consume_num))
end_struct_time = time.strftime("_%Y_%m_%d_%H_%M_%S", time.localtime())
# b_agent.bank_mod_close(b_mod)
# e_product_agent.enterprise_mod_close(e_product_mod)
# e_consume_agent.enterprise_mod_close(e_consume_mod)

# # =====导出数据=====
# b_path = 'E://run//' + 'DQN银行20w eps_0.95_8w_0.02 lr=0.008 (本金-贷款)无折扣.csv'
# e_path = 'E://run//' + 'DQN企业20w eps_0.95_8w_0.02 lr=0.008 (本金-贷款)无折扣.csv'
# b_path = 'E://run//' + 'DRQN银行20w eps=0.02 lr=0.008 (本金-贷款)无折扣.csv'
e_path = 'E://run//' + 'DDPG/百分比0.5;企业正常;银行正常;储备金每回合增长10%(基于流动性)' +  end_struct_time + '.csv'
# b_path = 'E://run//' + 'NN-DQN银行20w eps=0.02 lr=0.008 (本金-贷款)无折扣.csv'
# e_path = 'E://run//' + 'NN-DQN企业20w eps=0.02 lr=0.008 (本金-贷款)无折扣.csv'
# d M0 NP0 deltaX0 M1 NP1 deltaX1
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
# [d M0 NP0 deltaX0 M1 NP1 deltaX1 ] * (episode * day)
# d = {'天数':csv_output_data[0],'生产企业现金':csv_output_data[1],'生产企业定价':csv_output_data[2],'生产企业产量':csv_output_data[3],
#         '消费企业现金':csv_output_data[4],'消费企业定价':csv_output_data[5],'消费企业产量':csv_output_data[6]}
columns = ['回合','天数',"统计:生产企业现金","统计:消费企业现金","统计:银行现金","统计:生产企业机器意愿","统计:消费企业机器意愿","统计:生产企业工人意愿","统计:消费企业工人意愿",
           "统计:生产企业产量意愿","统计:消费企业产量意愿","统计:生产企业今日交易价格","统计:消费企业今日交易价格","统计:生产企业次日定价","统计:消费企业次日定价",
           "统计:生产企业实际产量","统计:消费企业实际产量","统计:生产企业实际交易量","统计:消费企业实际交易量"]
for j in e_column:
    columns.append("生产企业" + j)
    columns.append("消费企业" + j)
for j in b_column:
    columns.append("银行" + j)

columns.append("还款周期")
columns.append("银行储备金")
columns.append("银行借贷与现金比例")
columns.append("剩余可放贷额度")
columns.append("借贷是否超过储备金上限")
columns.append("生产企业本回合交易量是否高于产量")
columns.append("消费企业本回合交易量是否高于产量")


columns.append("e_err1")
columns.append("e_err2")
columns.append("b_err")
print(len(csv_output_data[0]),len(columns))

df = DataFrame(csv_output_data,columns=columns,dtype=float)
df.to_csv(
    e_path,
    index=False,
    encoding='utf-8_sig'
)

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
print('结束时间：',end_struct_time)
print("均值 中位数 方差 标准差 最大值 最小值")
e_r_res=[[] for i in range(e_num)]

for i in range(e_num):
    e_r_res[i].append(e_y[i][0])
    for j in range(1, len(e_y[i])):
        e_r_res[i].append(e_r_res[i][j-1]+e_y[i][j])

count_episode = 1000

print("WNDF1", set_data_round([np.mean(WNDF1[-count_episode: -1]), np.median(WNDF1[-count_episode:-1]), np.var(WNDF1[-count_episode:-1]),
                               np.std(WNDF1[-count_episode:-1]), max(WNDF1[-count_episode:-1]), min(WNDF1[-count_episode:-1])]))
print("WNDF2", set_data_round([np.mean(WNDF2[-count_episode:-1]), np.median(WNDF2[-count_episode:-1]), np.var(WNDF2[-count_episode:-1]),
                               np.std(WNDF2[-count_episode:-1]), max(WNDF2[-count_episode:-1]), min(WNDF2[-count_episode:-1])]))
print("WNDB1", set_data_round([np.mean(WNDB1[-count_episode:-1]), np.median(WNDB1[-count_episode:-1]), np.var(WNDB1[-count_episode:-1]),
                               np.std(WNDB1[-count_episode:-1]), max(WNDB1[-count_episode:-1]), min(WNDB1[-count_episode:-1])]))
print("WNDB2", set_data_round([np.mean(WNDB2[-count_episode:-1]), np.median(WNDB2[-count_episode:-1]), np.var(WNDB2[-count_episode:-1]),
                               np.std(WNDB2[-count_episode:-1]), max(WNDB2[-count_episode:-1]), min(WNDB2[-count_episode:-1])]))
print("real_WNDB1", set_data_round([np.mean(real_WNDB1[-count_episode:-1]), np.median(real_WNDB1[-count_episode:-1]), np.var(real_WNDB1[-count_episode:-1]),
                               np.std(real_WNDB1[-count_episode:-1]), max(real_WNDB1[-count_episode:-1]), min(real_WNDB1[-count_episode:-1])]))
print("real_WNDB2", set_data_round([np.mean(real_WNDB2[-count_episode:-1]), np.median(real_WNDB2[-count_episode:-1]), np.var(real_WNDB2[-count_episode:-1]),
                               np.std(real_WNDB2[-count_episode:-1]), max(real_WNDB2[-count_episode:-1]), min(real_WNDB2[-count_episode:-1])]))
print("总利润1", set_data_round([np.mean(e_y[0][-count_episode:-1]), np.median(e_y[0][-count_episode:-1]), np.var(e_y[0][-count_episode:-1]),
                               np.std(e_y[0][-count_episode:-1]), max(e_y[0][-count_episode:-1]), min(e_y[0][-count_episode:-1])]))
print("总利润2", set_data_round([np.mean(e_y[1][-count_episode:-1]), np.median(e_y[1][-count_episode:-1]), np.var(e_y[1][-count_episode:-1]),
                              np.std(e_y[1][-count_episode:-1]), max(e_y[1][-count_episode:-1]), min(e_y[1][-count_episode:-1])]))
print("loss1", set_data_round([np.mean(e_td_err1[-count_episode:-1]), np.median(e_td_err1[-count_episode:-1]), np.var(e_td_err1[-count_episode:-1]),
                               np.std(e_td_err1[-count_episode:-1]), max(e_td_err1[-count_episode:-1]), min(e_td_err1[-count_episode:-1])]))
print("loss2", set_data_round([np.mean(e_td_err2[-count_episode:-1]), np.median(e_td_err2[-count_episode:-1]), np.var(e_td_err2[-count_episode:-1]),
                              np.std(e_td_err2[-count_episode:-1]), max(e_td_err2[-count_episode:-1]), min(e_td_err2[-count_episode:-1])]))
print("loss_bank_forward", set_data_round([np.mean(b_td_err[0:count_episode]), np.median(b_td_err[0:count_episode]), np.var(b_td_err[0:count_episode]),
                              np.std(b_td_err[0:count_episode]), max(b_td_err[0:count_episode]), min(b_td_err[0:count_episode])]))
print("loss1_forward", set_data_round([np.mean(e_td_err1[0:count_episode]), np.median(e_td_err1[0:count_episode]), np.var(e_td_err1[0:count_episode]),
                               np.std(e_td_err1[0:count_episode]), max(e_td_err1[0:count_episode]), min(e_td_err1[0:count_episode])]))
print("loss2_forward", set_data_round([np.mean(e_td_err2[0:count_episode]), np.median(e_td_err2[0:count_episode]), np.var(e_td_err2[0:count_episode]),
                              np.std(e_td_err2[0:count_episode]), max(e_td_err2[0:count_episode]), min(e_td_err2[0:count_episode])]))
print("loss_bank", set_data_round([np.mean(b_td_err[-count_episode:-1]), np.median(b_td_err[-count_episode:-1]), np.var(b_td_err[-count_episode:-1]),
                              np.std(b_td_err[-count_episode:-1]), max(b_td_err[-count_episode:-1]), min(b_td_err[-count_episode:-1])]))
print("游戏结束时余额1", set_data_round([np.mean(final_M[0]), np.median(final_M[0]), np.var(final_M[0]), np.std(final_M[0]), max(final_M[0]), min(final_M[0])]))
print("游戏结束时余额2", set_data_round([np.mean(final_M[1]), np.median(final_M[1]), np.var(final_M[1]), np.std(final_M[1]), max(final_M[1]), min(final_M[1])]))
print("游戏结束时支付总利息1", set_data_round([np.mean(total_iD[0]), np.median(total_iD[0]), np.var(total_iD[0]), np.std(total_iD[0]), max(total_iD[0]), min(total_iD[0])]))
print("游戏结束时支付总利息2", set_data_round([np.mean(total_iD[1]), np.median(total_iD[1]), np.var(total_iD[1]), np.std(total_iD[1]), max(total_iD[1]), min(total_iD[1])]))
print("银行利润", set_data_round([np.mean(b_y[0][-count_episode:-1]), np.median(b_y[0][-count_episode:-1]), np.var(b_y[0][-count_episode:-1]),
                              np.std(b_y[0][-count_episode:-1]), max(b_y[0][-count_episode:-1]), min(b_y[0][-count_episode:-1])]))
print("结束天数", set_data_round([np.mean(live_day), np.median(live_day), np.var(live_day),
                              np.std(live_day), max(live_day), min(live_day)]))
print("机器存货", set_data_round([np.mean(K_X[-count_episode:-1]), np.median(K_X[-count_episode:-1]), np.var(K_X[-count_episode:-1]),
                              np.std(K_X[-count_episode:-1]), max(K_X[-count_episode:-1]), min(K_X[-count_episode:-1])]))
print("粮食存货", set_data_round([np.mean(L_X[-count_episode:-1]), np.median(L_X[-count_episode:-1]), np.var(L_X[-count_episode:-1]),
                              np.std(L_X[-count_episode:-1]), max(L_X[-count_episode:-1]), min(L_X[-count_episode:-1])]))
print("机器定价", set_data_round([np.mean(K_NP[-count_episode:-1]), np.median(K_NP[-count_episode:-1]), np.var(K_NP[-count_episode:-1]),
                              np.std(K_NP[-count_episode:-1]), max(K_NP[-count_episode:-1]), min(K_NP[-count_episode:-1])]))
print("粮食定价", set_data_round([np.mean(L_NP[-count_episode:-1]), np.median(L_NP[-count_episode:-1]), np.var(L_NP[-count_episode:-1]),
                              np.std(L_NP[-count_episode:-1]), max(L_NP[-count_episode:-1]), min(L_NP[-count_episode:-1])]))
print("K1决策", set_data_round([np.mean(K_desired[0][-count_episode:-1]), np.median(K_desired[0][-count_episode:-1]), np.var(K_desired[0][-count_episode:-1]),
                              np.std(K_desired[0][-count_episode:-1]), max(K_desired[0][-count_episode:-1]), min(K_desired[0][-count_episode:-1])]))
print("K2决策", set_data_round([np.mean(K_desired[1][-count_episode:-1]), np.median(K_desired[1][-count_episode:-1]), np.var(K_desired[1][-count_episode:-1]),
                              np.std(K_desired[1][-count_episode:-1]), max(K_desired[1][-count_episode:-1]), min(K_desired[1][-count_episode:-1])]))
print("L1决策", set_data_round([np.mean(L_desired[0][-count_episode:-1]), np.median(L_desired[0][-count_episode:-1]), np.var(L_desired[0][-count_episode:-1]),
                              np.std(L_desired[0][-count_episode:-1]), max(L_desired[0][-count_episode:-1]), min(L_desired[0][-count_episode:-1])]))
print("L2决策", set_data_round([np.mean(L_desired[1][-count_episode:-1]), np.median(L_desired[1][-count_episode:-1]), np.var(L_desired[1][-count_episode:-1]),
                              np.std(L_desired[1][-count_episode:-1]), max(L_desired[1][-count_episode:-1]), min(L_desired[1][-count_episode:-1])]))


print("银行储备金：",bank_base,"现金/可新放出贷款倍率：",bank_γ)
print("act决策百分比：",act_percent,"WNDF决策百分比：",WBDF_percent,"银行决策百分比：",b_percent)
print("还钱周期:",debt_T,"利率:",Debt_i)
print("结束回合",episode,"结束天数",day)

plt.figure('企业单回合总利润')
plt.plot(np.array(e_y[0]), c='r', label='product')
plt.plot(np.array(e_y[1]), c='b', label='consume')
plt.ylabel('企业单回合总利润')
plt.xlabel('游戏回合')
plt.legend(loc='best')

plt.figure('企业单回合总利润之和')
plt.plot(np.array(e_r_res[0]), c='r', label='product')
plt.plot(np.array(e_r_res[1]), c='b', label='consume')
# plt.plot(np.array(e_r_res[0]+e_r_res[1]), c='y', label='3')
plt.ylabel('企业单回合总利润之和')
plt.xlabel('游戏回合')
plt.legend(loc='best')

plt.figure('游戏结束天数')
plt.plot(np.array(live_day), c='r', label='1')
plt.ylabel('游戏结束天数')
plt.xlabel('游戏回合')
plt.legend(loc='best')

plt.figure('银行利润')
plt.plot(np.array(b_y[0]), c='r', label='1')
plt.ylabel('银行利润')
plt.xlabel('游戏回合')
plt.legend(loc='best')

b_y_total = [0]
for i in b_y[0]:
    b_y_total.append(b_y_total[len(b_y_total)-1] + i)

plt.figure('银行利润总和')
plt.plot(np.array(b_y_total), c='r', label='1')
plt.ylabel('银行利润总和')
plt.xlabel('游戏回合')
plt.legend(loc='best')

plt.figure('游戏结束时机器存货')
plt.plot(np.array(K_X), c='r', label='1')
plt.ylabel('游戏结束时机器存货')
plt.xlabel('游戏回合')
plt.legend(loc='best')

plt.figure('KNP')
plt.plot(np.array(K_NP), c='r', label='1')
plt.ylabel('KNP')
plt.xlabel('游戏回合')
plt.legend(loc='best')

plt.figure('LNP')
plt.plot(np.array(L_NP), c='r', label='1')
plt.ylabel('LNP')
plt.xlabel('游戏回合')
plt.legend(loc='best')

plt.figure('游戏结束时粮食存货')
plt.plot(np.array(L_X), c='r', label='1')
plt.ylabel('游戏结束时粮食存货')
plt.xlabel('游戏回合')
plt.legend(loc='best')

plt.figure('K1')
plt.plot(np.array(K_desired[0]), c='r', label='1')
plt.ylabel('K1')
plt.xlabel('游戏回合')
plt.legend(loc='best')

plt.figure('K2')
plt.plot(np.array(K_desired[1]), c='r', label='1')
plt.ylabel('K2')
plt.xlabel('游戏回合')
plt.legend(loc='best')

plt.figure('L1')
plt.plot(np.array(L_desired[0]), c='r', label='1')
plt.ylabel('L1')
plt.xlabel('游戏回合')
plt.legend(loc='best')

plt.figure('L2')
plt.plot(np.array(L_desired[1]), c='r', label='1')
plt.ylabel('L2')
plt.xlabel('游戏回合')
plt.legend(loc='best')

plt.figure('WNDF1')
plt.plot(np.array(WNDF1), c='r', label='1')
plt.ylabel('WNDF1')
plt.xlabel('游戏回合')
plt.legend(loc='best')

plt.figure('WNDF2')
plt.plot(np.array(WNDF2), c='r', label='1')
plt.ylabel('WNDF2')
plt.xlabel('游戏回合')
plt.legend(loc='best')

plt.figure('WNDB1')
plt.plot(np.array(WNDB1), c='r', label='1')
plt.ylabel('WNDB1')
plt.xlabel('游戏回合')
plt.legend(loc='best')

plt.figure('WNDB2')
plt.plot(np.array(WNDB2), c='r', label='1')
plt.ylabel('WNDB2')
plt.xlabel('游戏回合')
plt.legend(loc='best')

plt.figure('real_WNDB1')
plt.plot(np.array(real_WNDB1), c='r', label='1')
plt.ylabel('real_WNDB1')
plt.xlabel('游戏回合')
plt.legend(loc='best')

plt.figure('real_WNDB2')
plt.plot(np.array(real_WNDB2), c='r', label='1')
plt.ylabel('real_WNDB2')
plt.xlabel('游戏回合')
plt.legend(loc='best')

plt.figure('游戏结束时余额')
plt.plot(np.array(final_M[0]), c='r', label='1')
plt.plot(np.array(final_M[1]), c='b', label='2')
plt.ylabel('游戏结束时余额')
plt.xlabel('游戏回合')
plt.legend(loc='best')

plt.figure('游戏结束时欠款')
plt.plot(np.array(final_D[0]), c='r', label='1')
plt.plot(np.array(final_D[1]), c='b', label='2')
plt.ylabel('游戏结束时欠款')
plt.xlabel('游戏回合')
plt.legend(loc='best')

plt.figure('游戏结束时支付总利息')
plt.plot(np.array(total_iD[0]), c='r', label='1')
plt.plot(np.array(total_iD[1]), c='b', label='2')
plt.ylabel('游戏结束时支付总利息')
plt.xlabel('游戏回合')
plt.legend(loc='best')

plt.figure('游戏结束时总收入')
plt.plot(np.array(total_R[0]), c='r', label='1')
plt.plot(np.array(total_R[1]), c='b', label='2')
plt.ylabel('游戏结束时总收入')
plt.xlabel('游戏回合')
plt.legend(loc='best')

plt.figure('游戏结束时总支出')
plt.plot(np.array(total_C[0]), c='r', label='1')
plt.plot(np.array(total_C[1]), c='b', label='2')
plt.ylabel('游戏结束时总支出')
plt.xlabel('游戏回合')
plt.legend(loc='best')

for i in range(1,len(e_td_err1)):
    e_td_err1[i] = e_td_err1[i] + e_td_err1[i-1]

for i in range(1,len(e_td_err2)):
    e_td_err2[i] = e_td_err2[i] + e_td_err2[i-1]

for i in range(1,len(b_td_err)):
    b_td_err[i] = b_td_err[i] + b_td_err[i-1]

plt.figure('product_td_err_full')
plt.plot(np.array(e_td_err1), c='r', label='1')
plt.ylabel('product_td_err')
plt.xlabel('游戏天数')
plt.legend(loc='best')

plt.figure('consume_td_err_full')
plt.plot(np.array(e_td_err2), c='r', label='1')
plt.ylabel('consume_td_err')
plt.xlabel('游戏天数')
plt.legend(loc='best')

plt.figure('bank_td_err_full')
plt.plot(np.array(b_td_err), c='r', label='1')
plt.ylabel('bank_td_err')
plt.xlabel('游戏天数')
plt.legend(loc='best')

plt.figure('real_WNDB2')
plt.plot(np.array(real_WNDB2), c='r', label='1')
plt.ylabel('real_WNDB2')
plt.xlabel('游戏回合')
plt.legend(loc='best')

plt.figure('real_WNDB2')
plt.plot(np.array(real_WNDB2), c='r', label='1')
plt.ylabel('real_WNDB2')
plt.xlabel('游戏回合')
plt.legend(loc='best')

plt.figure('hist_consume_action_WNDF')
plt.hist(action_count[0][0],10)

plt.figure('hist_consume_action_K')
plt.hist(action_count[0][1],10)

plt.figure('hist_consume_action_L')
plt.hist(action_count[0][2],10)

plt.figure('hist_consume_action_P')
plt.hist(action_count[0][3],10)

plt.figure('hist_product_action_WNDF')
plt.hist(action_count[1][0],10)

plt.figure('hist_produc_action_K')
plt.hist(action_count[1][1],10)

plt.figure('hist_produc_action_L')
plt.hist(action_count[1][2],10)

plt.figure('hist_produc_action_P')
plt.hist(action_count[1][3],10)

plt.figure('hist_bank_action_WNDF')
plt.hist(b_action_count[0],10)

plt.figure('hist_bank_action_K')
plt.hist(b_action_count[0],10)

plt.show()
