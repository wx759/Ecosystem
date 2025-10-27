import numpy as np
import random
import matplotlib.pyplot as plt
import math
from  enum import  IntEnum

class E_DATA(IntEnum):
    # 输入数据
    M = 0         # 现金（与银行D平账）
    X = 1         # 存货
    D = 2         # 负债本金（与银行Ω平账）
    R = 3         # 收入
    iD = 4        # 负债利息（加入银行现金，偿还利息时减少的本金应对应减少银行的D）
    C = 5         # 成本支出
    π = 6         # 利润
    P = 7         # 市场价，即另一企业的NP值
    # 输出数据
    WNDF = 8      # 追加贷款意愿，即第二日希望追加多少的贷款
    K = 9         # 当日决策购买机器
    L = 10        # 当日决策雇佣劳动力
    k = 11        # 根据当日所持金和定价，执行机构最终决定购买机器的数量
    l = 12        # 根据当日所持金和定价，执行机构最终决定雇佣劳动力的数量
    getK = 13     # 当日实际获得机器
    getL = 14     # 当日实际获得劳动力
    NP = 15      # 次日定价
    total_π = 16
    total_C = 17
    total_R = 18
    total_iD = 19
    current_product = 20
    get_WNDF = 21
    current_sell = 22
e_column = ["现金","存货","负债","收入","利息","成本","利润","市场价","贷款意愿","决策机器","决策劳动力","执行机器","执行劳动力","获得机器","获得劳动力","定价","总利润","总支出","总收入","总利息","产量","新获得贷款","实际交易量"]
class B_DATA(IntEnum):
    M = 0         # 现金（等价于利润）
    Ω = 1         # Ω 分别为一、二类企业债券 数据格式为列表
    D = 2         # D 分别为银行对一、二类企业的欠款 数据格式为列表
    WNDB = 3      # 分别为一、二类企业贷款意愿 数据格式为列表
    real_WNDB = 4 # 分别为一、二类企业执行侧贷款意愿 数据格式为列表
    OB = 5       # 分别为一、二类企业观察，对应数据项应为列表，是企业观察，对应数据为e_data

b_column = ["现金","生产企业意愿借出","消费企业意愿借出","生产企业执行侧意愿借出","消费企业执行侧意愿借出"]

def is_list(d):
    return isinstance(d, list)


# =================================================================== 业务相关方法 ======================================

γ = 0.95
act_num = 5
b_act_num = 11
bank_base = 1000 #银行储备金
bank_γ = 1 # 可放出贷款总和 = bank_base + bank_γ * 银行现金M
act_percent = [-0.9,-0.05, 0,  0.05,0.9]
WBDF_percent = [0.8,0.6,0.4, 0.2, 0]
b_percent = [1,0.9,0.8,0.7,0.6,0.5,0.4,0.3,0.2, 0.1, 0]
is_over_stack = False


def set_init(e_data):
    for data in e_data:
        data[E_DATA.total_C.value] = γ*data[E_DATA.total_C.value] + (1-γ)*data[E_DATA.C.value]
        data[E_DATA.total_R.value] = γ*data[E_DATA.total_R.value] + (1-γ)*data[E_DATA.R.value]
        data[E_DATA.total_π.value] = data[E_DATA.total_π.value] + data[E_DATA.π.value]
        data[E_DATA.total_iD.value] += data[E_DATA.iD.value]
        data[E_DATA.π.value] = -data[E_DATA.C.value]  # 前一天的成本，这样之后运算时就会 利润 = 今日的收益 - 昨日的成本
        data[E_DATA.R.value] = 0
        data[E_DATA.C.value] = 0



# 返回债务清算结果 银行现金增加只有还利息

# 输入参数 b_data 目标银行数据，e_mod 为待偿还企业目标债券编号，e_data 企业数据， d 天数，debt_T 还款周期，debt_record，债务记录，Debt_i 利率
# 返回参数 破产为 True，否则 False
# 流程 偿还利息→偿还本金
def set_debt(b_data, e_mod, e_data, d, debt_T, debt_record, Debt_i):
    if is_list(e_mod):
        isFall  = [None for i in e_mod]
        isDown = False
        for i in e_mod:
            isFall[i] = set_debt(b_data, i, e_data, d, debt_T, debt_record, Debt_i)
            isDown = isFall[i]  or isDown
        return isFall,isDown  # 即使有企业破产，也要将所有mod结算完成再退出

    iD = Debt_i * e_data[e_mod][E_DATA.D.value]                                # 待偿还的利息
    debt = debt_record[e_mod][d % debt_T]                                             # N天前的借款，即待偿还本金
    # print(debt_record[e_mod])
    if e_data[e_mod][E_DATA.M.value] < debt + iD:                              # 企业现金无法偿还债务
        b_data[B_DATA.M.value] -= (debt + iD - e_data[e_mod][E_DATA.M.value])  # 银行需要用自身利润填补烂账带来的损失
        e_data[e_mod][E_DATA.M.value] -= debt + iD                             # 破产金额变负数的情况
        # b_data[B_DATA.D.value][e_mod] = 0                                    # 破产金额不变负数而是清零的情况 连带下面四句一起
        # b_data[B_DATA.Ω.value][e_mod] = 0
        # e_data[e_mod][E_DATA.D.value] = 0
        # e_data[e_mod][E_DATA.M.value] = 0                                      # 之后便一笔勾销，企业破产
        return True
    # ==================利息结算部分=============================
    e_data[e_mod][E_DATA.iD.value] = iD    # 记录利息
    e_data[e_mod][E_DATA.M.value] -= iD    # 偿还利息
    b_data[B_DATA.M.value] += iD           # 利息收入算入现金中
    b_data[B_DATA.D.value][e_mod] -= iD    # 平账，利息偿还给银行后，银行欠企业的钱同等减少

    # ==================本金结算部分=============================
    # 此环节银行没有盈利，M不变
    e_data[e_mod][E_DATA.M.value] -= debt  # 偿还本金 现金减少
    b_data[B_DATA.D.value][e_mod] -= debt  # 平账，对应银行欠企业的钱减少
    e_data[e_mod][E_DATA.D.value] -= debt  # 偿还本金 债务减少
    b_data[B_DATA.Ω.value][e_mod] -= debt  # 平账，对应的银行对企业的债权减少

    debt_record[e_mod][d%debt_T] = debt_record[e_mod][debt_T]

    return False


# 银行给企业放贷
# 输入参数 b_data 目标银行数据，e_mod 为待借款企业编号，e_data 企业数据， d 天数，debt_T 还款周期，debt_record，债务记录，Debt_i 利率
def rent_money(b_data, e_mod, e_data, d, debt_T, debt_record):
    if is_list(e_mod):
        for i in e_mod:
            rent_money(b_data, i, e_data, d, debt_T, debt_record)
        return
    ND = min(b_data[B_DATA.real_WNDB.value][e_mod], e_data[e_mod][E_DATA.WNDF.value])  # 贷款额度为银行可贷款和企业意愿贷款的最小值
    e_data[e_mod][E_DATA.get_WNDF.value] = ND
    # ND = b_data[B_DATA.WNDB.value][e_mod] * e_data[e_mod][E_DATA.WNDF.value]  # 贷款额度为银行根据报价给出的百分比
    e_data[e_mod][E_DATA.M.value] += ND  # 企业现金增加贷款额度
    b_data[B_DATA.D.value][e_mod] += ND  # 银行平账增加欠债
    e_data[e_mod][E_DATA.D.value] += ND  # 企业债务增加贷款额度
    b_data[B_DATA.Ω.value][e_mod] += ND  # 银行平账增加债券
    # debt_record[e_mod][d % debt_T] = ND         # T天后还债
    debt_record[e_mod][debt_T] = ND         # 存入temp中


# 交易前对决策数据进行审核
# 如果金钱不足以购买全部商品 将以k/l = K/L 的比例获取实际购买数量
# self_shop,target_shop 商品编号(K或L)，用来查看NP和P分别对应什么产品, self_design target_design同理 传参时不加.value,以此提高识别度

def trade_check(e_data, e_mod, self_shop, target_shop, self_design, target_design):
    if is_list(e_mod):
        for i in e_mod:
            trade_check(e_data,i,self_shop,target_shop,self_design,target_design)
        return;
    data = e_data[e_mod]
    M = data[E_DATA.M.value]
    # 现金够 则决策多少就够买多少
    if (data[self_shop.value] * data[E_DATA.NP.value] + data[target_shop.value] * data[E_DATA.P.value] ) <= M:
        data[self_design.value] = data[self_shop.value]
        data[target_design.value] = data[target_shop.value]
    # 现金不够 则等比分配
    else:
        percent = M/(data[self_shop.value] * data[E_DATA.NP.value] + data[target_shop.value] * data[E_DATA.P.value])
        data[self_design.value] = percent * data[self_shop.value]
        data[target_design.value] = percent * data[target_shop.value]



# 企业交易
# 输入参数 b_data 银行数据，seller 卖家，独一家，buyer 买家，可以有多家， e_data 企业数据
#          shop 商品编号(K或L)，用来决定交易机器还是人力, get 实际获得商品的数组下标编号（getK或getL) 传参时不加.value,以此提高识别度
def trade(b_data, seller, buyer, e_data, shop, get_shop):
    if not is_list(buyer):
        buyer = [buyer]
    wanted = [None for i in buyer]                          # 通过动作决策得出的希望购买的数量
    real_get = [None for i in buyer]                      # 根据剩余库存分配出的实际的购买数量，库存足够则足额分配，不足时等比分配
    X = e_data[seller][E_DATA.X.value]                        # 实际库存数量
    NP = e_data[seller][E_DATA.NP.value]                      # 售价
    for i in buyer:
        wanted[i] = e_data[i][shop.value]
    # ==========================成交过程，处理买家=================================
    for i in buyer:
        real_get[i] = min(wanted[i], wanted[i]/sum(wanted) * X)
        e_data[i][get_shop.value] = real_get[i]        # 储存实际购买数据
        # 用前一天的消费作为成本
        e_data[i][E_DATA.C.value] += real_get[i] * NP  # 计入成本
        e_data[i][E_DATA.M.value] -= real_get[i] * NP  # 交钱
        b_data[B_DATA.D.value][i] -= real_get[i] * NP  # 银行平账
    # ==========================结算过程，处理卖家=================================
    sum_get = sum(real_get)
    e_data[seller][E_DATA.current_sell.value] = sum_get
    e_data[seller][E_DATA.X.value] = max(0, X-sum_get)  # 库存减少，防止因为数据精度造成负数库存
    e_data[seller][E_DATA.M.value] += sum_get * NP      # 买家金钱增加
    b_data[B_DATA.D.value][seller] += sum_get * NP              # 银行平账
    e_data[seller][E_DATA.R.value] += sum_get * NP      # 买家将销售额计入收入


def show_bank(b_data):
    print(" 银行利润：" + str(b_data[B_DATA.M.value]) +
          " 银行债券：" + str(b_data[B_DATA.Ω.value]) +
          " 银行欠债：" + str(b_data[B_DATA.D.value]) +
          " 银行贷款意愿：" + str(b_data[B_DATA.WNDB.value]))


def show_enterprise(e_data, name):
    print(name)
    print(" 企业现金：" + str(e_data[E_DATA.M.value]) +
          " 企业存货：" + str(e_data[E_DATA.X.value]) +
          " 企业欠债：" + str(e_data[E_DATA.D.value]) +
          " 企业收入：" + str(e_data[E_DATA.R.value]) +
          " 企业利息：" + str(e_data[E_DATA.iD.value]) +
          " 企业成本：" + str(e_data[E_DATA.C.value]) +
          " 企业利润：" + str(e_data[E_DATA.π.value]) +
          " 市场价：" + str(e_data[E_DATA.P.value]) +
          " 企业贷款意愿：" + str(e_data[E_DATA.WNDF.value]) +
          " 企业机器：" + str(e_data[E_DATA.K.value]) +
          " 企业工人：" + str(e_data[E_DATA.L.value]) +
          " 企业可承受机器：" + str(e_data[E_DATA.k.value]) +
          " 企业可承受工人：" + str(e_data[E_DATA.l.value]) +
          " 企业得到机器：" + str(e_data[E_DATA.getK.value]) +
          " 企业得到工人：" + str(e_data[E_DATA.getL.value]) +
          " 企业定价：" + str(e_data[E_DATA.NP.value]) +
          " 企业总成本：" + str(e_data[E_DATA.total_C.value]) +
          " 企业总收入：" + str(e_data[E_DATA.total_R.value]) +
          " 企业总利润：" + str(e_data[E_DATA.total_π.value])
          )


def set_data_round(data):
    for i in range(len(data)):
        if not is_list(data[i]):
            data[i] = np.round(data[i], 3)
        else:
            set_data_round(data[i])
    return data




def daily_settlement(b_data, e_data, t, d):
    for i in range(len(e_data)):
        e = e_data[i]
        # e_data[i][E_DATA.π.value] = e[E_DATA.R.value] - e[E_DATA.C.value] - e[E_DATA.iD.value]  # 利润＝收入-生产成本-利息
        e_data[i][E_DATA.π.value] += e[E_DATA.R.value] - e[E_DATA.iD.value]  # 利润＝收入-昨天生产成本-利息  （成本开头扣除）
        b_data[B_DATA.OB.value][i] = e_data[i]
    set_data_round(b_data)
    set_data_round(e_data)



def show_daily(b_data, e_data, t, d):
    print("第" + str(t) + "个回合第" + str(d) + "天")
    show_bank(b_data)
    show_enterprise(e_data[0], "第一类企业")
    show_enterprise(e_data[1], "第二类企业")


def csv_kuaizhao(b_data, e_data, t, d,debt_t,e_product_mod,e_consume_mod,csv_ouput):
    csv_data = []  #
    csv_data.append(t)
    csv_data.append(d)
    csv_data.append(e_data[e_product_mod[0]][E_DATA.M.value])
    csv_data.append(e_data[e_consume_mod[0]][E_DATA.M.value])
    csv_data.append(b_data[B_DATA.M.value])
    csv_data.append(e_data[e_product_mod[0]][E_DATA.K.value])
    csv_data.append(e_data[e_consume_mod[0]][E_DATA.K.value])
    csv_data.append(e_data[e_product_mod[0]][E_DATA.L.value])
    csv_data.append(e_data[e_consume_mod[0]][E_DATA.L.value])
    csv_data.append( produce(e_data[e_product_mod[0]][E_DATA.K.value],e_data[e_product_mod[0]][E_DATA.L.value]) )
    csv_data.append( produce(e_data[e_consume_mod[0]][E_DATA.K.value],e_data[e_product_mod[0]][E_DATA.L.value]) )
    csv_data.append(e_data[e_product_mod[0]][E_DATA.NP.value])
    csv_data.append(e_data[e_consume_mod[0]][E_DATA.NP.value])
    csv_ouput.append(csv_data)

def csv_update(b_data, e_data, t, d,debt_t,e_product_mod,e_consume_mod,csv_ouput):

    pos = len(csv_ouput) - 1
    csv_ouput[pos].append(e_data[e_product_mod[0]][E_DATA.NP.value])
    csv_ouput[pos].append(e_data[e_consume_mod[0]][E_DATA.NP.value])
    csv_ouput[pos].append(e_data[e_product_mod[0]][E_DATA.current_product.value])
    csv_ouput[pos].append(e_data[e_consume_mod[0]][E_DATA.current_product.value])
    csv_ouput[pos].append(e_data[e_product_mod[0]][E_DATA.current_sell.value])
    csv_ouput[pos].append(e_data[e_consume_mod[0]][E_DATA.current_sell.value])

    for i in E_DATA:
        csv_ouput[pos].append(e_data[0][i.value])
        csv_ouput[pos].append(e_data[1][i.value])

    csv_ouput[pos].append(b_data[B_DATA.M.value])
    csv_ouput[pos].append(b_data[B_DATA.WNDB.value][0])
    csv_ouput[pos].append(b_data[B_DATA.WNDB.value][1])
    csv_ouput[pos].append(b_data[B_DATA.real_WNDB.value][0])
    csv_ouput[pos].append(b_data[B_DATA.real_WNDB.value][1])

    csv_ouput[pos].append(debt_t)
    csv_ouput[pos].append(bank_base)
    csv_ouput[pos].append(bank_γ)

    able_new_Ω = bank_γ * (bank_base + b_data[B_DATA.M.value]) - b_data[B_DATA.Ω.value][0] - \
                 b_data[B_DATA.Ω.value][1]
    csv_ouput[pos].append(able_new_Ω)

    if sum(b_data[B_DATA.Ω.value]) > bank_base * bank_γ:
        csv_ouput[pos].append(1)
    else:
        csv_ouput[pos].append(0)

    if e_data[e_product_mod[0]][E_DATA.current_sell.value]>e_data[e_product_mod[0]][E_DATA.current_product.value]:
        csv_ouput[pos].append(1)
    else:
        csv_ouput[pos].append(0)

    if e_data[e_consume_mod[0]][E_DATA.current_sell.value]>e_data[e_consume_mod[0]][E_DATA.current_product.value]:
        csv_ouput[pos].append(1)
    else:
        csv_ouput[pos].append(0)


# 企业生产
# 输入数据:e_data 企业数据 e_mod 欲生产的企业编号
def product(e_data, e_mod):
    if is_list(e_mod):
        for i in e_mod:
            product(e_data, i)
        return
    e_data[e_mod][E_DATA.current_product.value] = produce(e_data[e_mod][E_DATA.getK],e_data[e_mod][E_DATA.getL])  # 修改点
    e_data[e_mod][E_DATA.X.value] += e_data[e_mod][E_DATA.current_product.value]

# =================================================================== 数据处理相关方法 =================================

def produce(K, L):
    return 2.3 * (K ** 0.5) * (L ** 0.5)


def update_B_after_E_action(b_data,e_data):
    for i in range(len(e_data)):
        b_data[B_DATA.OB.value][i] = e_data[i]

# 处理银行数据，返回可用于神经网络的组合
def set_b_state(b_data,b_mod):
    if is_list(b_mod):
        res = []
        for i in b_mod:
            res.append(set_b_state(b_data, i))
        return res
    e_data_0 = b_data[B_DATA.OB.value][0]
    e_data_1 = b_data[B_DATA.OB.value][1]
    return list([b_data[B_DATA.M.value], b_data[B_DATA.Ω.value][0], b_data[B_DATA.Ω.value][1], b_data[B_DATA.D.value][0],
                 b_data[B_DATA.D.value][1], b_data[B_DATA.WNDB.value][0], b_data[B_DATA.WNDB.value][1],b_data[B_DATA.real_WNDB.value][0], b_data[B_DATA.real_WNDB.value][1],
                 e_data_0[E_DATA.WNDF.value], e_data_1[E_DATA.WNDF.value], e_data_0[E_DATA.π.value], e_data_1[E_DATA.π.value],
                 e_data_0[E_DATA.X.value],e_data_1[E_DATA.X.value],e_data_0[E_DATA.iD.value],e_data_1[E_DATA.iD.value],
                 e_data_0[E_DATA.NP.value],e_data_1[E_DATA.NP.value]])


# 处理企业数据，返回可用于神经网3
def set_e_state(e_data, e_mod):
    if is_list(e_mod):
        res = []
        for i in e_mod:
            res.append(set_e_state(e_data, i))
        return res
    state = e_data[e_mod]
    return list([state[E_DATA.M.value], state[E_DATA.X.value], state[E_DATA.D.value], state[E_DATA.R.value],
                 state[E_DATA.iD.value], state[E_DATA.C.value], state[E_DATA.π.value], state[E_DATA.total_π.value],
                 state[E_DATA.P.value], state[E_DATA.K.value], state[E_DATA.L.value],state[E_DATA.WNDF.value],
                 state[E_DATA.NP.value], state[E_DATA.getK.value], state[E_DATA.getL.value], state[E_DATA.k.value], state[E_DATA.l.value]])


# 给数据设置决策动作
# e_state: [[state1.1,state1.2,...,],[state2.1,state2.2,...,]]
def set_e_action_type(e_state, type=0):
    state = e_state.copy()
    for i in range(len(state)):
        state[i][9] = type
        state[i] = np.array(state[i])
    return state


# 将决策得出的b动作存入b_data
# b_action为[WNDB1, WNDB2 ... WNDBn]
def set_b_action(b_data, b_action):
    for i in range(len(b_action)):
        b_data[B_DATA.WNDB.value][i] = b_action[i]
    return

def set_b_action_percent(b_data,e_data, b_action, e_product_mod, e_consume_mod,day):
    # 决策所得为企业报价 * 百分比
    action = b_action[0]
    b_data[B_DATA.WNDB.value][e_product_mod[0]] = e_data[e_product_mod[0]][E_DATA.WNDF.value] * b_percent[action%b_act_num]
    action //= b_act_num
    b_data[B_DATA.WNDB.value][e_consume_mod[0]] = e_data[e_product_mod[0]][E_DATA.WNDF.value] * b_percent[action%b_act_num]
    if day == 0: # 第一天强制借贷
        b_data[B_DATA.real_WNDB][e_product_mod[0]] = 100
        b_data[B_DATA.real_WNDB][e_consume_mod[0]] = 100
        return
    # 剩余可新放出的贷款
    able_new_Ω =  bank_γ * (bank_base + b_data[B_DATA.M.value]) - b_data[B_DATA.Ω.value][e_product_mod[0]] - b_data[B_DATA.Ω.value][e_consume_mod[0]]
    print("able_new_Ω: ",able_new_Ω)
    # 剩余大于决策
    if able_new_Ω > b_data[B_DATA.WNDB.value][e_product_mod[0]] + b_data[B_DATA.WNDB.value][e_consume_mod[0]]:
        b_data[B_DATA.real_WNDB][e_product_mod[0]] = b_data[B_DATA.WNDB.value][e_product_mod[0]]
        b_data[B_DATA.real_WNDB][e_consume_mod[0]] = b_data[B_DATA.WNDB.value][e_consume_mod[0]]
    # 剩余不足则平分
    else:
        percent = able_new_Ω / (b_data[B_DATA.WNDB.value][e_product_mod[0]] + b_data[B_DATA.WNDB.value][e_consume_mod[0]])
        b_data[B_DATA.real_WNDB][e_product_mod[0]] = percent * b_data[B_DATA.WNDB.value][e_product_mod[0]]
        b_data[B_DATA.real_WNDB][e_consume_mod[0]] = percent * b_data[B_DATA.WNDB.value][e_consume_mod[0]]

    return


Reverse = False


# 将决策得出的e动作存入e_data
# e_action为[WNDF, K, L, NP] other_action:另一个企业的动作，主要用于复制市场价P
def set_e_action(e_data, e_mod, e_action, first_day):
    if is_list(e_mod):
        for i in e_mod:
            set_e_action(e_data, i, e_action, first_day)
        return

    if first_day:
        e_data[e_mod][E_DATA.WNDF.value] = 100
    else:
        e_data[e_mod][E_DATA.WNDF.value] = (act_num - e_action[0] % act_num) * 20 if Reverse else(e_action[0] % act_num) * 20
    e_action[0] //= act_num
    e_data[e_mod][E_DATA.K.value] = (act_num - e_action[0] % act_num) * 1 if Reverse else (e_action[0] % act_num) * 1
    e_action[0] //= act_num
    e_data[e_mod][E_DATA.L.value] = (act_num - e_action[0] % act_num) * 1 if Reverse else (e_action[0] % act_num) * 1
    e_action[0] //= act_num


def set_e_action_percent(e_data, e_mod, e_action, first_day):
    if is_list(e_mod):
        for i in e_mod:
            set_e_action_percent(e_data, i, e_action, first_day)
        return

    if first_day:
        e_data[e_mod][E_DATA.WNDF.value] = 100
        e_action[0]//=act_num
        e_data[e_mod][E_DATA.K.value] = e_action[0] + 1
        e_action[0]//=act_num
        e_data[e_mod][E_DATA.L.value] = e_action[0] + 1
        e_action[0]//=act_num
    else:
        e_data[e_mod][E_DATA.WNDF.value] = e_data[e_mod][E_DATA.M.value] * (WBDF_percent[e_action[0] % act_num])
        e_action[0] //= act_num
        e_data[e_mod][E_DATA.K.value] = e_data[e_mod][E_DATA.K.value] * (1+act_percent[e_action[0] % act_num])
        # e_data[e_mod][E_DATA.K.value] = max(1,e_data[e_mod][E_DATA.K.value])
        e_action[0] //= act_num
        e_data[e_mod][E_DATA.L.value] = e_data[e_mod][E_DATA.L.value] * (1+act_percent[e_action[0] % act_num])
        # e_data[e_mod][E_DATA.L.value] = max(1,e_data[e_mod][E_DATA.L.value])
        e_action[0] //= act_num



def code_action(e_action):
    res = 0
    for action in e_action:
        res *= act_num
        res += action[0]
    return [res]

def set_e_P(e_data, e_mod, NP ):
    if is_list(e_mod):
        for i in e_mod:
            set_e_P(e_data, i, NP)
        return

    e_data[e_mod][E_DATA.NP.value] = NP[0]
    e_data[1-e_mod][E_DATA.P.value] = e_data[e_mod][E_DATA.NP.value]

#  公布第二天的市场价
def set_e_P_percent(e_data, e_mod, NP , first_day):
    if is_list(e_mod):
        for i in e_mod:
            set_e_P_percent(e_data, i, NP, first_day)
        return
    if first_day or e_data[e_mod][E_DATA.NP.value] == 0.0:
        e_data[e_mod][E_DATA.NP.value] = NP[0] + 1
        e_data[1 - e_mod][E_DATA.P.value] = NP[0] + 1
    else:
        e_data[e_mod][E_DATA.NP.value] =e_data[e_mod][E_DATA.NP.value] * (1+act_percent[NP[0]])
        # e_data[e_mod][E_DATA.NP.value] = max(1,e_data[e_mod][E_DATA.NP.value])
        e_data[1-e_mod][E_DATA.P.value] = e_data[e_mod][E_DATA.NP.value]



# 检测是否有库存剩余，如果有人库存为0，游戏无法再进行
def StoreOver(e_data):
    for data in e_data:
        if data[E_DATA.X.value] == 0:
            return True
    return False

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
