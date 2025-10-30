from .Enterprise import Enterprise
from .Bank_config import Bank_config
import math
import random
M = 0  # 现金（等价于利润）
Ω = 1  # Ω 分别为一、二类企业债券 数据格式为列表
D = 2  # D 分别为银行对一、二类企业的欠款 数据格式为列表
WNDB = 3  # 分别为一、二类企业贷款意愿 数据格式为列表
real_WNDB = 4  # 分别为一、二类企业执行侧贷款意愿 数据格式为列表
OB = 5  # 分别为一、二类企业观察，对应数据项应为列表，是企业观察，对应数据为e_data
bank_current = 6
bank_current = 6
class Bank:
    def __init__(self,
                 config:Bank_config
                 ):                          # 字典数据全部以 （主体名称，数据内容）格式存储
        self.config = config
        self.name = config.name
        self.money = 0                       # M 银行现金，即利润总和
        self.profit = 0                      # 利润，等价于当前回合收回利息数，或破产后减少的金额
        self.total_profit = 0
        self.able_fund = 0
        self.debt = {}                       # D 银行对企业的欠款，即 企业现金
        self.bond = {}                       # Ω 银行对企业的债券，即 企业欠银 行的钱
        self.bond_detail = {}                # 待还款细则 为{(主体名:str,list[debt_time])}
        self.should_payback = {}             # 当日待还款数值 {(主体名:str,待还款:float)}

        self.WNDB = {}                       # 该回合银行决策对企业的放贷
        self.real_WNDB = {}                  # 该回合银行实际执行对企业的放贷
        self.observation = {}                # 银行对企业的观察，为引用传递，设定完观察无需修改
        self.loss = {}
        self.reward = {}
        self.reward_train = {}
        # 以上为数据，以下为参数
        self.debt_time = config.debt_time
        self.debt_i = config.debt_i
        self.fund = self.config.fund                     # 银行储备金
        self.fund_rate = config.fund_rate           # 储备金率，银行总共可以放出的贷款额度为 sum(bond) <= fund_rate * (fund + money)
        self.fund_increase = config.fund_increase   # 储备金每回合增长，即 fund = fund * (1 + fund_increase) ^ day
        self.action_function = config.action_function
        self.total_reward = {'WNDB': 0, }
        self.reward_decay = 0.95
        self.step = 0
        self.reward_WNDB= 0.0

    # 观察企业的负债 资金 还款 贷款意愿 实际贷款意愿 企业对银行借款详情
    def observe(self, target: Enterprise):
        self.observation[target.name] = target
        self.debt[target.name] = 0
        self.bond[target.name] = 0
        self.WNDB[target.name] = 0
        self.real_WNDB[target.name] = 0
        self.bond_detail[target.name] = [0 for x in range(self.debt_time)]
        self.should_payback[target.name] = 0

    # 新的回合 某个企业破产或达到预制天数上线T 一回合结束 重置环境和智能体的属性
    def new_episode(self):
        self.money = 0  # M 银行现金，即利润总和
        self.profit = 0  # 利润，等价于当前回合收回利息数，或破产后减少的金额
        self.total_profit = 0
        self.able_fund = 0
        self.debt = {}  # D 银行对企业的欠款，即 企业现金
        self.bond = {}  # Ω 银行对企业的债券，即 企业欠银 行的钱
        self.bond_detail = {}  # 待还款细则 为{(主体名:str,list[debt_time])}
        self.should_payback = {}  # 当日待还款数值 {(主体名:str,待还款:float)}
        self.loss = {}
        self.reward = {}
        self.reward_train = {}
        self.total_reward = {'WNDB': 0}
        self.step = 0

        self.WNDB = {}  # 该回合银行决策对企业的放贷
        self.real_WNDB = {}  # 该回合银行实际执行对企业的放贷
        self.fund = self.config.fund                     # 银行储备金

        # keys：企业1，企业2
        for key in self.observation.keys():
            self.debt[key] = 0
            self.bond[key] = 0
            self.WNDB[key] = 0
            self.real_WNDB[key] = 0
            self.bond_detail[key] = [0 for x in range(self.debt_time)]
            self.should_payback[key] = 0

    def new_day(self,day):
        for key in self.observation.keys():
            self.WNDB[key] = 0
            self.real_WNDB[key] = 0
            self.should_payback[key] = 0
        self.develop()
        self.profit = 0  # 利润，等价于当前回合收回利息数，或破产后减少的金额
        self.reward_WNDB=0.0

    def set_action(self,
               target:str,
               action:any,
               actionType: str = 'WNDB'):
        try:
            self.WNDB[target] = self.action_function[actionType](self, target, action)
        except KeyError:    # actionType方法不存在 则直接赋值
            self.WNDB[target] = action

    def check_rent(self):
        self.able_fund = round(self.fund_rate * (self.money + self.fund)-sum(self.bond.values()), 2)
        total_WNDB = sum(self.WNDB.values())
        percent = min(self.able_fund/(total_WNDB + 1e-2), 1)   # 储备金不够总待借出，则按比例
        for key in self.real_WNDB.keys():
            self.real_WNDB[key] = round(self.WNDB[key] * percent, 2)


    def rent(self, name: str, day:int) -> float:
        rent_val = self.real_WNDB[name]
        self.bond_detail[name][day % self.debt_time] = rent_val
        self.debt[name] += rent_val          # 银行给出现金，银行对企业债务加一笔
        self.bond[name] += rent_val          # 银行放出贷款，银行对企业债权加一笔
        if rent_val > 0:
            # 这个系数是需要调试的超参数。它的值需要足以抵消并超过“闲置惩罚”。
            # 例如，如果闲置惩罚是 -0.2，那么这个奖励至少应该是 +0.2 以上才有意义。
            loan_action_reward_coefficient = 0.3
            self.reward_WNDB += loan_action_reward_coefficient
        return rent_val

    def answer_debt(self, name: str, day):   # 回合开始由企业询问该回合待还款本金和待还款利息
        self.should_payback[name] = self.bond_detail[name][day % self.debt_time]  # 记录下当日待还款
        return {'money': round(self.should_payback[name], 2), 'iD':round(self.debt_i * self.bond[name], 2)}

    def deal_payback(self, name:str, payback: dict):   # 处理来自企业的还款
        self.debt[name] = round(self.debt[name] - payback['payback'], 2)      # 银行收回现金，银行对企业债务减一笔
        self.bond[name] = round(self.bond[name] - payback['payback'], 2)      # 银行收回贷款，银行对企业债权减一笔
        self.profit = round(self.profit + payback['iD'], 2)                    # 银行收回利息，银行利润加一笔
        self.total_profit += self.profit
        self.money = round(self.money + payback['iD'], 2)                     # 银行收回利息，银行现金（总利润）加一笔
        self.debt[name] = round(self.debt[name] - payback['iD'], 2)           # 银行收回利息，银行对企业欠款减少

        # 【【【新增】】】利息收入奖励
        # 目标: 激励银行进行有利可图的投资。
        # 每收到一笔利息，就累加一个正奖励。
        scaled_interest_reward = payback['iD'] / 100.0
        # 2. 然后，我们可以用一个系数来调整这个缩放后奖励的权重。
        #    现在因为基础奖励已经很小了，我们可以使用一个较大的系数（例如1.0）来赋予它权重。
        interest_reward_coefficient = 1.0  # 可调试的超参数

        self.reward_WNDB += scaled_interest_reward * interest_reward_coefficient

    def trade_callback(self, name: str, delta_money:float):
        # 企业处理完订单后变动的金钱数量，平衡资产债务表
        self.debt[name] = round(self.debt[name] + delta_money, 2)



    def develop(self):  # 经济发展，储备金增加
        # 根据市面上流动的现金来决定，市面上现金为银行对企业债务之和
        self.fund += sum(self.debt.values()) * self.fund_increase

    #银行状态 自身属性2 + production1状态 33 + cousumption1状态中自己的属性13
    def custom_state(self,day,lim_day):
        # state = [self.money , self.able_fund]
        state = [self.money/100, self.able_fund/1000]
        flag = True
        for key in self.observation.keys():
            if flag:
                ns = self.observation[key].get_state()
                flag = False
            else:
                ns = self.observation[key].get_state()[0:13]
            state = state + ns
        # for key in self.observation.keys():
        #     state = state + self.observation[key].get_state()

        self.state = state

    def get_state(self):
        return self.state

    def custom_reward(self, day, lim_day):
        self.reward['WNDB'] = self.profit / 100
        # 平滑延长奖励：从80%开始增长，最高放大到1.5倍
        self.reward_train = self.reward.copy()
        progress = day / lim_day
        if progress > 0.8:
            bonus_factor = (progress - 0.8) * 0.5  # 在80%→100%之间线性增至0.1倍
            self.reward_train['WNDB'] =(1 + bonus_factor) * self.reward['WNDB']

        # 最终生存奖励
        if day == lim_day -2 :
            self.reward_train['WNDB'] += lim_day * 0.2 / 80 # 额外奖

        return self.reward

    # def custom_reward(self,day,lim_day):
    #     '''
    #     self.reward['WNDB'] = self.profit/1
    #     '''
    #     self.reward['WNDB'] = self.profit/100
    #     if day >= lim_day * 0.95:
    #         self.reward['WNDB'] = self.reward['WNDB'] +(day * 0.1)/100

    # def custom_reward(self):
    #     """
    #     【【【修改后】】】
    #     实现全新的、基于“利息收入”和“闲置资金惩罚”的奖励函数。
    #     注意：利息奖励部分已在 deal_payback 方法中计算并累加到 self.reward_WNDB。
    #     这里我们只计算并累加“闲置资金惩罚”。
    #     """
    #     # --- 计算惩罚项：闲置资金惩罚 ---
    #     # 目标: 惩罚银行囤积现金、不参与经济活动的行为。
    #
    #     # 首先，计算银行的总资产。总资产 = 现金(self.money) + 所有未偿还贷款的本金总额(sum(self.bond.values()))
    #     # 您的代码中 self.bond 记录了所有债权，非常方便。
    #     total_assets = self.money + sum(self.bond.values())
    #
    #     # 防止除零错误
    #     if total_assets > 0:
    #         # 计算现金比例。这里的现金我们用 self.money，它代表银行的利润和自有资金。
    #         cash_ratio = self.money / total_assets
    #
    #         # 如果现金比例超过一个阈值（例如80%），则施加惩罚
    #         idle_threshold = 0.7  # 可调试的超参数
    #         if cash_ratio > idle_threshold:
    #             # 惩罚的大小与超出阈值的程度成正比
    #             idle_penalty_coefficient = -0.2  # 可调试的超参数(负数)
    #             penalty_from_idle_fund = (cash_ratio - idle_threshold) * idle_penalty_coefficient
    #
    #             # 将惩罚累加到本回合的总奖励中
    #             self.reward_WNDB += penalty_from_idle_fund
    #
    #     # 将最终计算出的奖励赋值给 self.reward['WNDB']
    #     self.reward['WNDB'] = self.reward_WNDB/100


    def get_reward(self):
        decay = self.reward_decay ** self.step
        for key in self.total_reward:
            self.total_reward[key] += self.reward[key] * decay
        self.step += 1
        return self.reward_train


    def get_fail_reward(self):
        fail_reward = {'WNDB':0}
        decay = self.reward_decay ** self.step
        # for key in self.observation:
        #     if self.observation[key].is_falled():
        #         fail_reward['WNDB'] -= self.bond[key]
        fail_reward['WNDB'] = -10
        self.total_reward['WNDB'] += fail_reward['WNDB'] * decay
        return fail_reward


    def final_settlement(self):
        for key in self.observation:
            # 如果某企业破产
            # 因为还款时现金已经清零了
            # 此时理论上银行对企业债务debt为0
            # 此时剩余的银行对企业的债权bond，就是企业欠银行且还不上的贷款
            # 这部分需要银行自掏腰包
            if self.observation[key].is_falled():
                delta_money = self.bond[key]
                self.profit -= delta_money
                self.total_profit += self.profit
                self.money -= delta_money



    def __str__(self):
        res = self.name + \
                    "\n 银行利润：" + str(self.money) +\
                    " 银行债券：" + str(self.bond) +\
                    " 银行欠债：" + str(self.debt) +\
                    " 银行贷款意愿：" + str(self.WNDB) + \
                    " 银行实际贷款意愿：" + str(self.real_WNDB) + \
                    "\n 储备金:" + str(self.fund_rate * (self.money + self.fund)) +\
                    " 剩余可用储备金:" + str(round(self.fund_rate * (self.money + self.fund)-sum(self.bond.values()), 2)) + \
                    " 借款详情:" + str(self.bond_detail) +'\n'
        for key in self.real_WNDB:
            res = res + '银行实际借给' + str(key) + " " + str(self.real_WNDB[key]) + " 的贷款\n"
        return res





