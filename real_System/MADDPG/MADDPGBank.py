from real_System.MADDPG.MADDPGEnterprise import Enterprise
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
                 name: str,
                 fund: float = 100,
                 fund_rate: float = 1,
                 fund_increase: float = 0,
                 debt_time: int = 6,
                 debt_i: float = 0.005,
                 is_random:bool = False,
                 agent: any = None,
                 agent_config: any = None,
                 action_function: dict = {},  # 类型为字典 str:function()，决定设置各变量的方法
                 ):                          # 字典数据全部以 （主体名称，数据内容）格式存储
        self.name = name
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

        # 以上为数据，以下为参数

        self.debt_time = debt_time
        self.debet_i = debt_i
        self.base_fund = fund
        self.fund = self.base_fund                     # 银行储备金
        self.fund_rate = fund_rate           # 储备金率，银行总共可以放出的贷款额度为 sum(bond) <= fund_rate * (fund + money)
        self.fund_increase = fund_increase   # 储备金每回合增长，即 fund = fund * (1 + fund_increase) ^ day
        self.Agent = None
        self.agent_factory = agent
        self.agent_config = agent_config
        self.action_function = action_function
        self.is_random = is_random
        self.total_reward = {'WNDB': 0, }
        self.reward_decay = 0.95
        self.step = 0

    def observe(self, target: Enterprise):
        self.observation[target.name] = target
        self.debt[target.name] = 0
        self.bond[target.name] = 0
        self.WNDB[target.name] = 0
        self.real_WNDB[target.name] = 0
        self.bond_detail[target.name] = [0 for x in range(self.debt_time)]
        self.should_payback[target.name] = 0

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
        self.total_reward = {'WNDB': 0}
        self.step = 0

        self.WNDB = {}  # 该回合银行决策对企业的放贷
        self.real_WNDB = {}  # 该回合银行实际执行对企业的放贷
        self.fund = self.base_fund                     # 银行储备金

        for key in self.observation.keys():
            self.debt[key] = 0
            self.bond[key] = 0
            self.WNDB[key] = 0
            self.real_WNDB[key] = 0
            self.bond_detail[key] = [0 for x in range(self.debt_time)]
            self.should_payback[key] = 0

    def new_day(self):
        for key in self.observation.keys():
            self.WNDB[key] = 0
            self.real_WNDB[key] = 0
            self.should_payback[key] = 0
        self.develop()
        self.profit = 0  # 利润，等价于当前回合收回利息数，或破产后减少的金额

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
        return rent_val

    def answer_debt(self, name: str, day):   # 回合开始由企业询问该回合待还款本金和待还款利息
        self.should_payback[name] = self.bond_detail[name][day % self.debt_time]  # 记录下当日待还款
        return {'money': round(self.should_payback[name], 2), 'iD':round(self.debet_i * self.bond[name], 2)}

    def deal_payback(self, name:str, payback: dict):   # 处理来自企业的还款
        self.debt[name] = round(self.debt[name] - payback['payback'], 2)      # 银行收回现金，银行对企业债务减一笔
        self.bond[name] = round(self.bond[name] - payback['payback'], 2)      # 银行收回贷款，银行对企业债权减一笔
        self.profit = round(self.profit + payback['iD'], 2)                    # 银行收回利息，银行利润加一笔
        self.total_profit += self.profit
        self.money = round(self.money + payback['iD'], 2)                     # 银行收回利息，银行现金（总利润）加一笔
        self.debt[name] = round(self.debt[name] - payback['iD'], 2)           # 银行收回利息，银行对企业欠款减少

    def trade_callback(self, name: str, delta_money:float):
        # 企业处理完订单后变动的金钱数量，平衡资产债务表
        self.debt[name] = round(self.debt[name] + delta_money, 2)



    def develop(self):  # 经济发展，储备金增加
        # 根据市面上流动的现金来决定，市面上现金为银行对企业债务之和
        self.fund += sum(self.debt.values()) * self.fund_increase

    def get_state(self):
        state = [self.money/100, self.able_fund/1000]
        flag = True
        for key in self.observation.keys():
            if flag:
                ns = self.observation[key].get_state()
                flag=False
            else:
                ns = self.observation[key].get_state()[0:13]
            state = state + ns
        return state

    def get_action(self, state: list, new_ep):
        if self.is_random:
            return [random.random() - 0.5,random.random() - 0.5,random.random() - 0.5,random.random() - 0.5]
        start = 0
        if self.Agent is None:
            self.Agent = {}
            for key in self.agent_config:
                self.agent_config[key].set_scope(self.name + "_" + key)
                self.agent_config[key].set_state_dim(len(state))
                self.Agent[key] = self.agent_factory(self.agent_config[key])
        action = []
        for key in self.Agent:
            a = self.Agent[key].run_bank(state, new_ep)
            for i in a :
                action.append(i)

        return action

    def custom_reward(self):
        self.reward['WNDB'] = self.profit/100

    def get_reward(self):
        decay = self.reward_decay ** self.step
        for key in self.total_reward:
            self.total_reward[key] += self.reward[key] * decay
        self.step += 1
        return self.reward

    def get_fail_reward(self):
        fail_reward = {'WNDB':0}
        decay = self.reward_decay ** self.step
        # for key in self.observation:
        #     if self.observation[key].is_falled():
                # fail_reward['WNDB'] -= self.bond[key]
        fail_reward['WNDB'] = -10
        self.total_reward['WNDB'] += fail_reward['WNDB'] * decay
        return fail_reward

    def feedback(self, state, action, state_: list, reward: dict, is_train: bool = True, is_end: bool = False):
        if self.is_random:
            for key in self.agent_config:
                self.loss[key] = 0
            return self.loss
        start = 0
        for key in self.Agent:
            offset = self.agent_config[key].action_dim
            self.loss[key] = self.Agent[key].env_upd(state, action, state_, reward[key], is_train,is_end)
            start = start + offset
        return self.loss

    def mark(self):
        if self.Agent is None:
            return ""
        return self.Agent['WDNB'].mark()

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





