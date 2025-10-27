import math
from .Order import Order
from .Enterprise_config import Enterprise_config
import copy
import random
import numpy as np

# 市场主体 消费品企业为消费市场主体 生产资料企业为生产市场主体  消费者、生产者的海外市场分别为对应市场主体
# 海外市场库存无限，价格固定
class Enterprise:
    def __init__(self,
                 config: Enterprise_config,
                 ):
        self.config = config
        self.action_functiontion = config.action_function  # 决定各个变量的动作方式 类型为字典str:function(),如定义数据K使用方法f为百分比递增，{'K':f}
        self.name = config.name  # 当前主体名称
        self.output_name = config.output_name  # 当前主体产出名称
        self.cal_gamma = config.gamma

        self.loss = {}

        # 以上为参数,以下为运行时数据
        self.money = self.config.money  # M    现金
        self.stock = self.config.stock  # X    库存
        self.debt = 0.0  # D    负债
        self.revenue = 0.0  # R    收入
        self.should_payback = 0.0  # 当日待还款
        self.iDebt = 0.0  # iD   本回合待偿还利息
        self.last_cost = 0.0
        self.cost = 0.0  # C    成本
        self.economy_profit = 0.0  # 金融利润 为当日新借贷-偿还本金-偿还利息
        self.business_profit = 0.0  # 商业利润 为今日收入减去昨日成本
        self.price = self.config.price  # P   定价
        self.next_price = self.config.price  # 次日定价
        self.WNDF = self.config.WNDF  # WNDF 追加贷款意愿
        self.get_WNDF = 0.0  # 当前回合实际获取贷款数
        self.total_profit = 0.0  # 累计利润和
        self.total_cost = 0.0  # 累计成本和
        self.total_revenue = 0.0  # 累计收入和
        self.total_idebt = 0.0  # 累计利息和
        self.output = 0.0  # 当前回合产量
        self.sales = 0.0  # 当前回合销量
        self.intention_policy = {}  # 当日决策购买产品意愿   类型为字典，放入购买产品如{'K':5.5,'L':6.6}
        self.purchase_intention = None  # 此时的购买意愿，当只购买一次的时候等同于intention_policy，当需要到第三方市场进行二次购买时为购买前的意愿
        self.real_intention = {}  # 根据当日所持金和定价，执行机构最终决定购买意愿 类型格式同上
        self.get_shop = {}  # 当日实际获得产品      类型格式同上
        self.all_shop = {}  # 当日市场所有商品 类型为字典 dict{商品名str,dict{'price':list[1,2,3],'num':list[1,2,3}} 用于训练
        self.reward = None
        self.reward_gamma = 0.95
        self.is_fall = False
        self.market_record = {}  # 企业可以观察到市场每一笔交易，记录其中用于训练输入数据
        self.state = []
        self.last_output = 0
        self.total_reward = {'economy': 0, 'business': 0}
        self.reward_decay = 0.95
        self.step = 0
        # 格式 dict{买家名str,dict{卖家名str,dict{'price':价格float,'num':数量float}}}

    def init(self,
             shop_dir: list = [],  # 类型为字典,传入商品类别及其初值，如{'K':5.5,'L':6.6}
             e_buyer: list = [],  # 参与商品购买的企业
             e_seller: list = []  # 参与商品售卖的企业
             # 以上两个变量用于区分第三方市场与非第三方市场，用于训练时的输入数据
             ):
        self.shop_dir = shop_dir
        self.e_buyer = e_buyer
        self.e_seller = e_seller

        for key in self.shop_dir:
            self.intention_policy[key] = self.config.intention
            self.real_intention[key] = 0.0
            self.get_shop[key] = 0.0

    def new_day(self,day):

        self.total_cost += self.cost
        self.total_revenue += self.revenue
        self.total_idebt += self.iDebt
        if day > 1:  #
            self.total_profit += self.business_profit  # 利润 = 今日的收益 - 昨日的成本

        self.business_profit = -self.cost  # 统计完总利润后清空  ?
        self.economy_profit = 0
        self.revenue = 0
        self.last_cost = self.cost
        self.cost = 0
        self.sales = 0.0  # 当前回合销量
        self.purchase_intention = None  # 此时的购买意愿，当只购买一次的时候等同于intention_policy，当需要到第三方市场进行二次购买时为购买前的意愿
        self.real_intention = {}  # 根据当日所持金和定价，执行机构最终决定购买意愿 类型格式同上
        self.get_shop = {}  # 当日实际获得产品清空防止干扰
        self.all_shop = {}  # 当日市场所有商品定价清空防止干扰

        self.market_record = {}  # 企业可以观察到市场每一笔交易，记录其中用于训练输入数据

    def new_episode(self):
        self.money = self.config.money  # M    现金
        self.stock = self.config.stock  # X    库存
        self.debt = 0.0  # D    负债
        self.revenue = 0.0  # R    收入
        self.should_payback = 0.0  # 当日待还款
        self.iDebt = 0.0  # iD   利息
        self.last_cost = 0.0
        self.cost = 0.0  # C    成本
        self.economy_profit = 0.0  # 金融利润
        self.business_profit = 0.0  # 商业利润
        self.price = self.config.price  # P   定价
        self.next_price = self.config.price  # 次日定价
        self.WNDF = self.config.WNDF  # WNDF 追加贷款意愿
        self.get_WNDF = 0.0  # 当前回合实际获取贷款数
        self.total_profit = 0.0  # 累计利润和
        self.total_cost = 0.0  # 累计成本和
        self.total_revenue = 0.0  # 累计收入和
        self.total_idebt = 0.0  # 累计利息和
        self.output = 0.0  # 当前回合产量
        self.sales = 0.0  # 当前回合销量
        self.intention_policy = {}  # 当日决策购买产品意愿   类型为字典，放入购买产品如{'K':5.5,'L':6.6}
        self.purchase_intention = None  # 此时的购买意愿，当只购买一次的时候等同于intention_policy，当需要到第三方市场进行二次购买时为购买前的意愿
        self.real_intention = {}  # 根据当日所持金和定价，执行机构最终决定购买意愿 类型格式同上
        self.get_shop = {}  # 当日实际获得产品      类型格式同上
        self.all_shop = {}  # 当日市场所有商品 类型为字典 dict{商品名str,dict{'price':list[1,2,3],'num':list[1,2,3}} 用于训练
        self.is_fall = False
        self.market_record = {}  # 企业可以观察到市场每一笔交易，记录其中用于训练输入数据
        self.state = []
        self.loss = {}
        self.last_output = 0
        self.total_reward = {'economy': 0, 'business': 0}
        self.step = 0
        self.total_sales = 0

        for key in self.shop_dir:
            self.intention_policy[key] = self.config.intention
            self.real_intention[key] = 0.0
            self.get_shop[key] = 0.0

    def daily_settlement(self):
        self.business_profit += self.revenue

    def set_action(self,
                   target: str,
                   action: any,
                   failTarget: str = 'intention_policy'):  # 如果target不为已实例变量，则寻找已实例化字典变量failTarget
        try:  # 优先选择当前类已有的变量实例，
            self.__getattribute__(target)
            self.__dict__[target] = self.action_functiontion[target](self, target, action)
        except AttributeError:  # 如果没有对应的对象实例，到字典intention_policy中
            try:
                self.__dict__[failTarget][target] = self.action_functiontion[target](self, target, action)
            except KeyError:  # 如果action_functiontion中没有对应的方法，默认为赋值
                self.__dict__[failTarget][target] = action
        except KeyError:  # 已有变量实例，而action_functiontion中没有对应的方法，默认为赋值
            self.__dict__[target] = action

    def offer(self):
        # 因为调用该方法时需要提供的价格是次日价格，所以是next_price
        # 这样可能会觉得self.price根本没用到，但那是因为self.price唯一用到的地方只有训练时
        # 作为上一回合的交易价格喂给模型
        return {'shop_type': self.output_name, 'num': self.stock, 'price': self.next_price}

    def require(self):
        require_list = {}
        for key in self.real_intention.keys():
            require_list[key] = self.real_intention[key]
        return require_list

    # 此函数用来调整购买需求，若钱不够则按百分比分配
    def check_intention(self, shop_price: dict):
        total_money = 0
        total_shop = 0
        # 如果是本日首次购买，则按照决策商品需求的来重新调整需求
        if self.purchase_intention is None:
            self.purchase_intention = {}
            for key in self.intention_policy.keys():
                self.purchase_intention[key] = self.intention_policy[key]
        # 不为None，即为本日复数次购买，按照剩余的商品需求来调整
        else:
            self.purchase_intention = {}
            for key in self.real_intention:
                self.purchase_intention[key] = self.real_intention[key]

        for key in self.purchase_intention.keys():
            total_money += self.purchase_intention[key] * shop_price[key]  # 总金钱数
            total_shop += self.purchase_intention[key]  # 总商品数
        percent = min(self.money / (total_money + 1e-2), 1)  # 如果当前现金比需要的钱多则为1，少则等比分配
        for key in self.purchase_intention.keys():
            # round:返回浮点数x的四舍五入值。0.5靠近偶数端-0
            self.real_intention[key] = round(self.purchase_intention[key] * percent, 2)

    def receive_market_record(self, order: Order):
        seller = order.seller
        buyer = order.buyer
        price = order.price
        num = order.num
        if buyer not in self.market_record:
            self.market_record[buyer] = {}
        if seller not in self.market_record[buyer]:
            self.market_record[buyer][seller] = {}
        self.market_record[buyer][seller]['price'] = price
        self.market_record[buyer][seller]['num'] = num

    # 自己是买方 交易账单
    def deal_bill(self, bill_list: list) -> float:
        delta_money = 0  # 将金钱变化上报给银行用
        for order in bill_list:
            if order.buyer == self.name:  # 自己是买方
                # 如果当日需要商品不在实际获得商品中，令实际获得【需要】为0
                if order.shop not in self.get_shop:
                    self.get_shop[order.shop] = 0
                self.get_shop[order.shop] += order.num
                cost = round(order.num * order.price, 2)
                self.cost += cost
                self.money -= cost
                delta_money -= cost
                self.real_intention[order.shop] = round(self.real_intention[order.shop] - order.num, 2)  # 部分需求满足，调整需求
                if self.real_intention[order.shop] < 1e-1:
                    self.real_intention.pop(order.shop)
        return delta_money

    #
    def payback(self, payback_list: list) -> float:
        delta_money = 0  # 将金钱变化上报给银行用
        for order in payback_list:
            if order.seller == self.name:  # 自己是卖方 商品必然是自身产出
                self.stock = self.stock - order.num
                if self.stock < 1e-1:  # 库存减少 防止round精度问题导致出现负数
                    self.stock = 0
                revenue = round(order.num * order.price, 2)
                self.revenue += revenue
                self.money += revenue
                self.sales += order.num
                delta_money += revenue
        self.total_sales += self.sales
        return delta_money

    def deal_rent(self, rent_val: float):
        self.money += rent_val  # 现金增加r
        self.debt += rent_val  # 债务增加
        self.get_WNDF = rent_val  # 实际获得贷款

    def ask_all_shop(self, shop_name: str, shop_info: dict):  # shop_info dict{'price':list,'num':list}
        if shop_name not in self.all_shop:
            self.all_shop[shop_name] = {}
        self.all_shop[shop_name]['price'] = shop_info['price']
        self.all_shop[shop_name]['num'] = shop_info['num']

    def ask_debt(self, bill: dict):  # 回合开始，决策前先询问当天待还款本金和利息
        # 如果是正常交易系统，可以无需这一步，直接在还款环节询问银行待还款金额即可
        # 但因为需要训练，所以回合开始前需要将这些训练用得到的参数存入自身成员，方便之后使用
        self.should_payback = bill['money']  # 待还款本金
        self.iDebt = bill['iD']

    def turn_back_money(self) -> dict:  # 还债
        pay_iDebt = min(self.iDebt, self.money)
        if self.money < self.iDebt:
            self.is_fall = True
        self.money = round(self.money - pay_iDebt, 2)  # 还债利息，现金减少
        pay_should_payback = min(self.should_payback, self.money)
        if self.money < self.should_payback:
            self.is_fall = True
        self.money = round(self.money - pay_should_payback, 2)  # 还债本金，现金减少
        self.debt = round(self.debt - pay_should_payback, 2)  # 偿还本金债务减少
        # 利息不再算入cost
        # self.cost += pay_iDebt
        self.economy_profit -= pay_iDebt
        return {'payback': pay_should_payback, 'iD': pay_iDebt}  # 将还款额度报告给银行

    def product(self):
        # A = 2.5
        produce = 2.5
        min_num = float('inf')  # 'inf' 正无穷
        # for shop_num in self.get_shop.values():
        #     produce *= math.sqrt(max(shop_num,0))
        # get_shop 实际获得的商品
        for shop_num in self.get_shop.values():
            min_num = min(min_num, shop_num)  # min_num 实际获得商品的最小价值
        # produce = round(produce, 2)
        produce = produce * max(min_num, 0)
        self.last_output = self.output  # 记录上次的输出
        self.output = produce  # 新的输出，本回合的生产量
        if produce < 1e-1:  # 1e-1=0.1
            self.is_fall = True
        self.stock += produce

    def get_state(self):
        return self.state

    # 因为银行的输入数据需要用到企业的state，所以一回合可能需要复数次调用get_state
    # 每次都从头计算感觉太蠢了，还是将计算与调用分离
    def custom_state(self,day,lim_day):
        '''
        # 如果pytorch的bn没有问题
        state = [
            self.money/1,           # 1  企业当前现金 #1000
            self.stock/1,           # 2  企业当前存货 #100
            self.debt/1,            # 3  企业当前欠款总额 #1000
            self.sales/1,         # 4  企业上回合售出
            self.output/1,            # 5  企业上回合产出
            # self.business_profit,          # 6  企业上回合利润(上回合收入减去前回合支出)
            # self.economy_profit,  # 7
            self.price/1,           # 8  企业上回合定价(上回合的P) #10
            self.next_price/1,      # 9  企业这回合定价(上回合的NP) #10
            self.WNDF/1,            # 10  企业上回合决策的借贷金额 #1000
            self.get_WNDF/1,        # 11 企业上回合实际拿到的借贷金额 #1000
            self.should_payback/1,  # 12 企业本回合待偿还本金 #1000
            self.iDebt/1             # 13 企业本回合待偿还利息 #10
        ]
        '''

        # 原本手动调的bn state=企业原始数据11 + 决策购买意愿2 + 交易数据16 + 价格 4
        state = [
            self.money / 1000,  # 1  企业当前现金 #1000
            self.stock / 100,  # 2  企业当前存货 #100
            self.debt / 1000,  # 3  企业当前欠款总额 #1000
            self.sales / 10,  # 4  企业上回合售出
            self.output / 10,  # 5  企业上回合产出
            # self.business_profit,          # 6  企业上回合利润(上回合收入减去前回合支出)
            # self.economy_profit,  # 7
            self.price / 10,  # 8  企业上回合定价(上回合的P) #10
            self.next_price / 10,  # 9  企业这回合定价(上回合的NP) #10
            self.WNDF / 1000,  # 10  企业上回合决策的借贷金额 #1000
            self.get_WNDF / 1000,  # 11 企业上回合实际拿到的借贷金额 #1000
            self.should_payback / 1000,  # 12 企业本回合待偿还本金 #1000
            self.iDebt / 10  # 13 企业本回合待偿还利息 #10
        ]

        #
        for shop in self.shop_dir:
            state.append(self.intention_policy[shop] / 10)  # 14 16企业上回合K/L需求数量(原始数据而非执行端重新分配后)
            # state.append(self.real_intention[shop])    # 15 17企业上回合K/L执行端重分配后需求数量
        # e_buyer: 参与购买的企业，此时是consumption1和production1
        # e_seller: 参与售卖的企业，此时是e_buyer基础上加上两个第三方市场，共4家
        # 此处state意为: 将所有参与购买的企业向其他参与售卖的企业购买商品的价格和数量的交易记录作为输入数据记入state中
        # 所以此处的循环为:
        # 2 * 4 * 2，一共16条数据
        for buyer in self.e_buyer:
            for seller in self.e_seller:
                try:
                    state.append(self.market_record[buyer][seller]['price'] / 10)  # 10
                    state.append(self.market_record[buyer][seller]['num'] / 10)  # 100
                except KeyError:  # 如果market_record中没有某个seller，说明这回合没有向这个seller购买商品，记为0
                    state.append(0)
                    state.append(0)
        for shop_name in self.all_shop:
            for val in self.all_shop[shop_name]['price']:
                state.append(val / 10)  # 10
            # state = state + self.all_shop[shop_name]['price']
        # state.append((lim_day-day)/lim_day)
        # risk = self.debt / (self.money + 1e-6)
        # risk = np.clip(risk, 0, 5) / 5.0  # 归一化到 [0, 1]
        # state.append(risk)
        state.append(day / (lim_day-2))  # 归一化的进度 [0, 1]
        state.append((lim_day - day-2) / (lim_day-2))  # 剩余时间比例
        state.append(1.0 if day >= (lim_day-2) * 0.9 else 0.0)  # 是否接近终点
        self.state = state

    #尝试给高生存天数一个奖励
    def custom_reward(self, day, lim_day):
        self.reward = {}

        # 基础部分
        self.reward['economy'] = (self.revenue - self.last_cost) / 100
        self.reward['business'] = self.revenue / 100

        # 平滑延长奖励：从80%开始增长，最高放大到1.5倍
        progress = day / lim_day
        if progress > 0.8:
            bonus_factor = (progress - 0.8) * 0.5  # 在80%→100%之间线性增至0.1倍
            self.reward['economy'] *= (1 + bonus_factor)
            self.reward['business'] *= (1 + bonus_factor)

        # 最终生存奖励 目前最优情况
        # if day == lim_day -2 :
        #     self.reward['economy'] += lim_day * 0.2 / 80 # 额外奖
        #     self.reward['business'] += lim_day * 0.2 / 80

        if day == lim_day -2 :
            self.reward['economy'] += lim_day * 0.2 / 80 # 额外奖
            self.reward['business'] += lim_day * 0.2 / 80

        return self.reward

    # 预留字段：将一个智能体任务分成两个计算reward
    # def custom_reward(self, day,lim_day):
    #     self.reward = {}
    #     # self.reward['economy'] = self.business_profit + self.economy_profit
    #     self.reward['economy'] = self.revenue * 1 - self.last_cost
    #     # self.reward['economy'] = self.output
    #     # self.reward['business'] = self.output
    #     # self.reward['business'] = 2*self.business_profit + self.economy_profit
    #     # self.reward['business'] = self.revenue * 2 -self.last_cost
    #     # self.reward['business'] = (self.revenue * 2 - self.last_cost + self.economy_profit)
    #     # 10.8 将business替换为这个，和师姐的保持一致
    #     self.reward['business'] = self.revenue
    #     if day >= lim_day * 0.95:
    #         self.reward['economy']  = self.reward['economy']+ day * 0.1
    #         self.reward['business'] = self.reward['business'] + day * 0.1
    #     self.reward['economy'] /= 100
    #     self.reward['business'] /= 100

    # 这个total reward用来在回合结束之后展示
    def get_reward(self):
        #  decay reward衰减因子 越小,衰减的越慢
        decay = self.reward_decay ** self.step  # ** 幂运算 decay = reward_decay的step次方
        #  total_reward ： 包含economy和business的字典 {'economy':0,'business':0}
        for key in self.total_reward:
            self.total_reward[key] += self.reward[key] * decay
        self.step += 1
        return self.reward

    # def get_fail_reward(self, day, lim_day):
    #     reward = {}
    #     if day ==lim_day:
    #         decay = self.reward_decay ** self.step
    #         for key in self.total_reward:
    #             reward[key] = lim_day * 0.1
    #             self.total_reward[key] += reward[key] * decay
    #     elif self.is_fall:
    #         decay = self.reward_decay ** self.step
    #         for key in self.total_reward:
    #             reward[key] = -10
    #             self.total_reward[key] += reward[key] * decay
    #     else:
    #         for key in self.total_reward:
    #             reward[key] = 0
    #     return reward

    def get_fail_reward(self):
        reward = {}
        if self.is_fall:
            decay = self.reward_decay ** self.step
            for key in self.total_reward:
                reward[key] = -10
                self.total_reward[key] += reward[key] * decay
        else:
            for key in self.total_reward:
                reward[key] = 0
        return reward

    def is_falled(self):
        return self.is_fall

    def __str__(self):
        res = self.name + \
              "\n 企业现金:" + str(self.money) + \
              " 企业存货:" + str(self.stock) + \
              " 企业欠债:" + str(self.debt) + \
              " 企业收入:" + str(self.revenue) + \
              " 企业本回合待偿还本金:" + str(self.should_payback) + \
              " 企业利息:" + str(self.iDebt) + \
              " 企业成本:" + str(self.cost) + \
              " 企业利润:" + str(self.business_profit) + \
              " 企业贷款意愿:" + str(self.WNDF) + \
              " 企业商品意愿:" + str(self.intention_policy) + \
              " 企业可承受商品:" + str(self.real_intention) + \
              " 企业得到商品:" + str(self.get_shop) + \
              " 企业定价:" + str(self.price) + \
              " 企业总成本:" + str(self.total_cost) + \
              " 企业总收入:" + str(self.total_revenue) + \
              " 企业总利润:" + str(self.total_profit) + '\n' + \
              " reward:" + str(self.reward) + '\n'

        res = res + "             企业生产了" + str(self.output) + "个" + str(self.output_name) + '\n'
        res = res + "             企业以" + str(self.price) + "的价格卖出了" + str(self.sales) + "个" + str(
            self.output_name) + '\n'

        return res
