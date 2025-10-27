
from real_System.MADDPG.MADDPGConfig import MADDPGConfig
from Agent.Config import Config
from real_System.MADDPG.MADDPGEnterprise import Enterprise
from real_System.MADDPG.MADDPGBank import Bank
from real_System.Market import Market
from real_System.MADDPG.maddpg_enterprise import enterprise_nnu
from real_System.MADDPG.maddpg_bank import bank_nnu
from real_System.Logger import Logger
import copy
import random

third_market_price = 100
enterprise_price = 8


def actWNDF(enterprise: Enterprise, target:str, action:any):  # action: -0.5~0.5
    # if enterprise.money == 0:
    #     res = 100
    # else:
    #     res = (enterprise.money + enterprise.stock * third_market_price * 0.8) * (action + 0.5)
        # res = enterprise.money * (action + 0.5)
    # if enterprise.is_random:
    #   res = random.randint(100,200)
    #else:
    res = enterprise.money * (action + 0.5)
    return res

def actWNDF_third_market(enterprise: Enterprise, target:str, action:any):  # action: -0.5~0.5

    res = (enterprise.money + enterprise.stock * third_market_price * 0.5) * (action + 0.5)
    return res

def actKL(enterprise: Enterprise, target: str, action:any):  # action: -0.5~0.5

    if enterprise.intention_policy[target] == 0:
        return 10 * (action + 0.5)
    return enterprise.intention_policy[target] * (1+action)


# 因为price决策是上一回合给的，此处的next_price是上回合赋值的，等价于这回合的price
# 所以将此刻next_price的值赋给price，并修改next_price的值以供下回合使用
def actPrice(enterprise: Enterprise, target:str, action:any):  # action: -0.5~0.5
    res = enterprise.next_price
    enterprise.next_price = enterprise.next_price * (1 + action)
    if enterprise.next_price == 0.0:
        enterprise.next_price = 10 * (action + 0.5)
    return res


def actWNDB(bank: Bank, target: str, action: any):      # action:0~1
    res = bank.observation[target].WNDF * action
    # if bank.observation[target].WNDF == 100.0:
    #     res = 100.0
    return res


action_controller = {
    # 'e_all': ['production1', 'consumption1', 'production_thirdMarket', 'consumption_thirdMarket'],     # 所有企业
    'e_all': ['production1', 'consumption1'],     # 所有企业
    'e_execute': ['production1', 'consumption1'],                                                      # 参与决策的企业
    # 'e_seller': ['production1', 'consumption1', 'production_thirdMarket', 'consumption_thirdMarket'],  # 参与售卖的企业
    'e_seller': ['production1', 'consumption1'],  # 参与售卖的企业
    'e_buyer': ['production1', 'consumption1'],                                                        # 参与购买的企业
    'b_execute': ['bank1']
}

e_action = ['WNDF', 'K', "L", "price"]
b_action = action_controller['e_execute']   # 银行的决策，本质就是给需要决策的企业借贷，所以动作空间和action_controller[一致
b = 'bank1'
all_shop = ['K', 'L']

enterprise_functions = {
    'WNDF':actWNDF,
    'K':actKL,
    'L':actKL,
    'price':actPrice
}


bank_functionos = {
    'WNDB':actWNDB
}


class MADDPGSystem:
    def __init__(self, name:str, episodes: float = 3000,
                 days: float = 100,
                 base_fund: float = 1000,
                 break_epi_mul_day:int = 100000,
                 fund_increse: float = 0.1,
                 debt_time:int = 6,
                 enterprise_ddpg_config:MADDPGConfig=None,
                 bank_ddpg_config:Config=None,
                 is_delay_feed_back:bool = True,
                 producer_random:bool = False,
                 consumer_random:bool = False,
                 bank_random:bool = False,
                 has_third_market = True,
                 logger_path:str = None):

        self.break_epi_mul_day = break_epi_mul_day
        if has_third_market:
            enterprise_functions['WNDF'] = actWNDF_third_market
        self.Enterprise = {
             'production1': Enterprise(name='production1', output_name='K', shop_dir=all_shop, price=enterprise_price,
                                       action_function=enterprise_functions, e_buyer=action_controller['e_buyer'],
                                       e_seller=action_controller['e_seller'], agent=enterprise_nnu,
                                       intention=5,agent_config=enterprise_ddpg_config,is_random=producer_random),

             'consumption1': Enterprise(name='consumption1', output_name='L', shop_dir=all_shop, price=enterprise_price,
                                        action_function=enterprise_functions, e_buyer=action_controller['e_buyer'],
                                        e_seller=action_controller['e_seller'], agent=enterprise_nnu,
                                        intention=5,agent_config=enterprise_ddpg_config,is_random=consumer_random),
         }
        if has_third_market:
            # 海外市场 无限供应 intention=0不购买只出售
            self.Enterprise['production_thirdMarket'] = Enterprise(name='production_thirdMarket', output_name='K', money=float('inf'),
                                                  price=third_market_price,intention=0,
                                                  stock=float('inf'), shop_dir=['K'], action_function=enterprise_functions)
            # 海外市场 无限供应 intention=0不购买只出售
            self.Enterprise['consumption_thirdMarket'] = Enterprise(name='consumption_thirdMarket', output_name='L', money=float('inf'),
                                                   price=third_market_price,intention=0,
                                       stock=float('inf'), shop_dir='L', action_function=enterprise_functions)
            action_controller['e_all'].append('production_thirdMarket')
            action_controller['e_all'].append('consumption_thirdMarket')
            action_controller['e_seller'].append('production_thirdMarket')
            action_controller['e_seller'].append('consumption_thirdMarket')
            self.trade_time = 2
        else:
            self.trade_time = 1

        self.bank = {
            'bank1': Bank(name='bank1', agent=bank_nnu, agent_config=bank_ddpg_config,
                          action_function=bank_functionos,fund=base_fund,fund_rate=1,fund_increase=fund_increse,
                          debt_time=debt_time,is_random=bank_random)
        }
        self.last_state = {}
        self.last_action = {}
        self.market = Market('market')
        self.logger = Logger(name=name,base_path=logger_path)
        self.episodes = episodes
        self.days = days
        self.is_train = True
        self.is_delay_feed_back = is_delay_feed_back
        self.epi_mul_day = 0

        self.config_output()

    def config_output(self):
        block = "\n\n\n\n\n"
        for key in self.__dict__:
            self.logger.output_config(key + ":" + str(self.__dict__[key]))
        self.logger.output_config(block)
        for enterprise in self.Enterprise:
            res = enterprise + ':\n\n'
            for key in self.Enterprise[enterprise].__dict__:
                if isinstance(self.Enterprise[enterprise].__dict__[key],dict):
                    for detail in self.Enterprise[enterprise].__dict__[key]:
                        res = res + ' ' + key + ': ' + str(self.Enterprise[enterprise].__dict__[key][detail]) + '\n'
                else:
                    res =res + ' ' + key + ': ' + str(self.Enterprise[enterprise].__dict__[key]) + '\n'
            self.logger.output_config(res)
            self.logger.output_config(block)

        for bank in self.bank:
            res = bank + ':\n\n'
            for key in self.bank[bank].__dict__:
                if isinstance(self.bank[bank].__dict__[key],dict):
                    for detail in self.bank[bank].__dict__[key]:
                        res = res + ' ' + key + ': ' + str(self.bank[bank].__dict__[key][detail]) + '\n'
                else:
                    res =res + ' ' + key + ': ' + str(self.bank[bank].__dict__[key]) + '\n'
            self.logger.output_config(res)
            self.logger.output_config(block)



    def new_episode(self):
        self.last_state = {}
        self.last_action = {}
        for key in self.Enterprise.keys():
            self.Enterprise[key].new_episode()
        for key in self.bank.keys():
            self.bank[key].new_episode()

    def new_day(self, day):
        for key in self.Enterprise.keys():
            self.Enterprise[key].new_day(day)
        for key in self.bank.keys():
            self.bank[key].new_day()

    # step 1
    # 一天开始前的准备工作 并获取当天的state
    def new_day_prepare(self, episode, day):
        if episode % 10 == 0:
            self.logger.output_to_txt(self.market.show_order())
        self.market.new_day()
        state = {}
        # step 1
        # 所有参与售卖的企业先供给商品到市场
        for key in action_controller['e_seller']:
            offer = self.Enterprise[key].offer()  # 企业供给商品，包括 商品类型 商品数量 商品价格
            # 市场接收商品
            self.market.receive(name=key, shop_name=offer['shop_type'], num=offer['num'], price=offer['price'])

        for key in action_controller['e_execute']:

            # step 2
            # 执行企业需要得知当前市场上所有自己需要的商品的价格和数量
            for shop_name in self.Enterprise[key].shop_dir:  # 遍历每个企业的shop_dir 得知该企业需要的商品类型有哪些
                shop_list = self.market.answer_all_price(shop_name=shop_name)  # 返回当前商品在市场上的数量和价格的list
                self.Enterprise[key].ask_all_shop(shop_name=shop_name, shop_info=shop_list)  # 传递给企业

            # 执行企业决策前，需要得知当前回合该企业待偿还的本金和利息
            debt = self.bank[b].answer_debt(name=key, day=day)  # 询问银行当日待偿还利息和本金
            self.Enterprise[key].ask_debt(bill=debt)  # 记下对银行的账单详情
            # 获取本日的state
            self.Enterprise[key].custom_state()
            state[key] = self.Enterprise[key].get_state()
        for key in action_controller['b_execute']:
            state[key] = self.bank[key].get_state()  # 获取当前训练用state

        return state

    def feed_back(self,state,state_,action,reward,is_train,is_end):
        loss = {}
        other_state_dict = {
            'production1':'consumption1',
            'consumption1':'production1'
        }
        for key in action_controller['e_execute']:
            # step 15
            # 开始将数据存入经验池
            # 获得了这一天的 r
            # 此时统计的四元组为[上一回合的state，上一回合的action，奖励，这一回合的state]
            # 因为上一回合的state -上一回合action-> 这一回合的state这个过程的奖励此时才真正体现出来
            # 一般情况is_end 为False，因为既然能到这一回合，说明上一回合没有破产
            # reward = self.Enterprise[key].get_reward()
            other_key = other_state_dict[key]
            # 时序错峰
            self.Enterprise[key].feedback(state=state[key],other_state=state[other_key],
                                              action=action[key],other_action=action[other_key],
                                              state_=state_[key],other_state_=state_[other_key],
                                              reward=reward[key], is_end=is_end)
            sample_batch = self.Enterprise[key].get_sample_batch()
            other_agent_policy_action,other_agent_policy_action_ = self.Enterprise[other_key].get_other_agent_policy_action(sample_batch)
            l = self.Enterprise[key].learn(other_agent_policy_action,other_agent_policy_action_,self.is_train)


            if l is not None:
                loss[key] = l
        # 获取银行的r
        # 同样此时的r也是上一回合到这一回合的reward
        # 故而is_end仍然等于False
        l = self.bank[b].feedback(state=state[b], action=action[b],
                              state_=state_[b], reward=reward[b], is_train=is_train, is_end=is_end)
        if l is not None:
            loss[b] = l
        self.logger.receive_loss(loss=loss)

    def run(self):

        # 将所有参与决策的企业加入银行的观察
        for key in action_controller['e_execute']:
            self.bank[b].observe(self.Enterprise[key])

        for key in action_controller['e_execute']:
            self.market.subscribe(self.Enterprise[key])

        for episode in range(self.episodes):
            if self.epi_mul_day > self.break_epi_mul_day and episode % 100 == 0:
                break
            self.is_train = False
            self.new_episode()
            print("第" + str(episode) + "回合开始")
            state = self.new_day_prepare(episode,0)
            if episode == self.episodes - 1000:
                self.logger.clear_action()
                # self.logger.clear_finish_data()
            for day in range(self.days):
                if day > 1:
                    self.is_train = True
                    self.epi_mul_day = self.epi_mul_day + 1
                self.new_day(day)
                action = {}

                # 需要参与智能体决策的企业进行决策
                # 第一天不进行决策，按照固定行动开局
                if day > 0:
                    for key in action_controller['e_execute']:

                        # step 4
                        # 自此，决策前准备已经就绪，开始决策
                        action[key] = self.Enterprise[key].get_action(state=state[key], new_ep=day is 1)  # 获取当前回合决策action[WNDF, K, L, NP]
                        self.logger.receive_toshow(key,self.Enterprise[key].get_network_show())
                        for i in range(len(e_action)):
                            self.Enterprise[key].set_action(target=e_action[i], action=action[key][i])  # 逐个设置当前回合的决策
                        # 统计决策动作数据
                        self.logger.receive_action(name=key, action=action[key], action_detail=e_action, episode=episode)

                    # step 5
                    # 银行开始决策
                    action[b] = self.bank[b].get_action(state=state[b], new_ep=day is 1)  # 获取当前回合决策action[]
                    for i in range(len(b_action)):
                        self.bank[b].set_action(target=b_action[i], action=action[b][i])  # 设置银行决策动作
                    self.logger.receive_action(name=b, action=action[b], action_detail=b_action, episode=episode)
                else:
                    for i in range(len(b_action)):
                        self.bank[b].set_action(target=b_action[i], action=1)  # 设置银行决策动作

                if episode % 10 == 0:
                    self.logger.output_to_txt("第" + str(episode) + "回合 第" + str(day) + "天：\n")
                    self.logger.output_to_txt('动作决策' + str(action))
                    self.logger.output_to_txt('state :' + str(state))

                # step 6
                # 自此，企业和银行都完成了决策和动作赋值
                # 接下来，银行开始借钱
                # 首先检查自己的决策，看看剩余储备金够不够借出
                self.bank[b].check_rent()
                for enterprise_key in action_controller['e_execute']:
                    rent_val = self.bank[b].rent(name=enterprise_key, day=day)  # 银行借出钱
                    self.Enterprise[enterprise_key].deal_rent(rent_val=rent_val)  # 企业处理借款

                # 自此，银行完成了放贷，企业完成了借款
                # 接下来，要开始进行商品交易
                # 目前生产消费各一家企业，如果买不够，则要去第三方市场买,所以交易两次
                # 交易途中不回款，防止同一笔钱反复利用
                for i in range(self.trade_time):
                    # step 7
                    # 市场从最低价开始交易，先摇获取市场中各个商品最低价
                    min_price = {}
                    for shop_name in all_shop:
                        min_price[shop_name] = self.market.answer_min_price(shop_name=shop_name)

                    # step 8
                    # 参与购买的企业根据当前最低市场价重新调整自身购买意愿，以便自身买得起
                    for key in action_controller['e_buyer']:
                        self.Enterprise[key].check_intention(shop_price=min_price)  # 根据每个所需商品的最低价格重新调整自身购买意愿
                        require_list = self.Enterprise[key].require()  # 获取企业需求
                        self.market.get_require(name=key, require_list=require_list)  # 将企业需求提交给市场
                    if episode % 10 == 0:
                        self.logger.output_to_txt(str(self.market))
                    # step 9
                    #  需求接收完成，市场开始分配商品给买方
                    self.market.assign()
                    for key in action_controller['e_buyer']:
                        bill = self.market.get_bill(buyer=key)  # 每个企业获取此次交易的账单
                        delta_money = self.Enterprise[key].deal_bill(bill_list=bill)  # 企业处理账单，获取此次变动金额
                        self.bank[b].trade_callback(name=key, delta_money=delta_money)  # 银行处理金钱变动

                # step 10
                # 参与售卖的企业获取回款
                for key in action_controller['e_execute']:
                    payback = self.market.get_payback(seller=key)  # 获取回款单
                    delta_money = self.Enterprise[key].payback(payback_list=payback)  # 获取金钱变动
                    self.bank[b].trade_callback(name=key, delta_money=delta_money)

                # step 11
                # 企业生产
                for key in action_controller['e_execute']:
                    self.Enterprise[key].product()

                # step 12
                # 银行收回贷款
                for key in action_controller['e_execute']:
                    self.bank[b].deal_payback(name=key, payback=self.Enterprise[key].turn_back_money())

                # 每日结束清算
                for key in action_controller['e_execute']:
                    self.Enterprise[key].daily_settlement()


                # step 13
                # 计算reward
                # 这个reward是上一回合到这一回合过程的reward
                for key in action_controller['e_execute']:
                    self.Enterprise[key].custom_reward()
                self.bank[b].custom_reward()

                is_end = False
                for key in action_controller['e_execute']:
                    is_end = is_end or self.Enterprise[key].is_falled()

                # 第一天不进行动作决策，按照固定操作开局
                if day > 0:
                    # 统计数据，一定要在这一步，在step15后部分数据更新为次日数据
                    for key in action_controller['e_execute']:
                        self.logger.receive_enterprise(episode=episode, day=day, target=self.Enterprise[key])

                    for key in action_controller['b_execute']:
                        self.logger.receive_bank(episode=episode, day=day, target=self.bank[key])

                    # step 14
                    # 破产清算



                    # 如果day == self.days-1 说明到了最后一天
                    # 此时理应将feed_back中的is_end设为True
                    # 但如果有人最后一个天破产
                    # 那么在step 16时将会还有一条is_end为True的带惩罚reward的数据
                    # 这种情况is_end就为False
                    feed_back_end = (day == self.days - 1) and not is_end

                    # 时序错峰一天
                    if self.is_delay_feed_back:
                        if self.is_train:
                            reward = {}
                            for key in action_controller['e_execute']:
                                reward[key] = self.Enterprise[key].get_reward()
                            reward[b] = self.bank[b].get_reward()
                            self.feed_back(state=self.last_state,action=self.last_action,
                                           state_=state,reward=reward,is_train=self.is_train,
                                           is_end=feed_back_end)

                    # 至此 所有上一回合的数据last_state和last_action都已使用完毕
                    # 所以接下来可以更新这些数据了
                    # 将action计入last_action
                    for key in action:
                        self.last_action[key] = action[key]
                    # 将state计入last_state
                    for key in state.keys():
                        self.last_state[key] = state[key]



                # 获取新的state
                # 如果无人破产,这个新的state,将会与当前的last_state，当前的last_action,共同在下一回合末与reward一同存入经验池
                state = self.new_day_prepare(episode, day + 1)  # 此时的s_ 为 day+1天的s

                # 时序不错峰
                if not self.is_delay_feed_back:
                    if  self.is_train:
                        reward = {}
                        for key in action_controller['e_execute']:
                            reward[key] = self.Enterprise[key].get_reward()
                        reward[b] = self.bank[b].get_reward()
                        self.feed_back(state=self.last_state, action=self.last_action,
                                       state_=state, reward=reward,is_train=self.is_train,
                                       is_end=feed_back_end)

                # step 16
                # 但是如果有人破产
                # 则进行结算
                if is_end:
                    reward = {}
                    for key in action_controller['e_execute']:
                        reward[key] = self.Enterprise[key].get_fail_reward()
                    reward[b] = self.bank[b].get_fail_reward()
                    if self.is_train:
                        # 企业破产清算
                        # 此时四元组将是[这一回合state，这一回合action，惩罚reward，下一回合state]
                        # print('before',self.Enterprise[key].total_reward)
                        self.feed_back(state=self.last_state, action=self.last_action,
                                       state_=state, reward=reward,is_train=True,
                                       is_end=True)



                # 输出数据
                if episode % 10 == 0:
                    self.logger.output_to_txt(str(self))
                    if day > 0:
                        self.logger.output_to_txt(self.epi_mul_day)
                    # print(self)


                if is_end or day == self.days-1:
                    print("第", episode, "回合 存活", day, "天,当前进度",self.epi_mul_day,"/",self.break_epi_mul_day,self.epi_mul_day * 100/self.break_epi_mul_day,'%\n')
                    # 统计结束数据
                    for key in action_controller['e_execute']:
                        self.logger.receive_finish_enterprise(episode=episode, day=day, target=self.Enterprise[key])
                        print('after',self.Enterprise[key].total_reward)
                    print('after',self.bank[b].total_reward)
                    for key in action_controller['b_execute']:
                        self.logger.receive_finish_bank(episode=episode, day=day, target=self.bank[key])

                    break
        self.finish()


    def finish(self):
        print("logger.finish")
        self.logger.finish()
        print("logger.show")
        self.logger.show_all()
        print("logger.tocsv")
        self.logger.to_csv()


    def __str__(self):
        res = ''
        for enterprise in action_controller['e_execute']:
            res += str(self.Enterprise[enterprise])

        for bank in action_controller['b_execute']:
            res += str(self.bank[bank])

        return res

