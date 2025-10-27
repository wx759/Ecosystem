# from real_System_remake.Logger_w import Logger
# from Agent.Config import Config
from real_System_remake.Enterprise import Enterprise
from real_System_remake.Bank import Bank
from real_System_remake.Market import Market
from real_System_remake.ddpg_enterprise import enterprise_nnu
from real_System_remake.ddpg_bank import bank_nnu
from real_System_remake.Logger import Logger
import copy
import random
from real_System_remake.Enterprise_config import Enterprise_config
from real_System_remake.Bank_config import Bank_config
import swanlab as wandb

third_market_price = 100
enterprise_price = 8

wandb_config = {
    'bank_ddpg_config': {
        'learning_rate_actor': 1e-3,
        'learning_rate_critic': 2e-3,
        'learning_rate_decay': 1,
        'random_seed': 184,
        'memory_capacity': 80000
    },
    'enterprise_ddpg_config': {
        'smooth_noise': 0.01,
        'learning_rate_actor': 1e-3,
        'learning_rate_critic': 2e-3,
        'learning_rate_decay': 1,
        'random_seed': 184,
        'memory_capacity': 80000
    },
    'attention_config': {
        'nums_heads': 4
    }

}


class Environment:
    def __init__(self, name: str,
                 lim_day: int = 200,
                 logger_path: str = None,
                 day_sum: int = 0,
                 day_max: int = 0,
                 day_sum_smooth: int = 0):

        self.day_sum_smooth = day_sum_smooth
        self.use_wandb = False
        if self.use_wandb:
            wandb.init(project="cortex22_anna", name=name, notes="Attention",
                       config=wandb_config)
        self.name = name
        self.lim_day = lim_day
        self.logger_path = logger_path
        self.day_sum = day_sum
        self.day_max = day_max
        self.eday = [0] * 301
        self.Enterprise = {}
        self.Bank = {}
        self.shop_dir = []
        self.action_controller = {
            'e_all': [],  # 所有企业
            'e_execute': [],  # 参与决策的企业
            'e_seller': [],  # 参与售卖的企业
            'e_buyer': [],  # 参与购买的企业
            'b_execute': []  # 参与决策的银行 ⭐预留字段，暂时没啥用，无脑0就行了
        }
        self.is_end = False
        self.terminated = False
        self.truncated = False

    def get_day(self):
        return self.day

    def get_enterprise_execute(self):
        return self.action_controller['e_execute']

    def get_bank_execute(self):
        return self.action_controller['b_execute']

    def add_bank(self, config: Bank_config):
        name = config.name
        self.Bank[name] = Bank(config=config)
        if name not in self.action_controller['b_execute']:
            self.action_controller['b_execute'].append(name)

    def add_enterprise_agent(self, config: Enterprise_config):
        name = config.name
        self.Enterprise[name] = Enterprise(config=config)
        if name not in self.action_controller['e_all']:
            self.action_controller['e_all'].append(name)
        if name not in self.action_controller['e_execute']:
            self.action_controller['e_execute'].append(name)
        if name not in self.action_controller['e_seller']:
            self.action_controller['e_seller'].append(name)
        if name not in self.action_controller['e_buyer']:
            self.action_controller['e_buyer'].append(name)
        if config.output_name not in self.shop_dir:
            self.shop_dir.append(config.output_name)

    def add_enterprise_thirdmarket(self, name: str, output_name: str, price: int):

        config = Enterprise_config(name=name, output_name=output_name,
                                   money=float('inf'),
                                   price=price,
                                   intention=0,
                                   stock=float('inf'),
                                   )
        self.Enterprise[name] = Enterprise(config=config)
        if name not in self.action_controller['e_all']:
            self.action_controller['e_all'].append(name)
        if name not in self.action_controller['e_seller']:
            self.action_controller['e_seller'].append(name)
        if output_name not in self.shop_dir:
            self.shop_dir.append(output_name)

    def init(self):
        self.market = Market('market')
        self.trade_time = len(self.Enterprise.keys()) // 2
        # self.logger = Logger(name=self.name, base_path=self.logger_path)

        for key in self.Enterprise:
            self.Enterprise[key].init(shop_dir=self.shop_dir,
                                      e_buyer=self.action_controller['e_buyer'],
                                      e_seller=self.action_controller['e_seller'])
        self.episode = -1
        self.day = 0
        # self.config_output()
        self.e_action = ['WNDF', 'K', "L", "price"]
        self.b_action = self.action_controller['e_execute']  # 银行的决策，本质就是给需要决策的企业借贷，所以动作空间和action_controller[一致

        for key in self.action_controller['e_execute']:
            self.Bank[self.action_controller['b_execute'][0]].observe(self.Enterprise[key])

        for key in self.action_controller['e_execute']:
            self.market.subscribe(self.Enterprise[key])

    def reset(self):
        self.is_end = False
        self.terminated = False  # NEW
        self.truncated = False  # NEW
        self.new_episode()
        self.state = {}
        self.action = {}
        self.day = 0
        self.state = self.new_day(self.episode, self.day,self.lim_day)
        for i in range(len(self.b_action)):
            self.Bank[self.action_controller['b_execute'][0]].set_action(target=self.b_action[i], action=1)  # 设置银行决策动作

        self.run_after_action()
        self.run_before_action()
        return self.state

    # def config_output(self):
    #     block = "\n\n\n\n\n"
    #     for key in self.__dict__:
    #         self.logger.output_config(key + ":" + str(self.__dict__[key]))
    #     self.logger.output_config(block)
    #     for enterprise in self.Enterprise:
    #         res = enterprise + ':\n\n'
    #         for key in self.Enterprise[enterprise].__dict__:
    #             if isinstance(self.Enterprise[enterprise].__dict__[key], dict):
    #                 for detail in self.Enterprise[enterprise].__dict__[key]:
    #                     res = res + ' ' + key + ': ' + str(self.Enterprise[enterprise].__dict__[key][detail]) + '\n'
    #             else:
    #                 res = res + ' ' + key + ': ' + str(self.Enterprise[enterprise].__dict__[key]) + '\n'
    #         self.logger.output_config(res)
    #         self.logger.output_config(block)
    #
    #     for bank in self.Bank:
    #         res = bank + ':\n\n'
    #         for key in self.Bank[bank].__dict__:
    #             if isinstance(self.Bank[bank].__dict__[key], dict):
    #                 for detail in self.Bank[bank].__dict__[key]:
    #                     res = res + ' ' + key + ': ' + str(self.Bank[bank].__dict__[key][detail]) + '\n'
    #             else:
    #                 res = res + ' ' + key + ': ' + str(self.Bank[bank].__dict__[key]) + '\n'
    #         self.logger.output_config(res)
    #         self.logger.output_config(block)

    def new_episode(self):
        self.episode = self.episode + 1
        print("第" + str(self.episode) + "回合开始")
        for key in self.Enterprise.keys():
            self.Enterprise[key].new_episode()
        for key in self.Bank.keys():
            self.Bank[key].new_episode()

    # step 1
    # 一天开始前的准备工作 并获取当天的state
    def new_day(self, episode, day,lim_day):
        # if episode % 10 == 0:
        #     self.logger.output_to_txt(self.market.show_order())
        self.market.new_day()
        b = self.action_controller['b_execute'][0]

        state = {}
        # step 1
        # 所有参与售卖的企业先供给商品到市场
        for key in self.action_controller['e_seller']:
            offer = self.Enterprise[key].offer()  # 企业供给商品，包括 商品类型 商品数量 商品价格
            # 市场接收商品
            self.market.receive(name=key, shop_name=offer['shop_type'], num=offer['num'], price=offer['price'])

        for key in self.action_controller['e_execute']:

            # step 2
            # 执行企业需要得知当前市场上所有自己需要的商品的价格和数量
            for shop_name in self.Enterprise[key].shop_dir:  # 遍历每个企业的shop_dir 得知该企业需要的商品类型有哪些
                shop_list = self.market.answer_all_price(shop_name=shop_name)  # 返回当前商品在市场上的数量和价格的list
                self.Enterprise[key].ask_all_shop(shop_name=shop_name, shop_info=shop_list)  # 传递给企业

            # 执行企业决策前，需要得知当前回合该企业待偿还的本金和利息
            debt = self.Bank[b].answer_debt(name=key, day=day)  # 询问银行当日待偿还利息和本金
            self.Enterprise[key].ask_debt(bill=debt)  # 记下对银行的账单详情
            # 获取本日的state
            self.Enterprise[key].custom_state(day,lim_day)
            state[key] = self.Enterprise[key].get_state()
        for key in self.action_controller['b_execute']:
            self.Bank[key].custom_state(day,lim_day)
            state[key] = self.Bank[key].get_state()  # 获取当前训练用state

        for key in self.Enterprise.keys():
            self.Enterprise[key].new_day(day)
        for key in self.Bank.keys():
            self.Bank[key].new_day(day)

        return state

    def observe(self):
        if self.is_end:
            print("第", self.episode, "回合 存活", self.day, "天")
            wandb.log({"存活天数": self.day})
            self.day_sum += self.day
            self.day_max = max(self.day_max, self.day)
            if self.episode % 100 == 0:
                wandb.log({"每一百回合平均存活天数": self.day_sum / 100})
                wandb.log({"每一百回合最高存活天数": self.day_max})
                self.day_max = 0
                self.day_sum = 0
            if self.episode >= 300:
                self.day_sum_smooth += self.day
                self.day_sum_smooth -= self.eday[self.episode % 300]
                self.eday[self.episode % 300] = self.day
                wandb.log({"每三百回合平均存活天数（平滑）": self.day_sum_smooth / 300})
            else:
                self.day_sum_smooth += self.day
                self.eday[self.episode] = self.day
            # 统计结束数据
            for key in self.action_controller['e_execute']:
                # self.logger.receive_finish_enterprise(episode=self.episode, day=self.day, target=self.Enterprise[key])

                print('after', self.Enterprise[key].total_reward)
            print('after', self.Bank[self.action_controller['b_execute'][0]].total_reward)
            # for key in self.action_controller['b_execute']:
            #     self.logger.receive_finish_bank(episode=self.episode, day=self.day, target=self.Bank[key])
            # if self.use_wandb:
            #     if self.episode % 100 == 0 and self.episode > 0:
            #         self.logger.wandb_log(epi=self.episode)
        info ={
            'terminated': self.terminated,
            'truncated': self.truncated,
        }
        return self.state, self.reward, self.is_end,info

    # 第t+1天开始observe之前run
    def run_before_action(self):
        self.day = self.day + 1
        self.state = self.new_day(self.episode, self.day,self.lim_day)
        self.action = {}

    # 第t天计算reward之后run
    def run_after_action(self):
        # if self.episode % 10 == 0:
        #     self.logger.output_to_txt("第" + str(self.episode) + "回合 第" + str(self.day) + "天：\n")
        #     self.logger.output_to_txt('动作决策' + str(self.action))
        #     self.logger.output_to_txt('state :' + str(self.state))
        b = self.action_controller['b_execute'][0]
        # step 6
        # 自此，企业和银行都完成了决策和动作赋值
        # 接下来，银行开始借钱
        # 首先检查自己的决策，看看剩余储备金够不够借出
        self.Bank[b].check_rent()
        for enterprise_key in self.action_controller['e_execute']:
            rent_val = self.Bank[b].rent(name=enterprise_key, day=self.day)  # 银行借出钱
            self.Enterprise[enterprise_key].deal_rent(rent_val=rent_val)  # 企业处理借款

        # 自此，银行完成了放贷，企业完成了借款
        # 接下来，要开始进行商品交易
        # 目前生产消费各一家企业，如果买不够，则要去第三方市场买,所以交易两次
        # 交易途中不回款，防止同一笔钱反复利用
        for i in range(self.trade_time):
            # step 7
            # 市场从最低价开始交易，先摇获取市场中各个商品最低价
            min_price = {}
            for shop_name in self.shop_dir:
                min_price[shop_name] = self.market.answer_min_price(shop_name=shop_name)

            # step 8
            # 参与购买的企业根据当前最低市场价重新调整自身购买意愿，以便自身买得起
            for key in self.action_controller['e_buyer']:
                self.Enterprise[key].check_intention(shop_price=min_price)  # 根据每个所需商品的最低价格重新调整自身购买意愿
                require_list = self.Enterprise[key].require()  # 获取企业需求
                self.market.get_require(name=key, require_list=require_list)  # 将企业需求提交给市场
            # if self.episode % 10 == 0:
            #     self.logger.output_to_txt(str(self.market))
            # step 9
            #  需求接收完成，市场开始分配商品给买方
            self.market.assign()
            for key in self.action_controller['e_buyer']:
                bill = self.market.get_bill(buyer=key)  # 每个企业获取此次交易的账单
                delta_money = self.Enterprise[key].deal_bill(bill_list=bill)  # 企业处理账单，获取此次变动金额
                self.Bank[b].trade_callback(name=key, delta_money=delta_money)  # 银行处理金钱变动

        # step 10
        # 参与售卖的企业获取回款
        for key in self.action_controller['e_execute']:
            payback = self.market.get_payback(seller=key)  # 获取回款单
            delta_money = self.Enterprise[key].payback(payback_list=payback)  # 获取金钱变动
            self.Bank[b].trade_callback(name=key, delta_money=delta_money)

        # step 11
        # 企业生产
        for key in self.action_controller['e_execute']:
            self.Enterprise[key].product()

        # step 12
        # 银行收回贷款
        for key in self.action_controller['e_execute']:
            self.Bank[b].deal_payback(name=key, payback=self.Enterprise[key].turn_back_money())

        # 每日结束清算
        for key in self.action_controller['e_execute']:
            self.Enterprise[key].daily_settlement()

        # step 13
        # 计算reward
        # 这个reward是上一回合到这一回合过程的reward
        reward = {}
        for key in self.action_controller['e_execute']:
            self.Enterprise[key].custom_reward(self.day,self.lim_day)
            reward[key] = self.Enterprise[key].get_reward()
        for key in self.action_controller['b_execute']:
            self.Bank[key].custom_reward(self.day,self.lim_day)
            reward[key] = self.Bank[key].get_reward()
        self.reward = reward

        # 自然终止（破产）
        self.terminated = any(self.Enterprise[key].is_falled() for key in self.action_controller['e_execute'])
        self.is_end = self.is_end or self.terminated
        # 第一天不进行动作决策，按照固定操作开局
        # if self.day > 0:
        #     # 统计数据，一定要在这一步，在step15后部分数据更新为次日数据
        #     for key in self.action_controller['e_execute']:
        #         self.logger.receive_enterprise(episode=self.episode, day=self.day, target=self.Enterprise[key])
        #
        #     for key in self.action_controller['b_execute']:
        #         self.logger.receive_bank(episode=self.episode, day=self.day, target=self.Bank[key])

            # step 14
            # 破产清算

            # 如果day == self.lim_day-1 说明到了最后一天
            # 此时理应将feed_back中的is_end设为True
            # 但如果有人最后一个天破产
            # 那么在step 16时将会还有一条is_end为True的带惩罚reward的数据
            # 这种情况is_end就为False

        # step 16
        # 但是如果有人破产
        # 则进行结算
        if self.is_end:
            reward = {}
            for key in self.action_controller['e_execute']:
                # 企业破产清算
                reward[key] = self.Enterprise[key].get_fail_reward()
            # 银行同理
            for key in self.action_controller['b_execute']:
                reward[key] = self.Bank[key].get_fail_reward()
            self.reward = reward

        # 输出数据
        # if self.episode % 10 == 0:
        #     self.logger.output_to_txt(str(self))

            # print(self)

    # 输入为{主体名字str:[]list}
    # 如：{'consumer':[1,2,3,...,n],'producer':[2,2,3,..,n],'bank1':[1,2,3,...,m]}
    def step(self, action: dict):
        b = self.action_controller['b_execute'][0]
        self.action = action
        if self.day > 0:
            for key in self.action_controller['e_execute']:
                # step 4
                # 自此，决策前准备已经就绪，开始决策
                for i in range(len(self.e_action)):
                    self.Enterprise[key].set_action(target=self.e_action[i], action=action[key][i])  # 逐个设置当前回合的决策
                # 统计决策动作数据
                # self.logger.receive_action(name=key, action=action[key], action_detail=self.e_action,
                #                            episode=self.episode)
            for i in range(len(self.b_action)):
                self.Bank[b].set_action(target=self.b_action[i], action=action[b][i])  # 设置银行决策动作
            # self.logger.receive_action(name=b, action=action[b], action_detail=self.b_action, episode=self.episode)
        self.run_after_action()
        self.run_before_action()
        if self.day >= self.lim_day - 1:
            self.truncated = True
            self.is_end = True


    def check_and_get_bankruptcy_count(self):
        """检查并返回本回合破产的企业数量"""
        bankruptcy_count = 0
        for key in self.Enterprise:
            if self.Enterprise[key].is_falled():
                bankruptcy_count += 1
        return bankruptcy_count


    # def finish(self):
    #
    #     print("logger.finish")
    #     self.logger.finish()
    #     print("logger.show")
    #     self.logger.show_all()
    #     print("logger.tocsv")
    #     self.logger.to_csv()
    #     if self.use_wandb:
    #         wandb.finish()

    def __str__(self):
        res = ''
        for enterprise in self.action_controller['e_execute']:
            res += str(self.Enterprise[enterprise])

        for bank in self.action_controller['b_execute']:
            res += str(self.Bank[bank])

        return res
