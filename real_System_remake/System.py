from real_System_remake.Environment import Environment
from real_System_remake.Enterprise_config import Enterprise_config
from real_System_remake.Bank_config import Bank_config
from real_System_remake.ddpg_enterprise import enterprise_nnu
from real_System_remake.ddpg_bank import bank_nnu
# from real_System_remake.td3_enterprise_lstm  import enterprise_nnu
# from real_System_remake.td3_bank_lstm import bank_nnu
# from real_System_remake.td3_enterprise_rbtree  import enterprise_nnu
# from real_System_remake.ddpg_bank import bank_nnu
# from real_System_remake.td3_enterprise_Newlstm  import enterprise_nnu
# from real_System_remake.td3_bank_Newlstm import bank_nnu
from Agent.Config import Config
from real_System_remake.Logger import Logger
#import tensorflow as tf
import copy
import atexit
import random
import gc
import torch
import torch.nn as nn
import wandb
use_wandb = False
stable_at = 80000
end_at = 100000
use_rbtree = False
# Notice 如果修改lstm的隐藏层节点数量，需要去经验池get batch函数里同步修改
enterprise_ddpg_config = Config(
        scope='',
        action_dim=4,
        action_bound=0.5,
        var_stable_at=stable_at,
        var_end_at=end_at,
        learning_rate_actor=1e-3,
        learning_rate_critic=2e-3,
        learning_rate_decay=1,
        random_seed=184,
        batch_size=3000,
        memory_capacity=80000,
        smooth_noise=0.01,
        is_QNet_smooth_critic=True,
        soft_replace_tau=0.01,
        actor_update_delay_times=3,
        policy_noise=0.2,
        max_hist_len=8,
        batch_lstm=400
    # state_dim=,  # 第一次调用时再初始化agent，以便动态适应状态空间
)

bank_ddpg_config =Config(
        scope='',
        action_dim=2,
        action_bound=0.5,
        var_stable_at=stable_at,
        var_end_at=end_at,
        learning_rate_actor=1e-3,
        learning_rate_critic=2e-3,
        learning_rate_decay=1,
        random_seed=184,
        batch_size=3000,
        memory_capacity=80000,
        smooth_noise=0.01,
        is_QNet_smooth_critic=True,
        soft_replace_tau=0.01,
        actor_update_delay_times=3,
        policy_noise=0.2,
        max_hist_len=8,
        batch_lstm=400
)

bank_config = Bank_config(
    name='bank1',
    fund=2000,
    fund_rate=1,
    fund_increase=0.1,
    debt_time=5
)

enterprise_config = Enterprise_config(
    name='',
    output_name='',
    price=8.0,intention=5.0)

enterprise_add_list = {
    'production1':'K',
    'consumption1':'L'
}



class System:
    def __init__(self):
        self.env = Environment(name='TD3', lim_day=100)
        # self.env = Environment(name='TD3_lim_day=200', lim_day=200)
        for key in enterprise_add_list:
            config = copy.deepcopy(enterprise_config)
            config.name = key
            config.output_name = enterprise_add_list[key]
            self.env.add_enterprise_agent(config=config)
        self.env.add_bank(bank_config)
        self.env.add_enterprise_thirdmarket(name='production_thirdMarket', output_name='K', price=100)
        self.env.add_enterprise_thirdmarket(name='consumption_thirdMarket', output_name='L', price=100)

        self.env.init()
        self.epiday=0
        self.e_execute = self.env.get_enterprise_execute()
        self.b_execute = self.env.get_bank_execute()
        self.execute = self.e_execute + self.b_execute
        self.Agent = {}
        for key in self.execute:
            self.Agent[key] = None

    def run(self):

        for episode in range(10000):
            if self.epiday>200000 and episode%100==0:
                break
            state = self.env.reset()
            last_state = None
            last_action = None
            # 第一次进来的话，构建enterprise智能体集合
            for target_key in self.e_execute:
                if self.Agent[target_key] is None:
                    config = copy.deepcopy(enterprise_ddpg_config)
                    config.set_scope(target_key)
                    config.set_state_dim(len(state[target_key]))
                    self.Agent[target_key] = enterprise_nnu(config)
            # 第一次进来的话，构建bank智能体集合
            for target_key in self.b_execute:
                if self.Agent[target_key] is None:
                    config = copy.deepcopy(bank_ddpg_config)
                    config.set_scope(target_key)
                    config.set_state_dim(len(state[target_key]))
                    self.Agent[target_key] = bank_nnu(config)
            new_ep = True
            while True:
                action = {}
                for target_key in self.e_execute:
                    action[target_key] = self.Agent[target_key].run_enterprise(state[target_key], new_ep)
                for target_key in self.b_execute:
                    action[target_key] = self.Agent[target_key].run_bank(state[target_key], new_ep)
                new_ep = False
                self.epiday=self.epiday+1

                self.env.step(action)
                next_state,reward,done,_= self.env.observe()

                # done的情况下，因为已知state 和 state_，reward为破产惩罚，处理逻辑不需要时序错峰
                if done:
                    for target_key in self.e_execute:
                        self.Agent[target_key].env_upd(state=state[target_key],
                                                       action=action[target_key],
                                                       state_=next_state[target_key],
                                                       reward=reward[target_key]['business'],
                                                       is_train=True,
                                                       is_end=done)
                        # 更新银行feedback 其实和上面一样，但是为了方便以后可能要拓展先区分开来
                    for target_key in self.b_execute:
                        self.Agent[target_key].env_upd(state=state[target_key],
                                                       action=action[target_key],
                                                       state_=next_state[target_key],
                                                       reward=reward[target_key]['WNDB'],
                                                       is_train=True,
                                                       is_end=done)
                    break
                else:
                    # 关键一步 时序错峰，详见时序错峰.png
                    if last_state is not None:
                        # 更新企业feedback
                        for target_key in self.e_execute:
                            self.Agent[target_key].env_upd(state=last_state[target_key],
                                                              action = last_action[target_key],
                                                              state_ = state[target_key],
                                                              reward = reward[target_key]['business'],
                                                              is_train = True,
                                                              is_end = done)
                        # 更新银行feedback 其实和上面一样，但是为了方便以后可能要拓展先区分开来
                        for target_key in self.b_execute:
                            self.Agent[target_key].env_upd(state=last_state[target_key],
                                                              action=last_action[target_key],
                                                              state_=state[target_key],
                                                              reward=reward[target_key]['WNDB'],
                                                              is_train = True,
                                                              is_end=done)
                if use_wandb:

                    var, critic_bank, actor_bank = self.Agent['bank1'].log()
                    _ , critic_production1, actor_production1 = self.Agent['production1'].log()
                    _ , crtic_cumsuption1, actor_consumption1 = self.Agent['consumption1'].log()
                    wandb.log({'actor_loss/bank1':actor_bank})
                    wandb.log({'actor_loss/production1':actor_production1})
                    wandb.log({'actor_loss/consumption1': actor_consumption1})
                    wandb.log({'critic_loss/bank':critic_bank})
                    wandb.log({'critic_loss/production1':critic_production1})
                    wandb.log({'critic_loss/consumption1':crtic_cumsuption1})
                    wandb.log({'探索噪声var':var})

                # 在一轮决策和feedback结束之后，反转对方数据存入缓冲池
                if use_rbtree:
                    if done:
                        self.Agent['production1'].env_upd_rbtree(state=state['consumption1'],
                                                           action=action['consumption1'],
                                                           state_=next_state['consumption1'],
                                                           reward=reward['consumption1']['business'])
                        self.Agent['consumption1'].env_upd_rbtree(state=state['production1'],
                                                                 action=action['production1'],
                                                                 state_=next_state['production1'],
                                                                 reward=reward['production1']['business'])
                        break
                    else:
                        if last_state is not None:
                            self.Agent['production1'].env_upd_rbtree(state=last_state['consumption1'],
                                                           action=last_action['consumption1'],
                                                           state_=state['consumption1'],
                                                           reward=reward['consumption1']['business'])
                            self.Agent['consumption1'].env_upd_rbtree(state=last_state['production1'],
                                                              action=last_action['production1'],
                                                              state_=state['production1'],
                                                              reward=reward['production1']['business'])

                last_state = state
                state = next_state
                last_action = action



        self.env.finish()


if __name__ == '__main__':
    for i in range(3):

        system = System()
        system.run()

        del system
        gc.collect()
        # 清空计算图
        torch.nn.Module.dump_patches = True
        torch.cuda.empty_cache()

        #tf.reset_default_graph()






