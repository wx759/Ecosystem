from real_System.System_mulAgent import System
from Agent.Config import Config
import tensorflow as tf
import atexit
import random
import gc

stable_at = 80000
end_at = float('inf')


enterprise_ddpg_config = {

    'business':Config(
        scope='',
        action_dim=4,
        action_bound=0.5,
        var_stable_at=stable_at,
        var_end_at=end_at,
        learning_rate_actor=0.001,
        learning_rate_critic=0.002,
        learning_rate_decay=1,
        random_seed=184,
        batch_size=3000,
        memory_capacity=80000,
        smooth_noise=0.01,
        is_QNet_smooth_critic=True,


    # state_dim=,  # 第一次调用时再初始化agent，以便动态适应状态空间
)
}

bank_ddpg_config ={
    'WNDB':Config(
        scope='',
        action_dim=4,
        action_bound=0.5,
        var_stable_at=stable_at,
        var_end_at=end_at,
        learning_rate_actor=0.001,
        learning_rate_critic=0.002,
        learning_rate_decay=1,
        random_seed=184,
        batch_size=3000,
        memory_capacity=80000,
        is_QNet_smooth_critic=True,

    # state_dim=,

)
}



if __name__ == '__main__':
    episodes= [30000,30000]
    # for i in episodes:
    #    system = System(name="加第三方市场;加0.8倍第三方市场倾销;储备金6000;储备金率1.01;生产率2.5;生产min;探索loop;reward为收入支出2：1;"
    #                         "时序延迟;首日不决策;去掉bn;产出为0则破产;两层网络20：5;学习率e-3;回合数" + str(i) + "_", episodes=i,
    #                    bank_ddpg_config=bank_ddpg_config, enterprise_ddpg_config=enterprise_ddpg_config,
    #                    is_delay_feed_back=True)
    #    system.run()
    #    tf.reset_default_graph()

    # for i in episodes:
    #     system = System(name="CHECK_reward为收入支出2：1;时序延迟;手动归一化;环境固定,固定价格9.99,只有producter决策;双智能体决策;relu;加第三方市场;加0.5倍第三方市场倾销;生产率2.5;生产min;探索loop;"
    #                          "首日不决策;产出为0则破产;一层网络20：5;"+ "_", episodes=i,
    #                     base_fund=2000,break_epi_mul_day=100000,producer_random=False,consumer_random=True,bank_random=True,
    #                     bank_ddpg_config=bank_ddpg_config, enterprise_ddpg_config=enterprise_ddpg_config,
    #                     is_delay_feed_back=True)
    #     system.run()
    #     tf.reset_default_graph()
    for i in episodes:
        # enterprise_ddpg_config['economy'].set_learning_rate(learning_rate_actor=i,learning_rate_critic=i * 2)
        # enterprise_ddpg_config['business'].set_learning_rate(learning_rate_actor=i,learning_rate_critic=i * 2)
        seed = random.randint(1, 1000)
        # enterprise_ddpg_config['economy'].set_seed(seed)
        enterprise_ddpg_config['business'].set_seed(seed)
        bank_ddpg_config['WNDB'].set_seed(seed)
        system = System(name="主体数量2;FINALTD3三方定价10;reward利润加收入;二层网络128：32;" + "_", episodes=i,
                        base_fund=2000, break_epi_mul_day=500000, producer_random=False, consumer_random=False,
                        bank_random=False,
                        bank_ddpg_config=bank_ddpg_config, enterprise_ddpg_config=enterprise_ddpg_config, debt_time=5,
                        is_delay_feed_back=True, has_third_market=True)
        system.run()

        del system
        gc.collect()
        tf.reset_default_graph()

    # for i in episodes:
    #
    #     # enterprise_ddpg_config['economy'].set_learning_rate(learning_rate_actor=i,learning_rate_critic=i * 2)
    #     # enterprise_ddpg_config['business'].set_learning_rate(learning_rate_actor=i,learning_rate_critic=i * 2)
    #     seed = random.randint(1, 1000)
    #     # enterprise_ddpg_config['economy'].set_seed(seed)
    #     enterprise_ddpg_config['business'].set_seed(seed)
    #     bank_ddpg_config['WNDB'].set_seed(seed)
    #     system = System(name="主体数量2;FINALTD3无第三方且无库存折算;reward利润加收入;二层网络128：32;" + "_", episodes=i,
    #                     base_fund=2000, break_epi_mul_day=150000, producer_random=False, consumer_random=False, bank_random=False,
    #                     bank_ddpg_config=bank_ddpg_config, enterprise_ddpg_config=enterprise_ddpg_config,debt_time=5,
    #                     is_delay_feed_back=True,has_third_market=False)
    #     system.run()
    #
    #     del system
    #     gc.collect()
    #     tf.reset_default_graph()




    # for i in episodes:
    #     system = System(name="BASELINE_环境固定,只有production决策;relu;储备金1000;储备金率1.1;去除critic_BN;加第三方市场;加0.5倍第三方市场倾销;生产率2.5;生产min;探索loop;reward为收入支出2：1;"
    #                          "时序延迟;首日不决策;产出为0则破产;两层网络20：5;学习率e-4;回合数" + str(i) + "_", episodes=i,
    #                     bank_ddpg_config=bank_ddpg_config, enterprise_ddpg_config=enterprise_ddpg_config,
    #                     is_delay_feed_back=True)
    #     system.run()
    #     tf.reset_default_graph()
    #
    # for i in episodes:
    #     system = System(name="加第三方市场;加0.8倍第三方市场倾销;生产率2.5;生产min;探索loop;reward为金融利润;减少动作空间;提高初始价格;"
    #                          "时序延迟;首日不决策;拉长训练次数;产出为0则破产;两层网络20：5;学习率e-4;回合数" + str(i) + "_", episodes=i,
    #                     bank_ddpg_config=bank_ddpg_config, enterprise_ddpg_config=enterprise_ddpg_config,
    #                     is_delay_feed_back=True)
    #     system.run()
    #     tf.reset_default_graph()

