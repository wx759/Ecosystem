# 历史遗留代码 不要管

from real_System_remake.Environment import Environment
from Agent.Config import Config
import tensorflow as tf
import atexit
import random
import gc

stable_at = 80000
end_at = 100000


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
        action_dim=2,
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
    episodes= [10000,10000,10000,10000,10000,10000]

    for i in episodes:

        # enterprise_ddpg_config['economy'].set_learning_rate(learning_rate_actor=i,learning_rate_critic=i * 2)
        # enterprise_ddpg_config['business'].set_learning_rate(learning_rate_actor=i,learning_rate_critic=i * 2)
        seed = random.randint(1, 1000)
        # enterprise_ddpg_config['economy'].set_seed(seed)
        enterprise_ddpg_config['business'].set_seed(seed)
        bank_ddpg_config['WNDB'].set_seed(seed)
        env = Environment(name="FINALTD3重构前remake;reward利润加营业额;二层网络128：32;" + "_", episodes=i,
                        base_fund=2000, break_epi_mul_day=200000, producer_random=False, consumer_random=False, bank_random=False,
                        bank_ddpg_config=bank_ddpg_config, enterprise_ddpg_config=enterprise_ddpg_config,debt_time=5,
                        is_delay_feed_back=True,has_third_market=True)
        env.run()

        del env
        gc.collect()
        tf.reset_default_graph()






