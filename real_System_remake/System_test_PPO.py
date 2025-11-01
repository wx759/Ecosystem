# import tensorflow as tf
import copy
import gc
import json
from fileinput import filename
from datetime import datetime
import numpy as np
import os
import random
import torch

import swanlab as wandb
from Agent.Config_PPO import Config_PPO
from real_System_remake.Bank_config import Bank_config
from real_System_remake.Enterprise_config import Enterprise_config
from real_System_remake.Environment import Environment
from real_System_remake.ppo_bank import bank_nnu
from real_System_remake.ppo_enterprise import enterprise_nnu
import torch
from torch.distributions import Normal
import torch.nn.functional as F

use_wandb = True
use_rbtree = False
lim_day = 500
enterprise_ppo_config = Config_PPO(
    scope='',
    state_dim=0,
    action_dim=4,
    hidden_dim=128,
    # 第一次调用时再初始化agent，以便动态适应状态空间)
)

bank_ppo_config = Config_PPO(
    scope='',
    state_dim=0,
    action_dim=2,
    hidden_dim=128,

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
    price=8.0, intention=5.0)

# 两个企业，一个生产K，一个生产L
enterprise_add_list = {
    'production1': 'K',
    'consumption1': 'L'
}


class System:
    def __init__(self):
        self.env = Environment(name='PPO', lim_day=lim_day)

        for key in enterprise_add_list:
            config = copy.deepcopy(enterprise_config)
            config.name = key
            config.output_name = enterprise_add_list[key]
            self.env.add_enterprise_agent(config=config)
        self.env.add_bank(bank_config)
        self.env.add_enterprise_thirdmarket(name='production_thirdMarket', output_name='K', price=100)
        self.env.add_enterprise_thirdmarket(name='consumption_thirdMarket', output_name='L', price=100)

        self.env.init()
        # self.epiday=0 #回合数，在算法太垃圾的时候可以提前结束。
        self.e_execute = self.env.get_enterprise_execute()
        self.b_execute = self.env.get_bank_execute()
        self.execute = self.e_execute + self.b_execute
        self.Agent = {}
        for key in self.execute:
            self.Agent[key] = None

    def run(self):
        seed = random.randint(0, 1000)
        config = Config_PPO(scope='', state_dim=0, action_dim=0, hidden_dim=0)
        wandb.init(project="TD3_vs_PPO", workspace="wx829", config={
            "random_seed": seed,
            "is_rms_state": config.is_rms_state,
            "is_rms_reward": config.is_rms_reward,
            "max_training_steps": config.MAX_TRAINING_STEPS,
            "total_step": config.total_step,
            "learning_rate_actor_enterprise": config.LEARNING_RATE_AC_Enterprise,
            "learning_rate_actor_bank": config.LEARNING_RATE_AC_Bank,
            "learning_rate_critic_enterprise": config.LEARNING_RATE_C_Enterprise,
            "learning_rate_critic_bank": config.LEARNING_RATE_C_Bank,
            "entropyRC_Enterprise": config.entropyRC_Enterprise,
            "entropyRC_Bank": config.entropyRC_Bank,
            "clip_range": config.CLIP_RANGE,
            "epoch": config.N_EPOCHS,
            "mini_batch": config.MINI_BATCH_SIZE,
            "update_timestep": config.UPDATE_TIMESTEP,
            "total_update": config.MAX_TRAINING_STEPS / config.UPDATE_TIMESTEP,
            "lim-day": lim_day
        })
        # 1. PPO 超参数
        update_timestep = config.UPDATE_TIMESTEP
        # max_training_timesteps = config.MAX_TRAINING_STEPS
        total_step =config.total_step
        # 2. 初始化智能体
        _temp_state = self.env.reset()
        for target_key in self.e_execute:
            if self.Agent[target_key] is None:
                config = copy.deepcopy(enterprise_ppo_config)
                config.set_scope(target_key)
                config.set_seed(seed)
                config.set_state_dim(len(_temp_state[target_key]))
                self.Agent[target_key] = enterprise_nnu(config)
        for target_key in self.b_execute:
            if self.Agent[target_key] is None:
                config = copy.deepcopy(bank_ppo_config)
                config.set_scope(target_key)
                config.set_seed(seed)
                config.set_state_dim(len(_temp_state[target_key]))
                self.Agent[target_key] = bank_nnu(config)

        # self.load_actor_only(save_dir="actors_only", note="")
        # 3. 开始训练循环
        state = self.env.reset()
        time_step = 0
        episode_num = 0

        while time_step < total_step:

            # --- 数据收集阶段 ---
            for _ in range(update_timestep):
                time_step += 1
                action, log_prob = {}, {}

                for target_key in self.e_execute:
                    act, lp = self.Agent[target_key].choose_action(state[target_key])
                    action[target_key], log_prob[target_key] = act, lp
                for target_key in self.b_execute:
                    act, lp = self.Agent[target_key].choose_action(state[target_key])
                    action[target_key], log_prob[target_key] = act, lp

                self.env.step(action)
                next_state, reward, done_env,info = self.env.observe()

                # NEW: 为每个 agent 计算 next_value = V(next_state)
                next_v = {}
                for k in self.e_execute:
                    next_v[k] = self.Agent[k].get_value(next_state[k])
                for k in self.b_execute:
                    next_v[k] = self.Agent[k].get_value(next_state[k])

                # NEW: 双掩码
                is_terminal_for_gae = bool(info.get('terminated', done_env))  # 自然终止才截断 bootstrap
                nonterminal = 0 if done_env else 1  # 结束(terminated 或 truncated)则断开 GAE 递推

                # CHANGED: 存 transition（含 next_value 与 nonterminal）
                for target_key in self.e_execute:
                    self.Agent[target_key].store_transition(
                        state[target_key],
                        action[target_key],
                        log_prob[target_key],
                        reward[target_key]['business'],
                        is_terminal_for_gae,
                        next_v[target_key],
                        nonterminal,
                    )
                for target_key in self.b_execute:
                    self.Agent[target_key].store_transition(
                        state[target_key],
                        action[target_key],
                        log_prob[target_key],
                        reward[target_key]['WNDB'],
                        is_terminal_for_gae,
                        next_v[target_key],
                        nonterminal,
                    )

                state = next_state

                if done_env:
                    state = self.env.reset()
                    episode_num += 1
                    print(f"Episode {episode_num} finished. Total timesteps: {time_step}")

            # --- 学习阶段 ---
            print(f"--- Timestep {time_step}. Updating policies... ---")

            for agent_key, agent in self.Agent.items():
                agent.learn(state[agent_key])
                agent.clear_memory()

            if use_wandb:
                critic_bank, actor_bank, avg_entropy_bank, clip_fraction_bank = self.Agent['bank1'].log()
                critic_production1, actor_production1, avg_entropy_production1, clip_fraction_production1 = self.Agent[
                    'production1'].log()
                critic_consumption1, actor_consumption1, avg_entropy_consumption1, clip_fraction_consumption1 = \
                    self.Agent['consumption1'].log()
                wandb.log({'actor_loss/bank1': actor_bank})
                wandb.log({'actor_loss/production1': actor_production1})
                wandb.log({'actor_loss/consumption1': actor_consumption1})

                wandb.log({'critic_loss/bank': critic_bank})
                wandb.log({'critic_loss/production1': critic_production1})
                wandb.log({'critic_loss/consumption1': critic_consumption1})

                wandb.log({'avg_entropy/bank': avg_entropy_bank})
                wandb.log({'avg_entropy/production1': avg_entropy_production1})
                wandb.log({'avg_entropy/consumption1': avg_entropy_consumption1})

                wandb.log({'clip_fraction/bank': clip_fraction_bank})
                wandb.log({'clip_fraction/production1': clip_fraction_production1})
                wandb.log({'clip_fraction/consumption1': clip_fraction_consumption1})

            print("--- Update finished. ---")

        print("--- Training finished. Starting evaluation... ---")
        # for i in range(100):
        #     avg_len_det = self.evaluate_policy(episodes=10)
        #     wandb.log({"eval/final_avg_len_det": avg_len_det})

        # self.env.finish()

    @staticmethod
    def set_seed(seed):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        os.environ["PYTHONHASHSEED"] = str(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        print(f"[INFO] Seed set to {seed}")

    # def evaluate_policy(self, episodes=10):
    #     lengths = []
    #
    #     for ep in range(episodes):
    #         state = self.env.reset()
    #         done = False
    #         steps = 0
    #
    #         while not done:
    #             action = {}
    #
    #             # 和 run 一样，按 target_key 取 state 的子部分
    #             for target_key in self.e_execute:
    #                 act = self.Agent[target_key].choose_action_deterministic(
    #                     state[target_key]
    #                 )
    #                 action[target_key] = act
    #
    #             for target_key in self.b_execute:
    #                 act= self.Agent[target_key].choose_action_deterministic(
    #                     state[target_key]
    #                 )
    #                 action[target_key] = act
    #
    #             # 环境交互
    #             self.env.step(action)
    #             next_state, reward, done = self.env.observe()
    #             steps += 1
    #
    #         lengths.append(steps)
    #     avg_len = np.mean(lengths)
    #     return avg_len
    def evaluate_policy(self, episodes=20, deterministic=False, threshold=None, use_ema=False):
        """
        对当前训练好的策略进行评估。
        参数：
            episodes: 评估回合数
            deterministic: 是否使用确定性策略（mu）
            threshold: 若某回合存活天数 < threshold，则记录失败轨迹
            use_ema: 是否使用 EMA 平滑权重（可选）
        """

        results = []
        failure_records = []

        # if use_ema:
        #     for key in self.Agent:
        #         if hasattr(self.Agent[key], "ema_params"):
        #             load_ema_to_model(self.Agent[key].actor, self.Agent[key].ema_params)

        print(f"\n开始评估: episodes={episodes}, deterministic={deterministic}, EMA={use_ema}")

        for ep in range(episodes):
            state = self.reset_with_noise()
            done = False
            day = 1
            trajectory = []

            while not done:
                action = {}

                # ======================
                # 为每类 Agent 选择动作
                # ======================
                for target_key in self.e_execute:
                    act = self.Agent[target_key].choose_action_deterministic(state[target_key])
                    action[target_key] = act

                for target_key in self.b_execute:
                    act = self.Agent[target_key].choose_action_deterministic(state[target_key])
                    action[target_key] = act

                # 环境步进
                self.env.step(action)
                next_state, reward, done = self.env.observe()

                trajectory.append({
                    "day": day,
                    "state": {k: state[k].tolist() if hasattr(state[k], 'tolist') else state[k] for k in state},
                    "action": {k: action[k].tolist() for k in action},
                    "reward": reward
                })

                state = next_state
                day += 1

            results.append(day)
            print(f"  Episode {ep + 1}: survived {day} days")

            # 记录失败轨迹
            if threshold and day < threshold:
                failure_records.append(trajectory)

        # ========== 汇总统计 ==========
        results = np.array(results)
        res = {
            "mean": float(np.mean(results)),
            "std": float(np.std(results)),
            "median": float(np.median(results)),
            "min": int(np.min(results)),
            "max": int(np.max(results)),
            "pct5": float(np.percentile(results, 5)),
            "pct95": float(np.percentile(results, 95))
        }

        print("\n=== 评估结果 ===")
        for k, v in res.items():
            print(f"{k}: {v}")

        # ========== 保存失败轨迹 ==========
        if failure_records:
            filename = f"failures_eval_seed_200{self.seed if hasattr(self, 'seed') else 0}.json"
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(failure_records, f, ensure_ascii=False, indent=2)
            print(f"失败轨迹已保存到 {filename}")

        return res

    def collect_state_statistics(self, episodes=20):
        """
        用最终策略采样若干条轨迹，返回每个 agent 每维 mean 和 std（flatten 后）。
        结果格式：dict(agent_key -> {'mean': np.array, 'std': np.array})
        """
        all_states_per_agent = {k: [] for k in self.Agent.keys()}

        for ep in range(episodes):
            state = self.env.reset()
            done = False
            while not done:
                # 随机/确定地用当前训练好策略采样动作（建议使用训练时的采样行为）
                action = {}
                for target_key in self.e_execute:
                    act, _ = self.Agent[target_key].choose_action(state[target_key])
                    action[target_key] = act

                for target_key in self.b_execute:
                    act = self.Agent[target_key].choose_action(state[target_key])
                    action[target_key] = act
                self.env.step(action)
                next_state, reward, done = self.env.observe()
                # 收集每个 agent 的 state（按 list/array 格式）
                for key in self.Agent.keys():
                    arr = np.array(next_state[key], dtype=np.float32).ravel()
                    all_states_per_agent[key].append(arr)
                state = next_state

        stats = {}
        for key, list_of_states in all_states_per_agent.items():
            if len(list_of_states) == 0:
                continue
            S = np.stack(list_of_states, axis=0)  # [T, D]
            stats[key] = {'mean': S.mean(axis=0), 'std': S.std(axis=0)}
        return stats

    def add_multiplicative_noise_to_state_vector(vec, std_vec, alpha):
        """
        vec: 1D numpy array (state flattened)
        std_vec: 1D numpy array same shape (per-dim std from collect)
        alpha: scalar factor (noise level relative to std)
        返回： noisy 1D array
        使用乘法噪声 s' = s * (1 + eps), eps ~ N(0, alpha * std_rel)
        这里用相对 std: std_rel = std / (|mean|+eps) 也可直接用 std。
        """
        eps_small = 1e-8
        # 若 std_vec 中有 0，退化到一个小常数
        noise_sigma = alpha * (std_vec + eps_small)
        eps = np.random.normal(0.0, noise_sigma, size=vec.shape)
        return (vec * (1.0 + eps)).astype(np.float32)

    def reset_with_noise(self, noise_scale=0.2):
        """
        带初始状态扰动的 reset
        对状态向量中的每个 agent 添加微小高斯噪声
        """
        state = self.env.reset()
        noisy_state = {}

        for key, value in state.items():
            # 如果是 list，先转成 np.array
            if isinstance(value, list):
                value = np.array(value, dtype=np.float32)

            # 如果是 numpy 数组，添加噪声
            if isinstance(value, np.ndarray):
                noise = np.random.normal(0, noise_scale, size=value.shape)
                noisy_value = value + noise
                noisy_state[key] = noisy_value.tolist()  # 转回 list，保证环境兼容
            else:
                # 对非数组（如标量、字典）保持原样
                noisy_state[key] = value

        return noisy_state

    def save_actors(self, save_dir="actors_only", note=""):
        """
        保存所有智能体的 Actor 参数（仅用于评估）
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_dir = save_dir + "_" + timestamp
        os.makedirs(save_dir, exist_ok=True)
        for target_key in self.e_execute:
            agent = self.Agent[target_key]
            filename = f"{save_dir}/{agent.scope}_actor.pt"
            torch.save(agent.enterprise.actor.state_dict(), filename)
            print(f"[🎯] 已保存 {agent.scope} 的 actor 至 {filename}")

        for target_key in self.b_execute:
            agent = self.Agent[target_key]
            filename = f"{save_dir}/{agent.scope}_actor.pt"
            torch.save(agent.bank.actor.state_dict(), filename)
            print(f"[🎯] 已保存 {agent.scope} 的 actor 至 {filename}")

    def load_actor_only(self, save_dir = "actors_only",note=""):
        for target_key in self.e_execute:
            agent = self.Agent[target_key]
            path = os.path.join(save_dir, f"{agent.scope}_actor_{note}.pt")
            if os.path.exists(path):
                agent.enterprise.actor.load_state_dict(torch.load(path))
                agent.enterprise.actor.eval()
                print(f"[🎯] 加载 actor: {agent.scope}")

        for target_key in self.b_execute:
            agent = self.Agent[target_key]
            path = os.path.join(save_dir, f"{agent.scope}_actor_{note}.pt")
            if os.path.exists(path):
                agent.bank.actor.load_state_dict(torch.load(path))
                agent.bank.actor.eval()
                print(f"[🎯] 加载 actor: {agent.scope}")

if __name__ == '__main__':
    # for i in range(3):
    system = System()

    system.run()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    # system.save_actors()
    # system.evaluate_policy(episodes=500, deterministic=False, threshold=180)
    del system
    gc.collect()
    # 清空计算图
    torch.nn.Module.dump_patches = True
    torch.cuda.empty_cache()

    # tf.reset_default_graph()
