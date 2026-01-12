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

import swanlab
from Agent.Config_PPO import Config_PPO
from real_System_remake.Bank_config import Bank_config
from real_System_remake.Enterprise_config import Enterprise_config
from real_System_remake.Environment import Environment
from real_System_remake.ppo_bank import bank_nnu
from real_System_remake.ppo_enterprise import enterprise_nnu
import torch
from torch.distributions import Normal
import torch.nn.functional as F
from swanlab.plugin.notification import EmailCallback
use_wandb = True
use_rbtree = False
lim_day = 100
# seed =125
enterprise_ppo_config = Config_PPO(
    scope='',
    state_dim=0,
    action_dim=4,
    hidden_dim=64,
)

bank_ppo_config = Config_PPO(
    scope='',
    state_dim=0,
    action_dim=2,
    hidden_dim=64,

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
        # 评估配置项
        self.eval_interval_steps = 10000
        self.eval_deterministic = True

    #新增“构建独立环境”的函数
    def _build_env(self, name: str):
        env = Environment(name=name, lim_day=lim_day)

        for key in enterprise_add_list:
            config = copy.deepcopy(enterprise_config)
            config.name = key
            config.output_name = enterprise_add_list[key]
            env.add_enterprise_agent(config=config)

        env.add_bank(bank_config)
        env.add_enterprise_thirdmarket(name='production_thirdMarket', output_name='K', price=100)
        env.add_enterprise_thirdmarket(name='consumption_thirdMarket', output_name='L', price=100)

        env.init()
        return env

    #新增窗口快照与恢复函数
    def _snapshot_agent_windows(self):
        snap = {}
        for k, agent in self.Agent.items():  #k是智能体名字
            if hasattr(agent, "get_window_state"):
                snap[k] = agent.get_window_state()
            else:
                snap[k] = None
        return snap

    def _restore_agent_windows(self, snap):
        for k, agent in self.Agent.items():
            if hasattr(agent, "set_window_state"):
                agent.set_window_state(snap.get(k))

    def evaluate_current_policy(self, steps: int, eval_episodes: int = 50, deterministic: bool = False):
        """
        每次调用：用独立 eval_env 跑 self.eval_episodes 回合。
        记录：
          - 平均存活天数
          - 破产率（terminated 占比）
          - 综合收益：按你选的 A 口径，分别记录 enterprise 的 total_reward['eval_business'] / ['business']，以及 bank 的 total_reward（尽量兼容）
        """
        import numpy as np

        # 1) 保存训练窗口（关键）
        window_snap = self._snapshot_agent_windows()

        # 2) 评估前清空窗口（每个 eval episode 都从干净窗口开始）
        for agent in self.Agent.values():
            if hasattr(agent, "reset_window"):
                agent.reset_window()

        for agent in self.Agent.values():
            if hasattr(agent, "enterprise"):
                agent.enterprise.actor.eval()
                agent.enterprise.critic.eval()
            if hasattr(agent, "bank"):
                agent.bank.actor.eval()
                agent.bank.critic.eval()

        # 3) 创建独立评估环境（与训练环境完全分开）
        eval_env = self._build_env(name=f"PPO_eval_at_{steps}")

        survival_days = []
        terminated_count = 0
        truncated_count = 0
        # 每个主体一个 dict，里面存每个 episode 的收益（后续取均值/方差）
        per_agent = {}

        for target_name in self.e_execute:
            per_agent[target_name] = {
                "eval_business": [],
            }
        for target_name in self.b_execute:
            per_agent[target_name] = {
                "WNDB": [],  # 银行利润
            }

            # ========= 4) 开始评估回合 =========
        eval_noise = 0.02
        for ep in range(eval_episodes):
            eval_env.set_eval_noise(True, scale=eval_noise)
            state = eval_env.reset()
            done = False

            # 每个评估回合开始，也清空窗口，避免跨回合泄漏
            for agent in self.Agent.values():
                if hasattr(agent, "reset_window"):
                    agent.reset_window()

            while not done:
                action = {}

                # 企业动作
                for k in self.e_execute:
                    if deterministic:
                        act = self.Agent[k].choose_action_deterministic(state[k])
                    else:
                        act, _, _, _ = self.Agent[k].choose_action(state[k])
                    action[k] = act

                # 银行动作
                for k in self.b_execute:
                    if deterministic:
                        act = self.Agent[k].choose_action_deterministic(state[k])
                    else:
                        act, _, _, _ = self.Agent[k].choose_action(state[k])
                    action[k] = act

                eval_env.step(action)
                next_state, reward, done, info = eval_env.observe()
                state = next_state

            # ========= 5) 回合结束统计 =========
            is_terminated = bool(info.get("terminated", False))  # 破产（自然终止）
            is_truncated = bool(info.get("truncated", False))  # 到达 lim_day 截断

            # 推荐互斥归因：破产优先；否则才算截断
            if is_terminated:
                terminated_count += 1
            elif is_truncated:
                truncated_count += 1

            # 存活天数（回合级）
            survival_days.append(eval_env.day)

            # 分企业收益：直接读 episode 末的 total_reward
            for target_name in self.e_execute:
                total_reward = eval_env.Enterprise[
                    target_name].total_reward  # dict: {'eval_business':..., 'business':...}
                per_agent[target_name]["eval_business"].append(100 * total_reward["eval_business"])

            # 分银行收益：优先 WNDB，否则 sum(values)
            for target_name in self.b_execute:
                trb = eval_env.Bank[target_name].total_reward["WNDB"] * 100
                per_agent[target_name]["WNDB"].append(trb)

            # ========= 6) 汇总统计 =========
        survival = np.array(survival_days, dtype=np.float32)

        result = {
            "steps": int(steps),
            # "eval_episodes": int(eval_episodes),
            # "deterministic": bool(deterministic),

            "avg_survival_days": float(survival.mean()),

            "terminated_count": int(terminated_count),
            "truncated_count": int(truncated_count),
            "bankruptcy_rate": float(terminated_count / max(1, eval_episodes)),
            "truncated_rate": float(truncated_count / max(1, eval_episodes)),

            "agents": {}
        }

        # 企业分别汇总
        for target_name in self.e_execute:
            eb = np.array(per_agent[target_name]["eval_business"], dtype=np.float32)
            result["agents"][target_name] = {
                "avg_total_eval_business": float(eb.mean()),
            }

        # 银行分别汇总
        for target_name in self.b_execute:
            bt = np.array(per_agent[target_name]["WNDB"], dtype=np.float32)
            result["agents"][target_name] = {
                "avg_total_reward": float(bt.mean()),
            }

        # ========= 7)swanlab 记录（按主体分别打点） =========
        # 全局指标
        swanlab_payload = {
            "eval/avg_survival_days": result["avg_survival_days"],
            "eval/bankruptcy_rate": result["bankruptcy_rate"],
        }

        # 分企业
        for target_name in self.e_execute:
            wandb_payload[f"eval/{target_name}/avg_total_eval_business"] = result["agents"][target_name][
                "avg_total_eval_business"]

        # 分银行
        for target_name in self.b_execute:
            swanlab_payload[f"eval/{target_name}/avg_total_reward"] = result["agents"][target_name]["avg_total_reward"]

        # 用训练步数对齐横轴
        swanlab.log(swanlab_payload, step=int(steps / 10000))

        # ========= 9) 恢复训练窗口（关键：继续未完成训练回合） =========
        self._restore_agent_windows(window_snap)
        # 评估后：切回 train（继续训练必须做）
        for agent in self.Agent.values():
            if hasattr(agent, "enterprise"):
                agent.enterprise.actor.train()
                agent.enterprise.critic.train()
            if hasattr(agent, "bank"):
                agent.bank.actor.train()
                agent.bank.critic.train()
        return result

    def run(self, seed=None):
        config = Config_PPO(scope='', state_dim=0, action_dim=0, hidden_dim=0)
        swanlab.init(project="CL_learn", workspace="wx829", config={
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
            "lim-day": lim_day,
            "gamma":config.GAMMA,
            "lambda":config.LAMDA,
            #transformer参数
            "trans_seq_len":config.seq_len,
            "trans_n_heads":config.n_heads,
            "trans_n_layers":config.n_layers
        })
        # 1. PPO 超参数
        update_timestep = config.UPDATE_TIMESTEP
        # max_training_timesteps = config.MAX_TRAINING_STEPS
        total_step = config.total_step
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

        # self.load_actor_only()
        # 3. 开始训练循环
        state = self.env.reset()
        time_step = 0
        update_num = 0
        episode_num = 1

        while time_step < total_step:

            # --- 数据收集阶段 ---
            for _ in range(update_timestep):
                time_step += 1
                if time_step % self.eval_interval_steps == 0:
                    print("start to evaluate")
                    if self.eval_deterministic:
                        self.evaluate_current_policy(steps=time_step, eval_episodes=50, deterministic=True)
                    else:
                        self.evaluate_current_policy(steps=time_step, eval_episodes=50, deterministic=False)
                action, log_prob, mus, sigmas = {}, {}, {}, {}

                for target_key in self.e_execute:
                    act, lp, mu, sigma = self.Agent[target_key].choose_action(state[target_key])
                    action[target_key], log_prob[target_key], mus[target_key], sigmas[target_key] = act, lp, mu, sigma
                for target_key in self.b_execute:
                    act, lp, mu, sigma = self.Agent[target_key].choose_action(state[target_key])
                    action[target_key], log_prob[target_key], mus[target_key], sigmas[target_key] = act, lp, mu, sigma

                self.env.step(action)
                next_state, reward, done_env, info = self.env.observe()

                # NEW: 为每个 agent 计算 next_value = V(next_state)
                next_v = {}
                for k in self.e_execute:
                    next_v[k] = self.Agent[k].get_value(next_state[k])
                for k in self.b_execute:
                    next_v[k] = self.Agent[k].get_value(next_state[k])

                # NEW: 双掩码
                is_terminated = bool(info.get('terminated', done_env))  # 自然终止才截断 bootstrap
                nonterminal = 0 if done_env else 1  # 结束(terminated 或 truncated)则断开 GAE 递推

                # CHANGED: 存 transition（含 next_value 与 nonterminal）
                for target_key in self.e_execute:
                    self.Agent[target_key].store_transition(
                        state[target_key],
                        mus[target_key],
                        sigmas[target_key],
                        action[target_key],
                        log_prob[target_key],
                        reward[target_key]['business'],
                        is_terminated,
                        next_v[target_key],
                        nonterminal,
                    )
                for target_key in self.b_execute:
                    self.Agent[target_key].store_transition(
                        state[target_key],
                        mus[target_key],
                        sigmas[target_key],
                        action[target_key],
                        log_prob[target_key],
                        reward[target_key]['WNDB'],
                        is_terminated,
                        next_v[target_key],
                        nonterminal,
                    )

                state = next_state

                if done_env:
                    print(f"Episode {episode_num} finished. Total timesteps: {time_step}")
                    #1. 环境重置
                    state = self.env.reset()
                    # 2. 【关键】重置所有智能体的历史窗口
                    for target_key in self.e_execute:
                        self.Agent[target_key].reset_window()
                    for target_key in self.b_execute:
                        self.Agent[target_key].reset_window()
                    episode_num += 1

            # --- 学习阶段 ---
            print(f"--- Timestep {time_step}. Updating policies... ---")

            for agent_key, agent in self.Agent.items():
                agent.learn(state[agent_key])
                agent.clear_memory()

            if use_wandb and update_num % 5 == 0:
                critic_bank, actor_bank, avg_entropy_bank, clip_fraction_bank = self.Agent['bank1'].log()
                critic_production1, actor_production1, avg_entropy_production1, clip_fraction_production1 = self.Agent[
                    'production1'].log()
                critic_consumption1, actor_consumption1, avg_entropy_consumption1, clip_fraction_consumption1 = \
                    self.Agent['consumption1'].log()
                swanlab.log({'actor_loss/bank1': actor_bank})
                swanlab.log({'actor_loss/production1': actor_production1})
                swanlab.log({'actor_loss/consumption1': actor_consumption1})

                swanlab.log({'critic_loss/bank': critic_bank})
                swanlab.log({'critic_loss/production1': critic_production1})
                swanlab.log({'critic_loss/consumption1': critic_consumption1})

                swanlab.log({'avg_entropy/bank': avg_entropy_bank})
                swanlab.log({'avg_entropy/production1': avg_entropy_production1})
                swanlab.log({'avg_entropy/consumption1': avg_entropy_consumption1})

                swanlab.log({'clip_fraction/bank': clip_fraction_bank})
                swanlab.log({'clip_fraction/production1': clip_fraction_production1})
                swanlab.log({'clip_fraction/consumption1': clip_fraction_consumption1})

            print("--- Update finished. ---")
            update_num += 1
        swanlab.finish()
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

    def save_actors(self, save_dir="actors_only"):
        """
        保存所有智能体的 Actor 参数（仅用于评估）
        """
        save_dir = save_dir + "_" + "lim_day=" + str(lim_day) + "_seed=" + str(seed)
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

    def load_actor_only(self, save_dir="actors_only_lim_day=150_seed=451"):
        for target_key in self.e_execute:
            agent = self.Agent[target_key]
            path = os.path.join(save_dir, f"{agent.scope}_actor.pt")
            if os.path.exists(path):
                agent.enterprise.actor.load_state_dict(torch.load(path))
                agent.enterprise.actor.train()
                print(f"[🎯] 加载 actor: {agent.scope}")

        for target_key in self.b_execute:
            agent = self.Agent[target_key]
            path = os.path.join(save_dir, f"{agent.scope}_actor.pt")
            if os.path.exists(path):
                agent.bank.actor.load_state_dict(torch.load(path))
                agent.bank.actor.train()
                print(f"[🎯] 加载 actor: {agent.scope}")


if __name__ == '__main__':
    # for i in range(3):
    seeds_to_run = [936, 981, 891, 125, 777, 888, 999, 123]
    for seed in seeds_to_run:
        print(f"=== 启动 seed={seed} 的实验 ===")
        system = System()
        system.run(seed=seed)
        # system.save_actors()
        # system.evaluate_policy(episodes=500, deterministic=False, threshold=180)
        del system
        gc.collect()
        # 清空计算图
        torch.nn.Module.dump_patches = True
        torch.cuda.empty_cache()
        print(f"=== seed={seed} 实验结束 ===\n")

    # tf.reset_default_graph()
