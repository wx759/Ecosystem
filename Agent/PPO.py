import os
import random

import torch
import torch.nn.functional as F
from torch.distributions import Normal
from Agent import Config_PPO
import numpy as np

from Agent.RuningMeanStd import RunningMeanStd
import swanlab as wandb
from torch.optim.lr_scheduler import LinearLR


class PPO:
    """完整的PPO训练器，封装训练逻辑和算法"""

    def __init__(self, config: Config_PPO):
        # 环境初始化

        self.set_global_seed(config.random_seed)
        self.scope = config.scope
        self.state_dim = config.state_dim
        self.action_dim = config.action_dim
        self.hidden_dim = config.hidden_dim

        # 初始化 RunningMeanStd 实例
        self.rms = RunningMeanStd(shape=self.state_dim)
        self.rms_reward = RunningMeanStd(shape=())
        # 添加一个控制归一化开关的标志，方便你测试对比效果
        self.is_rms = config.is_rms_state  # 初始设置为 True，如果你想暂时关闭可以改为 False
        self.is_rms_reward = config.is_rms_reward

        # 缓冲区列表
        self.states = []
        self.actions = []
        self.logprobs = []
        self.rewards = []
        self.is_terminals = []
        self.next_values = []  # NEW: V(next_state)
        self.nonterminal_masks = []  # NEW: 1 未结束，0 done(terminated 或 truncated)
        # 超参数
        self.gamma = config.GAMMA
        self.lamda = config.LAMDA
        self.epochs = config.N_EPOCHS
        self.eps = config.CLIP_RANGE
        self.mini_batch_size = config.MINI_BATCH_SIZE

        self.entropy_start_enterprise = config.entropyRC_Enterprise
        self.entropy_start_bank = config.entropyRC_Bank
        self.entropyRC_Enterprise = config.entropyRC_Enterprise
        self.entropyRC_Bank = config.entropyRC_Bank
        self.learning_rate_actor_bank = config.LEARNING_RATE_AC_Bank
        self.learning_rate_actor_enterprise = config.LEARNING_RATE_AC_Enterprise
        self.learning_rate_critic_bank = config.LEARNING_RATE_C_Bank
        self.learning_rate_critic_enterprise = config.LEARNING_RATE_C_Enterprise
        self.critic_loss = 0
        self.actor_loss = 0
        self.avg_entropy = 0.0
        self.avg_clip_frac = 0.0

        # 设备（GPU/CPU）
        # self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.device = 'cpu'
        # 初始化网络和优化器
        self.actor = PolicyNet(self.state_dim, self.hidden_dim, self.action_dim).to(self.device)
        self.critic = ValueNet(self.state_dim, self.hidden_dim).to(self.device)
        if config.scope == 'bank1':
            self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=self.learning_rate_actor_bank)
            self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=self.learning_rate_critic_bank)
        else:
            self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=self.learning_rate_actor_enterprise)
            self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=self.learning_rate_critic_enterprise)
        self.actor_scheduler = LinearLR(
            self.actor_optimizer,
            start_factor=1.0,
            end_factor=0.3,
            total_iters=config.total_update)

    def set_global_seed(self, seed):
        # pytorch_seed
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
        # python_seed
        random.seed(seed)

        # NumPy_seed
        np.random.seed(seed)

        # 设置python的hash随机种子
        os.environ['PYTHONHASHSEED'] = str(seed)
        torch.backends.cudnn.deterministic = True
        # 设置PyTorch使用的算法为确定性算法
        torch.backends.cudnn.benchmark = False

    def choose_action(self, state):

        state = torch.tensor(state, dtype=torch.float).unsqueeze(0).to(self.device)

        with torch.no_grad():
            mu, sigma = self.actor(state)
            dist = Normal(mu, sigma)
            raw_action = dist.sample()
            action = torch.tanh(raw_action) * 0.5
            log_prob = dist.log_prob(raw_action).sum(dim=-1)

            # 加入 Jacobian 修正项
            log_det_jacobian = 2 * (torch.log(torch.tensor(2.0)) - raw_action - F.softplus(-2 * raw_action))
            log_det_jacobian = log_det_jacobian.sum(dim=-1)

            correction = self.action_dim * torch.log(torch.tensor(0.5))
            log_prob = log_prob - log_det_jacobian - correction

        return action.cpu().numpy().flatten(), log_prob.cpu().item()

    def store_transition(self, state, action, logprob, reward, is_terminal, next_value, nonterminal):  # CHANGED
        self.states.append(state)
        self.actions.append(action)
        self.logprobs.append(logprob)
        self.rewards.append(reward)
        self.is_terminals.append(is_terminal)  # 仅自然终止
        self.next_values.append(next_value)  # V(next_state)
        self.nonterminal_masks.append(nonterminal)  # 递推是否继续

    def clear_memory(self):
        del self.states[:]
        del self.actions[:]
        del self.logprobs[:]
        del self.rewards[:]
        del self.is_terminals[:]
        del self.next_values[:]  # NEW
        del self.nonterminal_masks[:]  # NEW

    def get_value(self, state):
        # 1. 将单个状态转换为 (1, state_dim) 的 NumPy 数组
        state_np = np.array(state, dtype=np.float64)[np.newaxis, :]

        # 2. 对状态进行归一化（与 choose_action 和 learn 中的逻辑一致）
        if self.is_rms:
            state_np = self.rms.normalize(state_np)

        # 3. 转换为 PyTorch 张量并移动到设备
        state_tensor = torch.tensor(state_np, dtype=torch.float).to(self.device)

        with torch.no_grad():
            value = self.critic(state_tensor).item()  # Critic 接收归一化后的状态
        return value

    def learn(self, final_state, agent_type: str):
        # 1. 从缓冲区获取并处理状态数据
        raw_old_states_np = np.array(self.states, dtype=np.float32)
        if self.is_rms:
            normalized_old_states_np = self.rms.normalize(raw_old_states_np)
        else:
            normalized_old_states_np = raw_old_states_np
        # if self.is_rms_reward and agent_type != 'bank1':
        if self.is_rms_reward:
            rewards_np = np.array(self.rewards, dtype=np.float32)
            rewards_np = rewards_np / (rewards_np.std() + 1e-8)
            self.rewards = rewards_np.tolist()

        old_states_tensor = torch.tensor(normalized_old_states_np, dtype=torch.float).to(self.device)
        old_actions = torch.tensor(np.array(self.actions), dtype=torch.float).to(self.device)
        old_logprobs = torch.tensor(self.logprobs, dtype=torch.float).to(self.device)

        # GAE 计算
        state_values = self.critic(old_states_tensor).squeeze().detach()
        values = state_values.tolist()

        next_values = np.array(self.next_values, dtype=np.float32)
        is_terminals_np = np.array(self.is_terminals, dtype=np.float32)
        nonterminal_masks_np = np.array(self.nonterminal_masks, dtype=np.float32)

        advantages = []
        gae = 0.0
        for t in reversed(range(len(self.rewards))):
            # 自然终止禁用 bootstrap；截断允许 bootstrap
            bootstrap_next_v = (1.0 - is_terminals_np[t]) * float(next_values[t])
            delta = self.rewards[t] + self.gamma * bootstrap_next_v - values[t]
            # done（terminated 或 truncated）均停止递推
            gae = delta + self.gamma * self.lamda * nonterminal_masks_np[t] * gae
            advantages.insert(0, gae)

        advantages = torch.tensor(advantages, dtype=torch.float).to(self.device)

        returns = (advantages + state_values).detach()

        # advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # Mini-Batch 训练阶段
        batch_size_total = len(self.states)
        for epoch in range(self.epochs):
            # 创建列表以记录每个mini-batch的损失
            actor_loss_list = []
            critic_loss_list = []
            entropy_list, clip_frac_list = [], []

            shuffled_indices = torch.randperm(batch_size_total)
            for start_idx in range(0, batch_size_total, self.mini_batch_size):
                end_idx = min(start_idx + self.mini_batch_size, batch_size_total)
                batch_indices = shuffled_indices[start_idx:end_idx]
                mini_batch_old_states = old_states_tensor[batch_indices]
                mini_batch_old_actions = old_actions[batch_indices]
                mini_batch_old_logprobs = old_logprobs[batch_indices]
                mini_batch_advantages = advantages[batch_indices]
                mini_batch_returns = returns[batch_indices]

                mu, std = self.actor(mini_batch_old_states)

                logprobs = self.compute_log_prob(mu, std, mini_batch_old_actions)
                ratios = torch.exp(logprobs - mini_batch_old_logprobs.detach())
                clip_frac = ((ratios > 1 + self.eps) | (ratios < 1 - self.eps)).float().mean().item()
                clip_frac_list.append(clip_frac)

                surr1 = ratios * mini_batch_advantages
                surr2 = torch.clamp(ratios, 1 - self.eps, 1 + self.eps) * mini_batch_advantages
                actor_loss_per_batch = -torch.min(surr1, surr2).mean()

                dist = torch.distributions.Normal(mu, std)
                entropy = dist.entropy().mean().item()
                entropy_list.append(entropy)

                if agent_type == 'bank1':
                    entropy_loss = -self.entropyRC_Bank * dist.entropy().mean()
                else:
                    entropy_loss = -self.entropyRC_Enterprise * dist.entropy().mean()

                actor_loss_with_entropy = actor_loss_per_batch + entropy_loss

                current_values = self.critic(mini_batch_old_states).view(-1)
                critic_loss_per_batch = F.mse_loss(current_values, mini_batch_returns.view(-1))

                # 记录每个mini-batch的损失
                actor_loss_list.append(actor_loss_per_batch.item())
                critic_loss_list.append(critic_loss_per_batch.item())

                # 在每次 Mini-Batch 的训练开始前，将优化器中所有参数的梯度清零。
                self.actor_optimizer.zero_grad()
                self.critic_optimizer.zero_grad()

                # 反向传播 计算损失对各自网络参数的梯度。并将其存储在grad属性
                actor_loss_with_entropy.backward()
                critic_loss_per_batch.backward()

                # 梯度裁剪 防止梯度爆炸
                torch.nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=0.5)
                torch.nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=0.5)

                # 读取.grad存储的梯度，更新参数
                self.actor_optimizer.step()
                self.critic_optimizer.step()


            # 【修改后】保存所有mini-batch损失的平均值，用于日志记录
            self.actor_loss = np.mean(actor_loss_list)
            self.critic_loss = np.mean(critic_loss_list)

            self.avg_entropy = np.mean(entropy_list)
            self.avg_clip_frac = np.mean(clip_frac_list)
            # 👇 ===== KL早停机制 ===== 👇
          #  with torch.no_grad():
                # 用更新后的网络重新计算整个batch的log_prob
               # mu_new, std_new = self.actor(old_states_tensor)
             #   new_logprobs = self.compute_log_prob(mu_new, std_new, old_actions)

                # 计算KL散度（近似）
                # D_KL(π_old || π_new) ≈ E[log π_old - log π_new]
               # approx_kl = (old_logprobs - new_logprobs).mean().item()

                # 如果KL散度超过阈值，提前停止
                #if approx_kl > 1:  # 阈值可调
                   # print(f"[{agent_type}] Epoch {epoch + 1}/{self.epochs}: "
                       #   f"KL={approx_kl:.4f} > 0.1, 早停")
                  #  break
        if self.actor_scheduler.last_epoch < self.actor_scheduler.total_iters:
            self.actor_scheduler.step()

        # === 模拟退火式 熵衰减 ===
        progress = self.actor_scheduler.last_epoch / self.actor_scheduler.total_iters
        decay_factor = max(0.0, 1.0 - progress)
        self.entropyRC_Enterprise = self.entropy_start_enterprise * (0.32 + 0.68 * decay_factor)
        self.entropyRC_Bank = self.entropy_start_bank * (0.32 + 0.68 * decay_factor)
        # 【新增】诊断
        self.diagnose(old_states_tensor, old_actions, old_logprobs, agent_type)

    def get_loss(self):
        return self.critic_loss, self.actor_loss

    def get_test_indicator(self):
        return self.avg_entropy, self.avg_clip_frac

    @staticmethod
    def compute_log_prob(mu, std, action):
        dist = Normal(mu, std)
        # 动作被限制在 [-0.5, 0.5] 之间, 逆变换是 * 2
        scaled_action = torch.clamp(action * 2.0, -1.0 + 1e-6, 1.0 - 1e-6)
        raw_action = torch.atanh(scaled_action)

        log_prob_raw = dist.log_prob(raw_action)
        log_det_jacobian = 2 * (torch.log(torch.tensor(2.0)) - raw_action - F.softplus(-2 * raw_action))

        # 【修改后】与 choose_action 中一样，补上对数缩放因子 log(0.5)。
        # 这里的张量运算是元素级别的，所以直接减去标量即可，PyTorch会自动广播。
        log_prob = log_prob_raw - log_det_jacobian - torch.log(torch.tensor(0.5))

        return log_prob.sum(-1)

    def diagnose(self, old_states_tensor, old_actions, old_logprobs, agent_type):
        """诊断 PPO 训练过程中的关键指标"""
        with torch.no_grad():
            # 1. Actor 分布参数
            mu, std = self.actor(old_states_tensor)
            #训练后的策略，对训练数据（旧状态）重新预测时的标准差均值
            log_std_mean = torch.log(std).mean().item()
            v_pred = self.critic(old_states_tensor).squeeze(-1)
            #训练后的价值网络，对训练数据（旧状态）重新估值时的平均值和标准差
            v_mean =v_pred.mean().item()
            v_std = v_pred.std().item()

            # 3. KL divergence（新旧策略分布差异）
            new_logprobs = self.compute_log_prob(mu, std, old_actions)
            kl = (old_logprobs - new_logprobs).mean().item()
            if agent_type == 'production1':
                wandb.log({
                    "log_std_mean/production1": log_std_mean,
                    "KL_divergence/production1": kl,
                    "critic_mean/production1": v_mean,
                    "critic_std/production1": v_std,
                })
            elif agent_type == 'consumption1':
                wandb.log({
                    "log_std_mean/consumption1": log_std_mean,
                    "KL_divergence/consumption1": kl,
                    "critic_mean/consumption1": v_mean,
                    "critic_std/consumption1": v_std,
                })
            else:
                wandb.log({
                    "log_std_mean/bank": log_std_mean,
                    "KL_divergence/bank": kl,
                    "critic_mean/bank": v_mean,
                    "critic_std/bank": v_std,
                })

    def choose_action_deterministic(self, state):
        state = torch.tensor(state,dtype=torch.float).unsqueeze(0).to(self.device)

        with torch.no_grad():
            mu, sigma = self.actor(state)
            raw_action = mu
            # 约束动作范围（与训练时一致）
            action = torch.tanh(raw_action) * 0.5

        return action.cpu().numpy().flatten()


class PolicyNet(torch.nn.Module):
    def __init__(self, state_dim, hidden_dim, action_dim):
        super().__init__()
        self.fc1 = torch.nn.Linear(state_dim, hidden_dim)

        self.fc_mu = torch.nn.Linear(hidden_dim, action_dim)
        self.fc_std = torch.nn.Linear(hidden_dim, action_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        raw_mu = self.fc_mu(x)
        raw_std = torch.clamp(self.fc_std(x), -20, 2)
        std = F.softplus(raw_std) + 0.01  # 加一个小的偏置，防止 sigma 变为 0
        return raw_mu, std


class ValueNet(torch.nn.Module):
    """价值网络（Critic）"""

    def __init__(self, state_dim, hidden_dim):
        super().__init__()
        self.fc1 = torch.nn.Linear(state_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        return self.fc2(x)
