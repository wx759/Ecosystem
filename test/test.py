import argparse
import os.path
import gym
from gym.wrappers import TimeLimit
from gym.spaces import Discrete, Box, MultiBinary, MultiDiscrete, Space
import numpy as np
import random
import math
import time
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions.categorical import Categorical
from torch.utils.tensorboard import SummaryWriter
import pandas as pd
import wandb
from robot import Robot
from typing import NamedTuple, Tuple
# from gymnasium import spaces
from copy import deepcopy
from stable_baselines3.common.utils import zip_strict
from functools import partial


# taken from https://github.com/openai/baselines/blob/master/baselines/common/vec_env/vec_normalize.py
class RunningMeanStd(object):
    def __init__(self, epsilon=1e-4, shape=()):
        self.mean = np.zeros(shape, 'float64')
        self.var = np.ones(shape, 'float64')
        self.count = epsilon

    def update(self, x):
        batch_mean = np.mean([x], axis=0)
        batch_var = np.var([x], axis=0)
        batch_count = 1
        self.update_from_moments(batch_mean, batch_var, batch_count)

    def update_from_moments(self, batch_mean, batch_var, batch_count):
        self.mean, self.var, self.count = update_mean_var_count_from_moments(
            self.mean, self.var, self.count, batch_mean, batch_var, batch_count)

def update_mean_var_count_from_moments(mean, var, count, batch_mean, batch_var, batch_count):
    delta = batch_mean - mean
    tot_count = count + batch_count

    new_mean = mean + delta * batch_count / tot_count
    m_a = var * count
    m_b = batch_var * batch_count
    M2 = m_a + m_b + np.square(delta) * count * batch_count / tot_count
    new_var = M2 / tot_count
    new_count = tot_count

    return new_mean, new_var, new_count

# ALGO LOGIC: initialize agent here:
class CategoricalMasked(Categorical):
    def __init__(self, probs=None, logits=None, validate_args=None, masks=[]):
        self.masks = masks
        if len(self.masks) == 0:
            super(CategoricalMasked, self).__init__(probs, logits, validate_args)
        else:
            self.masks = masks.type(torch.BoolTensor).to(device)
            logits = torch.where(self.masks, logits, torch.tensor(-1e+8).to(device))
            super(CategoricalMasked, self).__init__(probs, logits, validate_args)

    def entropy(self):
        if len(self.masks) == 0:
            return super(CategoricalMasked, self).entropy()
        p_log_p = self.logits * self.probs
        p_log_p = torch.where(self.masks, p_log_p, torch.tensor(0.).to(device))
        return -p_log_p.sum(-1)

class Policy(nn.Module):
    def __init__(self):
        super(Policy, self).__init__()
        # self.features = nn.Sequential(
        #     nn.Conv2d(27, 16, kernel_size=2, ),
        #     nn.MaxPool2d(1),
        #     nn.ReLU())
        self.lstm = torch.nn.LSTM(input_size=8, hidden_size=128)
        self.fc = nn.Sequential(
            # nn.LayerNorm(8),
            # nn.Linear(8, 128),
            # nn.LayerNorm(128),
            # nn.LeakyReLU(),
            nn.Linear(128, 256),
            nn.LayerNorm(256),
            nn.LeakyReLU(),
            nn.Linear(256, 512),
            nn.LayerNorm(512),
            nn.LeakyReLU(),
            nn.Linear(512, 1024),
            nn.LeakyReLU(),
            nn.Linear(1024, 2048),
            nn.LeakyReLU(),
            nn.Linear(2048, 11 * 11 * 11)
        )

    def forward(self, x):
        # x = torch.Tensor(x).to(device)
        x = self.fc(x)
        return x

    def process_sequence(self, obs, lstm_states, episode_starts):
        # LSTM logic
        # (sequence length, batch size, features dim)
        # (batch size = n_envs for data collection or n_seq when doing gradient update)
        n_seq = lstm_states[0].shape[1]
        # Batch to sequence
        # (padded batch size, features_dim) -> (n_seq, max length, features_dim) -> (max length, n_seq, features_dim)
        # note: max length (max sequence length) is always 1 during data collection
        obs_sequence = obs.reshape((n_seq, -1, 8)).swapaxes(0, 1)
        episode_starts = episode_starts.reshape((n_seq, -1)).swapaxes(0, 1)

        # If we don't have to reset the state in the middle of a sequence
        # we can avoid the for loop, which speeds up things
        if torch.all(episode_starts == 0.0):
            lstm_output, lstm_states = self.lstm(obs_sequence, lstm_states)
            lstm_output = torch.flatten(lstm_output.transpose(0, 1), start_dim=0, end_dim=1)
            return lstm_output, lstm_states

        lstm_output = []
        # Iterate over the sequence
        for features, episode_start in zip_strict(obs_sequence, episode_starts):
            hidden, lstm_states = self.lstm(
                features.unsqueeze(dim=0),
                (
                    # Reset the states at the beginning of a new episode
                    (1.0 - episode_start).view(1, n_seq, 1) * lstm_states[0],
                    (1.0 - episode_start).view(1, n_seq, 1) * lstm_states[1],
                ),
            )
            lstm_output += [hidden]
        # Sequence to batch
        # (sequence length, n_seq, lstm_out_dim) -> (batch_size, lstm_out_dim)
        lstm_output = torch.flatten(torch.cat(lstm_output).transpose(0, 1), start_dim=0, end_dim=1)
        return lstm_output, lstm_states

    def get_action(self, x, action=None, invalid_action_masks=None, lstm_states=None, episode_starts=None):
        x = torch.Tensor(x).to(device)
        latent_pi, lstm_states_pi = self.process_sequence(x, lstm_states, episode_starts)

        logits = self.forward(latent_pi)
        # split_logits = torch.split(logits, env.action_space.nvec.tolist(), dim=1)

        if invalid_action_masks is not None:
            invalid_action_masks = invalid_action_masks.reshape(invalid_action_masks.shape[0], -1)
            # split_invalid_action_masks = torch.split(invalid_action_masks, env.action_space.nvec.tolist(), dim=1)
            multi_categoricals = [CategoricalMasked(logits=logits, masks=iam) for (logits, iam) in
                                  zip(logits, invalid_action_masks)]
        else:
            multi_categoricals = [Categorical(logits=logits) for logits in logits]

        entropy = np.zeros(len(multi_categoricals))
        if action is None:
            action = torch.stack([categorical.sample() for categorical in multi_categoricals])
        else:
            for i in range(len(multi_categoricals)):
                multi_categoricals[i].masks = invalid_action_masks[i].type(torch.BoolTensor).to(device)
                entropy[i] = multi_categoricals[i].entropy()
        action = torch.Tensor(action).to(device)
        logprob = torch.stack([categorical.log_prob(a) for a, categorical in zip(action, multi_categoricals)])
        return action, logprob, [torch.from_numpy(entropy)], multi_categoricals, lstm_states_pi

class Value(nn.Module):
    def __init__(self):
        super(Value, self).__init__()
        # self.features = nn.Sequential(
        #     nn.Conv2d(27, 16, kernel_size=2, ),
        #     nn.MaxPool2d(1),
        #     nn.ReLU())
        self.lstm = torch.nn.LSTM(input_size=8, hidden_size=128)
        self.fc = nn.Sequential(
            # nn.LayerNorm(8),
            # nn.Linear(3 * 2 + 2, 128),
            # nn.LayerNorm(128),
            # nn.LeakyReLU(),
            nn.Linear(128, 256),
            nn.LayerNorm(256),
            nn.LeakyReLU(),
            nn.Linear(256, 512),
            nn.LayerNorm(512),
            nn.LeakyReLU(),
            nn.Linear(512, 1024),
            nn.LeakyReLU(),
            nn.Linear(1024, 2048),
            nn.LeakyReLU(),
            nn.Linear(2048, 1)
        )

    def forward(self, x):
        # x = torch.Tensor(x).to(device)
        x = self.fc(x)
        return x

    def get_value(self, x, lstm_states, episode_starts):
        x = torch.Tensor(x).to(device)
        latent_vf, lstm_states_vf = self.process_sequence(x, lstm_states, episode_starts)
        value = self.forward(latent_vf)
        return value, lstm_states_vf

    def process_sequence(self, obs, lstm_states, episode_starts):
        # LSTM logic
        # (sequence length, batch size, features dim)
        # (batch size = n_envs for data collection or n_seq when doing gradient update)
        n_seq = lstm_states[0].shape[1]
        # Batch to sequence
        # (padded batch size, features_dim) -> (n_seq, max length, features_dim) -> (max length, n_seq, features_dim)
        # note: max length (max sequence length) is always 1 during data collection
        obs_sequence = obs.reshape((n_seq, -1, 8)).swapaxes(0, 1)
        episode_starts = episode_starts.reshape((n_seq, -1)).swapaxes(0, 1)

        # If we don't have to reset the state in the middle of a sequence
        # we can avoid the for loop, which speeds up things
        if torch.all(episode_starts == 0.0):
            lstm_output, lstm_states = self.lstm(obs_sequence, lstm_states)
            lstm_output = torch.flatten(lstm_output.transpose(0, 1), start_dim=0, end_dim=1)
            return lstm_output, lstm_states

        lstm_output = []
        # Iterate over the sequence
        for features, episode_start in zip_strict(obs_sequence, episode_starts):
            hidden, lstm_states = self.lstm(
                features.float().unsqueeze(dim=0),
                (
                    # Reset the states at the beginning of a new episode
                    (1.0 - episode_start).view(1, n_seq, 1) * lstm_states[0],
                    (1.0 - episode_start).view(1, n_seq, 1) * lstm_states[1],
                ),
            )
            lstm_output += [hidden]
        # Sequence to batch
        # (sequence length, n_seq, lstm_out_dim) -> (batch_size, lstm_out_dim)
        lstm_output = torch.flatten(torch.cat(lstm_output).transpose(0, 1), start_dim=0, end_dim=1)
        return lstm_output, lstm_states

class Agent():
    def __init__(self):
        self.pg = Policy().to(device)
        # self.pg.fc[0].register_forward_hook(get_activation('p_ln_8'))
        # self.pg.fc[1].register_forward_hook(get_activation('p_linear_8-128'))
        # self.pg.fc[2].register_forward_hook(get_activation('p_ln_128'))
        # self.pg.fc[3].register_forward_hook(get_activation('p_leakyreLU_128'))
        # self.pg.fc[4].register_forward_hook(get_activation('p_linear_128-256'))
        # self.pg.fc[5].register_forward_hook(get_activation('p_ln_256'))
        # self.pg.fc[6].register_forward_hook(get_activation('p_leakyreLU_256'))
        # self.pg.fc[7].register_forward_hook(get_activation('p_linear_256-512'))
        # self.pg.fc[8].register_forward_hook(get_activation('p_ln_512'))
        # self.pg.fc[9].register_forward_hook(get_activation('p_leakyreLU_512'))
        # self.pg.fc[10].register_forward_hook(get_activation('p_linear_512-1024'))
        # self.pg.fc[11].register_forward_hook(get_activation('p_leakyreLU_1024'))
        # self.pg.fc[12].register_forward_hook(get_activation('p_linear_1024-2048'))
        # self.pg.fc[13].register_forward_hook(get_activation('p_leakyreLU_2048'))
        # self.pg.fc[14].register_forward_hook(get_activation('p_linear_2048-1331'))
        self.vf = Value().to(device)
        # self.vf.fc[0].register_forward_hook(get_activation('v_ln_8'))
        # self.vf.fc[1].register_forward_hook(get_activation('v_linear_8-128'))
        # self.vf.fc[2].register_forward_hook(get_activation('v_ln_128'))
        # self.vf.fc[3].register_forward_hook(get_activation('v_leakyreLU_128'))
        # self.vf.fc[4].register_forward_hook(get_activation('v_linear_128-256'))
        # self.vf.fc[5].register_forward_hook(get_activation('v_ln_256'))
        # self.vf.fc[6].register_forward_hook(get_activation('v_leakyreLU_256'))
        # self.vf.fc[7].register_forward_hook(get_activation('v_linear_256-512'))
        # self.vf.fc[8].register_forward_hook(get_activation('v_ln_512'))
        # self.vf.fc[9].register_forward_hook(get_activation('v_leakyreLU_512'))
        # self.vf.fc[10].register_forward_hook(get_activation('v_linear_512-1024'))
        # self.vf.fc[11].register_forward_hook(get_activation('v_leakyreLU_1024'))
        # self.vf.fc[12].register_forward_hook(get_activation('v_linear_1024-2048'))
        # self.vf.fc[13].register_forward_hook(get_activation('v_leakyreLU_2048'))
        # self.vf.fc[14].register_forward_hook(get_activation('v_linear_2048-1'))
        self.pg_optimizer = None
        self.v_optimizer = None
        self.pg_lr_scheduler = None
        self.v_lr_scheduler = None
        self.loss_fn = nn.MSELoss()
        self.obs = None
        self.action = None
        self.actions = None
        self.probs = None
        self.logprobs = None
        self.logproba = None
        self.rewards = None
        self.episode_rewards = None
        self.invalid_action_mask = None
        self.invalid_action_masks = None
        self.values = None
        self.last_value = 0
        self.bootstrapped_rewards = None
        self.bootstrapped_values = None
        self.deltas = None
        self.advantages = None
        self.returns = None
        self.entropy = None
        self.entropys = None
        self.target_pg = Policy().to(device)
        self.inds = None
        self.minibatch_ind = None
        self.newlogproba = None
        self.ratio = None
        self.clip_adv = None
        self.policy_loss = None
        self.approx_kl = None
        self.new_values = None
        self.v_loss_unclipped = None
        self.v_clipped = None
        self.v_loss_clipped = None
        self.v_loss_max = None
        self.v_loss = None
        # LSTM
        self.last_lstm_states = None
        self.lstm_states = None
        self.hidden_states_pi = None
        self.cell_states_pi = None
        self.hidden_states_vf = None
        self.cell_states_vf = None
        # Buffer


def discount_cumsum(x, dones, gamma):
    """
    computing discounted cumulative sums of vectors that resets with dones
    input:
        vector x,  vector dones,
        [x0,       [0,
         x1,        0,
         x2         1,
         x3         0,
         x4]        0]
    output:
        [x0 + discount * x1 + discount^2 * x2,
         x1 + discount * x2,
         x2,
         x3 + discount * x4,
         x4]
    """
    discount_cumsum = np.zeros_like(x)
    discount_cumsum[-1] = x[-1]
    for t in reversed(range(x.shape[0] - 1)):
        discount_cumsum[t] = x[t] + gamma * discount_cumsum[t + 1] * (1 - dones[t])
    return discount_cumsum

class NormalizedEnv(gym.core.Wrapper):
    def __init__(self, env, ob=True, ret=True, clipob=10., cliprew=10., gamma=0.99, epsilon=1e-8):
        super(NormalizedEnv, self).__init__(env)
        self.ob_rms = RunningMeanStd(shape=len(self.observation_space)) if ob else None
        self.ret_rms = RunningMeanStd(shape=(1,)) if ret else None
        self.clipob = clipob
        self.cliprew = cliprew
        self.ret = np.zeros(())
        self.gamma = gamma
        self.epsilon = epsilon

    def step(self, action):
        obs, rews, news, infos = self.env.step(action)
        infos['episode_reward'] = rews
        # print("before", self.ret)
        self.ret = self.ret * self.gamma + rews
        # print("after", self.ret)
        obs = self._obfilt(obs)
        if self.ret_rms:
            self.ret_rms.update(np.array([self.ret].copy()))
            rews = np.clip(rews / np.sqrt(self.ret_rms.var + self.epsilon), -self.cliprew, self.cliprew)
        self.ret = self.ret * (1-float(news))
        return obs, rews, news, infos

    def _obfilt(self, obs):
        if self.ob_rms:
            self.ob_rms.update(obs)
            obs = np.clip((obs - self.ob_rms.mean) / np.sqrt(self.ob_rms.var + self.epsilon), -self.clipob, self.clipob)
            return obs
        else:
            return obs

    def reset(self):
        self.ret = np.zeros(())
        obs = self.env.reset()
        return self._obfilt(obs)

class RNNStates(NamedTuple):
    pi: Tuple[torch.Tensor, ...]
    vf: Tuple[torch.Tensor, ...]

class RecurrentRolloutBufferSamples(NamedTuple):
    observations: np.array
    actions: np.array
    invalid_action_masks: np.array
    old_values: torch.Tensor
    old_log_prob: torch.Tensor
    advantages: torch.Tensor
    returns: torch.Tensor
    lstm_states: RNNStates
    episode_starts: np.array
    mask: torch.Tensor

class Buffer():
    def __init__(
            self,
            buffer_size: int,
            gamma: float,
            gae_lambda: float,
            observation_space: int,
            action_space: int,
            invalid_action_mask_spaces: Tuple,
            hidden_state_shape: Tuple[int, int, int, int],
            device):
        self.buffer_size = buffer_size
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.observation_spaces = observation_space
        self.action_spaces = action_space
        self.invalid_action_mask_spaces = invalid_action_mask_spaces
        self.hidden_state_shape = hidden_state_shape
        self.device = device
        self.seq_start_indices, self.seq_end_indices = None, None
        self.generator_ready = False
        self._last_obs = None
        self.actions = None
        self.rewards = None
        self._last_episode_starts = None
        self.values = None
        self.log_probs = None
        self.lstm_states = None
        self.reset()

    def reset(self):
        self.observations = np.zeros((self.buffer_size, 1, self.observation_spaces))
        self.actions = np.zeros((self.buffer_size, 1, self.action_spaces))
        self.invalid_action_masks = np.zeros((self.buffer_size,) + self.invalid_action_mask_spaces)
        self.rewards = np.zeros((self.buffer_size, 1), dtype=np.float32)
        self.returns = np.zeros((self.buffer_size, 1), dtype=np.float32)
        self.episode_starts = np.zeros((self.buffer_size, 1), dtype=np.float32)
        self.values = np.zeros((self.buffer_size, 1), dtype=np.float32)
        self.log_probs = np.zeros((self.buffer_size, 1), dtype=np.float32)
        self.advantages = np.zeros((self.buffer_size, 1), dtype=np.float32)
        self.generator_ready = False
        self.pos = 0
        self.full = False
        self.hidden_states_pi = np.zeros(self.hidden_state_shape, dtype=np.float32)
        self.cell_states_pi = np.zeros(self.hidden_state_shape, dtype=np.float32)
        self.hidden_states_vf = np.zeros(self.hidden_state_shape, dtype=np.float32)
        self.cell_states_vf = np.zeros(self.hidden_state_shape, dtype=np.float32)
        self.generator_ready = False

    def add(self, last_obs, action, invalid_action_mask, reward, last_episode_start, value, log_prob, lstm_states):
        self.hidden_states_pi[self.pos] = np.array(lstm_states.pi[0].cpu().numpy())
        self.cell_states_pi[self.pos] = np.array(lstm_states.pi[1].cpu().numpy())
        self.hidden_states_vf[self.pos] = np.array(lstm_states.vf[0].cpu().numpy())
        self.cell_states_vf[self.pos] = np.array(lstm_states.vf[1].cpu().numpy())

        if len(log_prob.shape) == 0:
            # Reshape 0-d tensor to avoid error
            log_prob = log_prob.reshape(-1, 1)

        # Reshape needed when using multiple envs with discrete observations
        # as numpy cannot broadcast (n_discrete,) to (n_discrete, 1)
        # if isinstance(self.observation_space, spaces.Discrete):
        #     obs = last_obs.reshape((self.n_envs, *self.obs_shape))

        # Reshape to handle multi-dim and discrete action spaces, see GH #970 #1392
        action = action.reshape((1, 1))

        self.observations[self.pos] = np.array(last_obs).copy()
        self.actions[self.pos] = np.array(action).copy()
        self.invalid_action_masks[self.pos] = np.array(invalid_action_mask).copy()
        self.rewards[self.pos] = np.array(reward).copy()
        self.episode_starts[self.pos] = np.array(last_episode_start.cpu()).copy()
        self.values[self.pos] = value.clone().cpu().numpy().flatten()
        self.log_probs[self.pos] = log_prob.clone().cpu().numpy()
        self.pos += 1
        if self.pos == self.buffer_size:
            self.full = True

    def get(self, minibatch_size):
        # Prepare the data
        if not self.generator_ready:
            # hidden_state_shape = (self.n_steps, lstm.num_layers, self.n_envs, lstm.hidden_size)
            # swap first to (self.n_steps, self.n_envs, lstm.num_layers, lstm.hidden_size)
            for tensor in ["hidden_states_pi", "cell_states_pi", "hidden_states_vf", "cell_states_vf"]:
                self.__dict__[tensor] = self.__dict__[tensor].swapaxes(1, 2)

            # flatten but keep the sequence order
            # 1. (n_steps, n_envs, *tensor_shape) -> (n_envs, n_steps, *tensor_shape)
            # 2. (n_envs, n_steps, *tensor_shape) -> (n_envs * n_steps, *tensor_shape)
            for tensor in [
                "observations",
                "actions",
                "values",
                "log_probs",
                "advantages",
                "returns",
                "hidden_states_pi",
                "cell_states_pi",
                "hidden_states_vf",
                "cell_states_vf",
                "episode_starts",
            ]:
                self.__dict__[tensor] = self.swap_and_flatten(self.__dict__[tensor])
            self.generator_ready = True

        # Sampling strategy that allows any mini batch size but requires
        # more complexity and use of padding
        # Trick to shuffle a bit: keep the sequence order
        # but split the indices in two
        # split_index = np.random.randint(self.buffer_size)
        split_index = 0
        indices = np.arange(self.buffer_size)
        indices = np.concatenate((indices[split_index:], indices[:split_index]))

        env_change = np.zeros(self.buffer_size).reshape(self.buffer_size, 1)
        # Flag first timestep as change of environment
        env_change[0, :] = 1.0
        env_change = self.swap_and_flatten(env_change)

        start_idx = 0
        while start_idx < self.buffer_size:
            batch_inds = indices[start_idx: start_idx + minibatch_size]
            yield self._get_samples(batch_inds, env_change)
            start_idx += minibatch_size

    def _get_samples(self, batch_inds, env_change):
        # Retrieve sequence starts and utility function
        self.seq_start_indices, self.pad, self.pad_and_flatten = self.create_sequencers(self.episode_starts[batch_inds],
                                                                                   env_change[batch_inds], self.device)
        # Number of sequences
        # n_seq = len(self.seq_start_indices)
        # max_length = self.pad(self.actions[batch_inds]).shape[1]
        n_seq = args.lstm_sequence
        max_length = args.lstm_step
        padded_batch_size = n_seq * max_length
        # We retrieve the lstm hidden states that will allow
        # to properly initialize the LSTM at the beginning of each sequence
        lstm_states_pi = (
            # 1. (n_envs * n_steps, n_layers, dim) -> (batch_size, n_layers, dim)
            # 2. (batch_size, n_layers, dim)  -> (n_seq, n_layers, dim)
            # 3. (n_seq, n_layers, dim) -> (n_layers, n_seq, dim)
            self.hidden_states_pi[batch_inds][self.seq_start_indices].swapaxes(0, 1),
            self.cell_states_pi[batch_inds][self.seq_start_indices].swapaxes(0, 1),
        )
        lstm_states_vf = (
            # (n_envs * n_steps, n_layers, dim) -> (n_layers, n_seq, dim)
            self.hidden_states_vf[batch_inds][self.seq_start_indices].swapaxes(0, 1),
            self.cell_states_vf[batch_inds][self.seq_start_indices].swapaxes(0, 1),
        )
        lstm_states_pi = (self.to_torch(lstm_states_pi[0]).contiguous(), self.to_torch(lstm_states_pi[1]).contiguous())
        lstm_states_vf = (self.to_torch(lstm_states_vf[0]).contiguous(), self.to_torch(lstm_states_vf[1]).contiguous())

        return RecurrentRolloutBufferSamples(
            # (batch_size, obs_dim) -> (n_seq, max_length, obs_dim) -> (n_seq * max_length, obs_dim)
            observations=self.pad(self.observations[batch_inds]).cpu().numpy().reshape(padded_batch_size, self.observation_spaces),
            actions=self.pad(self.actions[batch_inds]).cpu().numpy().reshape(padded_batch_size, self.action_spaces),
            invalid_action_masks=self.pad(self.invalid_action_masks[batch_inds]).cpu().reshape((padded_batch_size,) + self.invalid_action_mask_spaces),
            old_values=self.pad_and_flatten(self.values[batch_inds]),
            old_log_prob=self.pad_and_flatten(self.log_probs[batch_inds]),
            advantages=self.pad_and_flatten(self.advantages[batch_inds]),
            returns=self.pad_and_flatten(self.returns[batch_inds]),
            lstm_states=RNNStates(lstm_states_pi, lstm_states_vf),
            episode_starts=self.pad_and_flatten(self.episode_starts[batch_inds]),
            mask=self.pad_and_flatten(np.ones_like(self.returns[batch_inds])),
        )

    def create_sequencers(self, episode_starts, env_change, device):
        """
        Create the utility function to chunk data into
        sequences and pad them to create fixed size tensors.

        :param episode_starts: Indices where an episode starts
        :param env_change: Indices where the data collected
            come from a different env (when using multiple env for data collection)
        :param device: PyTorch device
        :return: Indices of the transitions that start a sequence,
            pad and pad_and_flatten utilities tailored for this batch
            (sequence starts and ends indices are fixed)
        """
        # Create sequence if env changes too
        seq_start = np.logical_or(episode_starts, env_change).flatten()
        # First index is always the beginning of a sequence
        seq_start[0] = True
        # Retrieve indices of sequence starts
        # seq_start_indices = np.where(seq_start == True)[0]  # noqa: E712
        seq_start_indices = np.random.choice(np.array(args.batch_size), size=args.lstm_sequence, replace=False)
        seq_start_indices = np.sort(seq_start_indices)
        # End of sequence are just before sequence starts
        # Last index is also always end of a sequence
        # seq_end_indices = np.concatenate([(seq_start_indices - 1)[1:], np.array([len(episode_starts)])])
        # seq_end_indices = np.minimum(seq_start_indices + args.lstm_step - 1, args.batch_size - 1)
        seq_end_indices = np.zeros_like(seq_start_indices)
        for i, start_index in enumerate(seq_start_indices):
            end_index_candidate = start_index + args.lstm_step - 1
            if end_index_candidate >= len(episode_starts):
                end_index_candidate = len(episode_starts) - 1
            if np.any(episode_starts[start_index + 1:end_index_candidate + 1]):
                true_indices = np.where(episode_starts[start_index + 1:end_index_candidate + 1])[0]
                seq_end_indices[i] = start_index + true_indices[0]
            else:
                seq_end_indices[i] = end_index_candidate

        # Create padding method for this minibatch
        # to avoid repeating arguments (seq_start_indices, seq_end_indices)
        local_pad = partial(pad, seq_start_indices, seq_end_indices, device)
        local_pad_and_flatten = partial(pad_and_flatten, seq_start_indices, seq_end_indices, device)
        return seq_start_indices, local_pad, local_pad_and_flatten

    def compute_returns_and_advantage(self, last_values, dones):
        """
        Post-processing step: compute the lambda-return (TD(lambda) estimate)
        and GAE(lambda) advantage.

        Uses Generalized Advantage Estimation (https://arxiv.org/abs/1506.02438)
        to compute the advantage. To obtain Monte-Carlo advantage estimate (A(s) = R - V(S))
        where R is the sum of discounted reward with value bootstrap
        (because we don't always have full episode), set ``gae_lambda=1.0`` during initialization.

        The TD(lambda) estimator has also two special cases:
        - TD(1) is Monte-Carlo estimate (sum of discounted rewards)
        - TD(0) is one-step estimate with bootstrapping (r_t + gamma * v(s_{t+1}))

        For more information, see discussion in https://github.com/DLR-RM/stable-baselines3/pull/375.

        :param last_values: state value estimation for the last step (one for each env)
        :param dones: if the last step was a terminal step (one bool for each env).
        """
        # Convert to numpy
        last_values = last_values.clone().cpu().numpy().flatten()

        last_gae_lam = 0
        for step in reversed(range(self.buffer_size)):
            if step == self.buffer_size - 1:
                next_non_terminal = 1.0 - dones
                next_values = last_values
            else:
                next_non_terminal = 1.0 - self.episode_starts[step + 1]
                next_values = self.values[step + 1]
            delta = self.rewards[step] + self.gamma * next_values * next_non_terminal - self.values[step]
            last_gae_lam = delta + self.gamma * self.gae_lambda * next_non_terminal * last_gae_lam
            self.advantages[step] = last_gae_lam
        # TD(lambda) estimator, see Github PR #375 or "Telescoping in TD(lambda)"
        # in David Silver Lecture 4: https://www.youtube.com/watch?v=PnHCvfgC_ZA
        self.returns = self.advantages + self.values

    def to_torch(self, array: np.ndarray, copy: bool = True) -> torch.Tensor:
        """
        Convert a numpy array to a PyTorch tensor.
        Note: it copies the data by default

        :param array:
        :param copy: Whether to copy or not the data (may be useful to avoid changing things
            by reference). This argument is inoperative if the device is not the CPU.
        :return:
        """
        if copy:
            return torch.tensor(array, device=self.device)
        return torch.as_tensor(array, device=self.device)

    @staticmethod
    def swap_and_flatten(arr: np.ndarray) -> np.ndarray:
        """
        Swap and then flatten axes 0 (buffer_size) and 1 (n_envs)
        to convert shape from [n_steps, n_envs, ...] (when ... is the shape of the features)
        to [n_steps * n_envs, ...] (which maintain the order)

        :param arr:
        :return:
        """
        shape = arr.shape
        if len(shape) < 3:
            shape = (*shape, 1)
        return arr.swapaxes(0, 1).reshape(shape[0] * shape[1], *shape[2:])

def toThreeDigitsEncoding(num):
    res = []
    res.append(num // 121)
    num %= 121
    res.append(num // 11)
    res.append(num % 11)
    return res

def toOneHotEncoding(num):
    res = np.zeros((11, 11, 11)).reshape(-1)
    res[num] = 1
    res = res.reshape(11, 11, 11)
    return torch.Tensor(res)

def pad(seq_start_indices: np.ndarray,
        seq_end_indices: np.ndarray,
        device: torch.device,
        tensor: np.ndarray,
        padding_value: float = 0.0,
    ) -> torch.Tensor:
        """
        Chunk sequences and pad them to have constant dimensions.

        :param seq_start_indices: Indices of the transitions that start a sequence
        :param seq_end_indices: Indices of the transitions that end a sequence
        :param device: PyTorch device
        :param tensor: Tensor of shape (batch_size, *tensor_shape)
        :param padding_value: Value used to pad sequence to the same length
            (zero padding by default)
        :return: (n_seq, max_length, *tensor_shape)
        """
        # Create sequences given start and end
        seq = [torch.tensor(tensor[start: end + 1], device=device) for start, end in
               zip(seq_start_indices, seq_end_indices)]
        return torch.nn.utils.rnn.pad_sequence(seq, batch_first=True, padding_value=padding_value)

def pad_and_flatten(seq_start_indices: np.ndarray,
            seq_end_indices: np.ndarray,
            device: torch.device,
            tensor: np.ndarray,
            padding_value: float = 0.0,
    ) -> torch.Tensor:
        """
        Pad and flatten the sequences of scalar values,
        while keeping the sequence order.
        From (batch_size, 1) to (n_seq, max_length, 1) -> (n_seq * max_length,)

        :param seq_start_indices: Indices of the transitions that start a sequence
        :param seq_end_indices: Indices of the transitions that end a sequence
        :param device: PyTorch device (cpu, gpu, ...)
        :param tensor: Tensor of shape (max_length, n_seq, 1)
        :param padding_value: Value used to pad sequence to the same length
            (zero padding by default)
        :return: (n_seq * max_length,) aka (padded_batch_size,)
        """
        return pad(seq_start_indices, seq_end_indices, device, tensor, padding_value).flatten()

class Robot_1(Robot):
    def getAction(self, obs=None):
        self.action = [0, 0, 0]
        if self.sum > self.maxOprCoins:
            self.action[0] = random.randint(1, self.maxOprCoins)
        elif self.sum > 0:
            self.action[0] = random.randint(1, self.sum)
        self.sum -= self.action[0]
        return self.action

class Robot_2(Robot):
    def getAction(self, obs):
        self.action = [0, 0, 0]
        if self.sum > self.maxOprCoins:
            self.action[2] = random.randint(1, self.maxOprCoins)
        elif self.sum > 0:
            self.action[2] = random.randint(1, self.sum)
        self.sum -= self.action[2]
        if obs[0] != 0:
            if self.sum > self.maxOprCoins:
                self.action[0] = random.randint(1, self.maxOprCoins)
            elif self.sum > 0:
                self.action[0] = random.randint(1, self.sum)
                self.sum -= self.action[0]
        return self.action

class Robot_3(Robot):
    def getAction(self, obs):
        self.action = [0, 0, 0]
        if obs[1] != 0:
            if self.sum > self.maxOprCoins:
                if obs[1] == 9 or obs[1] == 10:
                    self.action[1] = 10
                else:
                    self.action[1] = random.randint(obs[1] + 1, self.maxOprCoins)
            elif self.sum > obs[1]:
                self.action[1] = random.randint(obs[1] + 1, self.sum)
            self.maxOprCoins -= self.action[1]
            self.sum -= self.action[1]
        if obs[0] != 0 and self.maxOprCoins != 0:
            if self.sum > self.maxOprCoins:
                self.action[0] = random.randint(1, self.maxOprCoins)
            elif self.sum > 0:
                self.action[0] = random.randint(1, self.sum)
            self.maxOprCoins -= self.action[0]
            self.sum -= self.action[0]
        if obs[2] != 0 and self.maxOprCoins != 0:
            if self.sum > self.maxOprCoins:
                self.action[2] = random.randint(1, self.maxOprCoins)
            elif self.sum > 0:
                self.action[2] = random.randint(1, self.sum)
            self.maxOprCoins -= self.action[2]
            self.sum -= self.action[2]
        if sum(obs) is 0:
            if self.sum > self.maxOprCoins:
                self.action[2] = random.randint(1, self.maxOprCoins)
            elif self.sum > 0:
                self.action[2] = random.randint(1, self.sum)
            self.sum -= self.action[2]
            np.random.shuffle(self.action)
        self.maxOprCoins = 10
        return self.action

class Robot_4(Robot):
    def getAction(self, obs):
        self.action = [0, 0, 0]
        if obs[0] is 0 and self.sum > 0:
            self.action[0] = 1
            self.maxOprCoins -= self.action[0]
            self.sum -= self.action[0]
        if obs[1] is 0 and self.sum > 0:
            self.action[1] = 1
            self.maxOprCoins -= self.action[1]
            self.sum -= self.action[1]
        # if obs[1] > 0 and obs[1] < 10:
        #     if self.sum > obs[1] and self.maxOprCoins > obs[1]:
        #         self.action[1] = self.maxOprCoins
        if obs[2] != 0:
            if self.sum > self.maxOprCoins:
                self.action[2] = self.maxOprCoins
            elif self.sum > 0:
                self.action[2] = self.sum
            self.sum -= self.action[2]
        self.maxOprCoins = 10
        return self.action

input_data = {}
output_data = {}
def get_activation(name):
    def hook(model, input, output):
        input_data[name] = input[0].detach()  # input type is tulple, only has one element, which is the tensor
        output_data[name] = output.detach()  # output type is tensor
    return hook

# Define the rules of the baskets
# def rule_1(input_1, input_2):
#     output_1, output_2 = 0, 0
#     if input_1 > 0 and input_2 > 0:
#         output_1 = 2 * input_1
#         output_2 = 2 * input_2
#     elif input_1 != 0 and input_2 == 0:
#         output_2 = 2 * input_1
#     elif input_1 == 0 and input_2 != 0:
#         output_1 = 2 * input_2
#     return [output_1, output_2]
#
# def rule_2(input_1 = 0, input_2 = 0):
#     output_1, output_2 = 0, 0
#     if input_1 > 0 and input_2 > 0:
#         if input_1 > input_2:
#             output_1 = 3 * (input_1 + input_2)
#         elif input_1 < input_2:
#             output_2 = 3 * (input_1 + input_2)
#         else:
#             output_1 = 3 * (input_1 + input_2)
#             output_2 = 3 * (input_1 + input_2)
#     elif input_1 != 0 and input_2 == 0:
#         output_2 = 2 * input_1
#     elif input_1 == 0 and input_2 != 0:
#         output_1 = 2 * input_2
#     return [output_1, output_2]
#
# def rule_3(input_1 = 0, input_2 = 0):
#     output_1, output_2 = 0, 0
#     if input_1 > 0 and input_2 > 0:
#         output_1 = 2 * input_1
#         output_2 = 2 * input_2
#     return [output_1, output_2]

if __name__== "__main__" :

    # basket_1 = Basket()
    # basket_2 = Basket()
    # basket_3 = Basket()
    # basket_1.setRule(rule_1)
    # basket_2.setRule(rule_2)
    # basket_3.setRule(rule_3)
    # baskets = []
    # baskets.append(basket_1)
    # baskets.append(basket_2)
    # baskets.append(basket_3)

    parser = argparse.ArgumentParser(description='PPO agent')
    # Comment arguments
    parser.add_argument('--exp-name', type=str, default=os.path.basename(__file__).rstrip(".py"),
                        help='the name of this experiment')
    parser.add_argument('--gym-id', type=str, default="gym_coinsgame:coinsgame-v0",
                        help='the id of the gym environment')
    parser.add_argument('--seed', type=int, default=3407,
                        help='seed of the experiment')
    parser.add_argument('--episode-length', type=int, default=0,
                        help='the maximum length of each episode')
    parser.add_argument('--total-timesteps', type=int, default=100000,
                        help='total timesteps of the experiments')
    parser.add_argument('--no-torch-deterministic', action='store_false', dest="torch_deterministic", default=True,
                        help='if toggled, `torch.backends.cudnn.deterministic=False`')
    parser.add_argument('--no-cuda', action='store_false', dest="cuda", default=True,
                        help='if toggled, cuda will not be enabled by default')
    parser.add_argument('--prod-mode', action='store_true', default=False,
                        help='run the script in production mode and use wandb to log outputs')
    # parser.add_argument('--capture-video', action='store_true', default=False,
    #                     help='weather to capture videos of the agent performances (check out `videos` folder)')
    # parser.add_argument('--wandb-project-name', type=str, default="cleanRL",
    #                     help="the wandb's project name")
    # parser.add_argument('--wandb-entity', type=str, default=None,
    #                     help="the entity (team) of wandb's project")
    # Algorithm specific arguments
    parser.add_argument('--batch-size', type=int, default=2048,
                        help='the batch size of ppo')
    parser.add_argument('--minibatch-size', type=int, default=256,
                        help='the mini batch size of ppo')
    parser.add_argument('--gamma', type=float, default=0.95,
                        help='the discount factor gamma')
    parser.add_argument('--gae-lambda', type=float, default=0.95,
                        help='the lambda for the general advantage estimation')
    parser.add_argument('--ent-coef', type=float, default=0.3,
                        help="coefficient of the entropy")
    parser.add_argument('--max-grad-norm', type=float, default=0.3,
                        help='the maximum norm for the gradient clipping')
    parser.add_argument('--clip-coef', type=float, default=0.5,
                        help="the surrogate clipping coefficient")
    parser.add_argument('--update-epochs', type=int, default=10,
                        help="the K epochs to update the policy")
    parser.add_argument('--kle-stop', action='store_true', default=False,
                        help='If toggled, the policy updates will be early stopped w.r.t target-kl')
    parser.add_argument('--kle-rollback', action='store_true', default=False,
                        help='If toggled, the policy updates will roll back to previous policy if KL exceeds target-kl')
    parser.add_argument('--target-kl', type=float, default=0.015,
                        help='the target-kl variable that is referred by --kl')
    parser.add_argument('--gae', action='store_true', default=True,
                        help='Use GAE for advantage computation')
    parser.add_argument('--policy-lr', type=float, default=3e-4,
                        help="the learning rate of the policy optimizer")
    parser.add_argument('--value-lr', type=float, default=3e-4,
                        help="the learning rate of the critic optimizer")
    parser.add_argument('--norm-obs', action='store_true', default=True,
                        help="Toggles observation normalization")
    parser.add_argument('--norm-returns', action='store_true', default=False,
                        help="Toggles returns normalization")
    parser.add_argument('--norm-adv', action='store_true', default=True,
                        help="Toggles advantages normalization")
    parser.add_argument('--obs-clip', type=float, default=10.0,
                        help="Value for reward clipping, as per the paper")
    parser.add_argument('--rew-clip', type=float, default=10.0,
                        help="Value for observation clipping, as per the paper")
    parser.add_argument('--anneal-lr', action='store_true', default=True,
                        help="Toggle learning rate annealing for policy and value networks")
    parser.add_argument('--weights-init', default="orthogonal", choices=["xavier", 'orthogonal'],
                        help='Selects the scheme to be used for weights initialization')
    parser.add_argument('--clip-vloss', action="store_true", default=True,
                        help='Toggles whether or not to use a clipped loss for the value function, as per the paper.')
    parser.add_argument('--pol-layer-norm', action='store_true', default=False,
                        help='Enables layer normalization in the policy network')
    parser.add_argument('--lstm-step', type=int, default=8,
                        help='Applies a multi-layer long short-term memory (LSTM) RNN to an input step')
    parser.add_argument('--lstm-sequence', type=int, default=128,
                        help='Applies a multi-layer long short-term memory (LSTM) RNN to an input sequence')
    args = parser.parse_args()
    if not args.seed:
        args.seed = int(time.time())

    config = dict(batch_size=args.batch_size,
                  minibatch_size=args.minibatch_size,
                  gamma=args.gamma,
                  gae_lambda=args.gae_lambda,
                  ent_coef=args.ent_coef,
                  max_grad_norm=args.max_grad_norm,
                  clip_coef=args.clip_coef,
                  update_epochs=args.update_epochs,
                  target_kl=args.target_kl,
                  policy_lr=args.policy_lr,
                  value_lr=args.value_lr,
                  lstm_step=args.lstm_step,
                  lstm_sequence=args.lstm_sequence)
    wandb.init(project="coinsgame-project",
               name="lstm",
               config=config)
    # TRY NOT TO MODIFY: seeding
    args.features_turned_on = sum([args.kle_stop, args.kle_rollback, args.gae, args.norm_obs, args.norm_returns, args.norm_adv, args.anneal_lr, args.clip_vloss, args.pol_layer_norm])
    # writer = SummaryWriter(f"runs/{args.gym_id}")
    # writer.add_text('hyperparameters', "|param|value|\n|-|-|\n%s" % (
    #     '\n'.join([f"|{key}|{value}|" for key, value in vars(args).items()])))
    device = torch.device('cuda' if torch.cuda.is_available() and args.cuda else 'cpu')
    env = gym.make(args.gym_id)
    # respect the default timelimit
    # assert isinstance(env.action_space, MultiDiscrete), "only MultiDiscrete action space is supported"
    # assert isinstance(env, TimeLimit) or int(
    #     args.episode_length), "the gym env does not have a built in TimeLimit, please specify by using --episode-length"
    # if isinstance(env, TimeLimit):
    #     if int(args.episode_length):
    #         env._max_episode_steps = int(args.episode_length)
    #     args.episode_length = env._max_episode_steps
    # else:
    #     env = TimeLimit(env, int(args.episode_length))
    args.episode_length = 500
    env = TimeLimit(env, int(args.episode_length))
    env = NormalizedEnv(env.env, ob=args.norm_obs, ret=args.norm_returns, clipob=args.obs_clip, cliprew=args.rew_clip, gamma=args.gamma)
    env = TimeLimit(env, int(args.episode_length))
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic
    env.seed(args.seed)
    env.action_space.seed(args.seed)
    env.observation_space.seed(args.seed)

    agent_1 = Agent()
    agent_2 = Robot_3()

    # MODIFIED: Separate optimizer and learning rates
    agent_1.pg_optimizer = optim.Adam(list(agent_1.pg.parameters()), lr=args.policy_lr)
    agent_1.v_optimizer = optim.Adam(list(agent_1.vf.parameters()), lr=args.value_lr)

    # MODIFIED: Initializing learning rate anneal scheduler when need
    if args.anneal_lr:
        anneal_fn = lambda f: max(0, 1 - f / args.total_timesteps)
        agent_1.pg_lr_scheduler = optim.lr_scheduler.LambdaLR(agent_1.pg_optimizer, lr_lambda=anneal_fn)
        agent_1.vf_lr_scheduler = optim.lr_scheduler.LambdaLR(agent_1.v_optimizer, lr_lambda=anneal_fn)

    # LSTM
    single_hidden_state_shape = (agent_1.pg.lstm.num_layers, 1, agent_1.pg.lstm.hidden_size)
    agent_1.last_lstm_states = RNNStates(
        (
            torch.zeros(single_hidden_state_shape, device=device),
            torch.zeros(single_hidden_state_shape, device=device),
        ),
        (
            torch.zeros(single_hidden_state_shape, device=device),
            torch.zeros(single_hidden_state_shape, device=device),
        ),
    )
    hidden_state_buffer_shape = (args.batch_size, agent_1.pg.lstm.num_layers, 1, agent_1.pg.lstm.hidden_size)

    # Buffer
    buffer = Buffer(args.batch_size, args.gamma, args.gae_lambda, 8, 1, (11, 11, 11),
                        hidden_state_buffer_shape, device)


    # TRY NOT TO MODIFY: start the game
    global_step = 0
    round = 0
    winTimeSum = [0, 0, 0, 0]
    while global_step < args.total_timesteps:
        # if args.capture_video:
        #     env.stats_recorder.done = True

        raw_next_obs = np.array(env.env.env.reset())
        robotObs = raw_next_obs[0:3]
        next_obs = np.array(env.reset())

        # ALGO Logic: Storage for epoch data
        agent_1.obs = np.empty((args.batch_size, 8))
        agent_2.sum = raw_next_obs[-1]

        # agent_1.actions = np.empty((args.batch_size, 11, 11, 11))
        agent_1.actions = np.empty((args.batch_size, 1))
        agent_1.logprobs = torch.zeros((args.batch_size, 1)).to(device)

        rewards = np.zeros((args.batch_size, 2))
        agent_1.rewards = np.zeros((args.batch_size,))
        agent_2.rewards = np.zeros((args.batch_size,))

        agent_1.episode_rewards = []
        agent_2.episode_rewards = []
        # invalid_action_stats = []

        dones = np.zeros((args.batch_size,))
        agent_1.values = torch.zeros((args.batch_size,)).to(device)

        agent_1.invalid_action_masks = torch.zeros((args.batch_size, 11, 11, 11))

        #Buffer
        buffer.reset()

        # LSTM
        agent_1.lstm_states = deepcopy(agent_1.last_lstm_states)
        # agent_1.hidden_states_pi = np.zeros()
        # agent_1.cell_states_pi = np.zeros()
        # agent_1.hidden_states_vf = np.zeros()
        # agent_1.cell_states_vf = np.zeros()

        winTime = [0, 0, 0, 0]
        # TRY NOT TO MODIFY: prepare the execution of the game.
        for step in range(args.batch_size):
            # env.render()
            global_step += 1
            round += 1
            agent_1.obs[step] = next_obs.reshape(-1).copy()

            # ALGO LOGIC: put action logic here
            agent_1.invalid_action_mask = torch.Tensor(env.env.env.actionMask(env.env.env.agents[0].sum))
            agent_1.invalid_action_masks[step] = agent_1.invalid_action_mask

            if round == 1:
                episode_starts = torch.tensor(1, dtype=torch.float32, device=device)
            else:
                episode_starts = torch.tensor(0, dtype=torch.float32, device=device)

            with torch.no_grad():
                agent_1.values[step], lstm_states_vf = agent_1.vf.get_value(agent_1.obs[step:step + 1], agent_1.lstm_states.vf, episode_starts)
                agent_1.action, agent_1.logproba, _, agent_1.probs, lstm_states_pi = agent_1.pg.get_action(agent_1.obs[step:step + 1],
                                                           invalid_action_masks=agent_1.invalid_action_masks[step:step + 1], lstm_states=agent_1.lstm_states.pi, episode_starts=episode_starts)
            agent_1.lstm_states = RNNStates(lstm_states_pi, lstm_states_vf)
            agent_1.actions[step] = agent_1.action[0].cpu().data
            agent_1.logprobs[step] = agent_1.logproba

            # TRY NOT TO MODIFY: execute the game and log data.
            action = [toThreeDigitsEncoding(agent_1.action[0].data), agent_2.getAction(robotObs)]
            wandb.log({'actions_1/agent_1_basket_1': action[0][0],
                       'actions_1/agent_1_basket_2': action[0][1],
                       'actions_1/agent_1_basket_3': action[0][2],
                       'actions_2/agent_2_basket_1': action[1][0],
                       'actions_2/agent_2_basket_2': action[1][1],
                       'actions_2/agent_2_basket_3': action[1][2]}, step = global_step)
            next_obs, rewards[step], dones[step], info = env.step(action)
            robotObs = info['action_1']
            agent_1.rewards[step] = rewards[step][0]
            agent_1.episode_rewards += [info['episode_reward'][0]]
            agent_2.rewards[step] = rewards[step][1]
            agent_2.episode_rewards += [info['episode_reward'][1]]
            # update the coins of robot
            agent_2.sum = info['coins_2']
            # invalid_action_stats += [info['invalid_action_stats']]
            next_obs = np.array(next_obs)
            wandb.log({'rewards/step_reward_1':agent_1.rewards[step],
                       'rewards/step_reward_2':agent_2.rewards[step]}, step = global_step)

            # Annealing the rate if instructed to do so.
            if args.anneal_lr:
                agent_1.pg_lr_scheduler.step()
                agent_1.vf_lr_scheduler.step()

            # Buffer
            # agent_1.lstm_states = RNNStates(agent_1.lstm_states.pi, agent_1.lstm_states.vf)
            buffer.add(agent_1.obs[step:step + 1], agent_1.actions[step], agent_1.invalid_action_masks[step],
                       agent_1.rewards[step], episode_starts, agent_1.values[step],
                       agent_1.logprobs[step], agent_1.last_lstm_states)
            agent_1.last_lstm_states = agent_1.lstm_states

            if dones[step]:
                # Computing the discounted returns:
                wandb.log({'rewards/episode_reward_1': np.sum(agent_1.episode_rewards),
                           'rewards/episode_reward_2': np.sum(agent_2.episode_rewards),
                           "episode_length": round}, step = global_step)
                if info['winner'] == 0:
                    winTime[0] += 1
                    winTimeSum[0] += 1
                    wandb.log({'winner/Draw': winTime[0]}, step = global_step)
                    wandb.log({'winTime/Draw': winTimeSum[0]}, step=global_step)
                elif info['winner'] == 1:
                    winTime[1] += 1
                    winTimeSum[1] += 1
                    wandb.log({'winner/Player_1': winTime[1]}, step=global_step)
                    wandb.log({'winTime/Player_1': winTimeSum[1]}, step=global_step)
                elif info['winner'] == 2:
                    winTime[2] += 1
                    winTimeSum[2] += 1
                    wandb.log({'winner/Player_2': winTime[2]}, step=global_step)
                    wandb.log({'winTime/Player_2': winTimeSum[2]}, step=global_step)
                elif info['winner'] == 3:
                    winTime[3] += 1
                    winTimeSum[3] += 1
                    wandb.log({'winner/Nobody': winTime[3]}, step=global_step)
                    wandb.log({'winTime/Nobody': winTimeSum[3]}, step=global_step)
                round = 0
                # writer.add_scalar("charts/episode_reward_1", np.sum(agent_1.episode_rewards), global_step)
                # writer.add_scalar("charts/episode_reward_2", np.sum(agent_2.episode_rewards), global_step)
                print(f"global_step={global_step}, episode_reward_1={np.sum(agent_1.episode_rewards)}, episode_reward_2={np.sum(agent_2.episode_rewards)}")
                with open("records/test.txt", 'a', encoding="utf-8") as file:
                    file.write("global_step = %d, episode_reward_1 = %d, episode_reward_2 = %d\n" % (global_step, np.sum(agent_1.episode_rewards), np.sum(agent_2.episode_rewards)))
                agent_1.episode_rewards = []
                agent_2.episode_rewards = []
                # for key, idx in zip(info['invalid_action_stats'], range(len(info['invalid_action_stats']))):
                #     writer.add_scalar(f"stats/{key}", pd.DataFrame(invalid_action_stats).sum(0)[idx], global_step)
                # invalid_action_stats = []
                next_obs = np.array(env.reset())
                raw_next_obs = np.array(env.env.env.reset())
                agent_2.sum = raw_next_obs[-1]

        # bootstrap reward if not done. reached the batch limit
        # last_value = 0
        # if not dones[step]:
        #     agent_1.last_value, _ = agent_1.vf.get_value(next_obs.reshape(1, -1)).detach().cpu().numpy()
        # agent_1.bootstrapped_rewards = np.append(agent_1.rewards, agent_1.last_value)

        with torch.no_grad():
            # Compute value for the last timestep
            if dones[step]:
                episode_starts = torch.tensor(1, dtype=torch.float32, device=device)
            else:
                episode_starts = torch.tensor(0, dtype=torch.float32, device=device)
            agent_1.last_value, _ = agent_1.vf.get_value(next_obs.reshape(1, -1), agent_1.lstm_states.vf, episode_starts)
        # if isinstance(agent_1.last_value, int):
        #     agent_1.bootstrapped_rewards = np.append(agent_1.rewards, agent_1.last_value)
        # else:
        #     agent_1.bootstrapped_rewards = np.append(agent_1.rewards, agent_1.last_value.cpu().numpy())
        buffer.compute_returns_and_advantage(agent_1.last_value, dones[step])

        # calculate the returns and advantages
        # if args.gae:
        #     agent_1.bootstrapped_values = np.append(agent_1.values.cpu(), agent_1.last_value.cpu())
        #     agent_1.deltas = agent_1.bootstrapped_rewards[:-1] + args.gamma * agent_1.bootstrapped_values[1:] * (
        #                 1 - dones) - agent_1.bootstrapped_values[:-1]
        #     agent_1.advantages = discount_cumsum(agent_1.deltas, dones, args.gamma * args.gae_lambda)
        #     agent_1.advantages = torch.Tensor(agent_1.advantages).to(device)
        #     agent_1.returns = agent_1.advantages + agent_1.values
        #     for i in range(args.batch_size):
        #         buffer.advantages[i] = agent_1.advantages[i].cpu().numpy()
        #         buffer.returns[i] = agent_1.returns[i].cpu().numpy()
        # else:
        #     agent_1.returns = discount_cumsum(agent_1.bootstrapped_rewards, dones, args.gamma)[:-1]
        #     agent_1.advantages = agent_1.returns - agent_1.values.detach().cpu().numpy()
        #     agent_1.advanftages = torch.Tensor(agent_1.advantages).to(device)
        #     agent_1.returns = torch.Tensor(agent_1.returns).to(device)
        #     for i in range(args.batch_size):
        #         buffer.advantages[i] = agent_1.advantages[i]
        #         buffer.returns[i] = agent_1.returns[i]

        # Advantage normalization
        # if args.norm_adv:
        #     EPS = 1e-10
        #     agent_1.advantages = (agent_1.advantages - agent_1.advantages.mean()) / (agent_1.advantages.std() + EPS)

        # Optimizaing policy network
        agent_1.entropys = []
        entropy_loss = []
        pg_losses, value_losses = [], []
        # agent_1.target_pg = Policy().to(device)
        # agent_1.inds = np.arange(args.batch_size, )
        for i_epoch_pi in range(args.update_epochs):
            # np.random.shuffle(agent_1.inds)
            approx_kl_divs = []
            for rollout_data in buffer.get(args.batch_size):
                mask = rollout_data.mask > 1e-8

            # for start in range(0, args.batch_size, args.minibatch_size):
            #     end = start + args.minibatch_size
            #     agent_1.minibatch_ind = agent_1.inds[start:end]
            #     agent_1.target_pg.load_state_dict(agent_1.pg.state_dict())

                # _, agent_1.newlogproba, _, _ = agent_1.pg.get_action(
                #     agent_1.obs[agent_1.minibatch_ind],
                #     torch.LongTensor(agent_1.actions[agent_1.minibatch_ind].astype(int)).to(device).T,
                #     agent_1.invalid_action_masks[agent_1.minibatch_ind,:])
                agent_1.new_values, _ = agent_1.vf.get_value(
                    rollout_data.observations,
                    rollout_data.lstm_states.vf,
                    rollout_data.episode_starts
                )

                _, agent_1.newlogproba, list, _, _ = agent_1.pg.get_action(
                    rollout_data.observations,
                    rollout_data.actions,
                    rollout_data.invalid_action_masks,
                    rollout_data.lstm_states.pi,
                    rollout_data.episode_starts
                )

                agent_1.advantages = rollout_data.advantages
                if args.norm_adv:
                    agent_1.advantages = (agent_1.advantages - agent_1.advantages[mask].mean()) / (agent_1.advantages[mask].std() + 1e-8)

                agent_1.ratio = torch.exp(agent_1.newlogproba - rollout_data.old_log_prob.reshape(-1, 1)).reshape(-1)
                agent_1.returns = rollout_data.returns
                agent_1.values = rollout_data.old_values.flatten()

                # Policy loss as in OpenAI SpinUp
                # agent_1.clip_adv = torch.where(agent_1.advantages > 0,
                #                        (1. + args.clip_coef) * agent_1.advantages,
                #                        (1. - args.clip_coef) * agent_1.advantages).to(device)

                # Entropy computation with resampled actions
                agent_1.entropy = agent_1.newlogproba[mask].exp() * agent_1.newlogproba[mask]
                # agent_1.entropy = list[0]
                agent_1.entropy = -torch.mean((agent_1.entropy.to(device)))
                # agent_1.entropys.append(agent_1.entropy.item())
                entropy_loss.append(agent_1.entropy.item())

                policy_loss_1 = agent_1.advantages * agent_1.ratio
                policy_loss_2 = agent_1.advantages * torch.clamp(agent_1.ratio, 1. - args.clip_coef, 1. + args.clip_coef)
                # policy_loss_2 = torch.where(agent_1.advantages > 0,
                #                             (1. + args.clip_coef) * agent_1.advantages,
                #                             (1. - args.clip_coef) * agent_1.advantages)
                # agent_1.policy_loss = -torch.mean((torch.min(policy_loss_1, policy_loss_2)[mask])) + args.ent_coef * agent_1.entropy
                agent_1.policy_loss = -torch.min(policy_loss_1, policy_loss_2)[mask].reshape(1, -1) + args.ent_coef * agent_1.entropy
                agent_1.policy_loss = torch.mean(agent_1.policy_loss)
                pg_losses.append(agent_1.policy_loss.item())

                agent_1.pg_optimizer.zero_grad()
                agent_1.policy_loss.backward()
                nn.utils.clip_grad_norm_(agent_1.pg.parameters(), args.max_grad_norm)
                agent_1.pg_optimizer.step()

                # agent_1.approx_kl = (agent_1.logprobs[:, agent_1.minibatch_ind] - agent_1.newlogproba).mean()
                with torch.no_grad():
                    log_ratio = agent_1.newlogproba.reshape(-1) - rollout_data.old_log_prob
                    agent_1.approx_kl = torch.mean(((torch.exp(log_ratio) - 1) - log_ratio)[mask]).cpu().numpy()
                    approx_kl_divs.append(agent_1.approx_kl)

                # Optimizing value network
                # agent_1.new_values = agent_1.vf.forward(agent_1.obs[agent_1.minibatch_ind]).view(-1)

                # Value loss clipping
                # if args.clip_vloss:
                #     agent_1.v_loss_unclipped = (((agent_1.new_values - agent_1.returns) ** 2)[mask])
                #     agent_1.v_clipped = agent_1.values + torch.clamp(agent_1.new_values - agent_1.values,
                #                                     -args.clip_coef, args.clip_coef)
                #     agent_1.v_loss_clipped = ((agent_1.v_clipped - agent_1.returns) ** 2[mask])
                #     agent_1.v_loss_max = torch.max(agent_1.v_loss_unclipped, agent_1.v_loss_clipped)
                #     agent_1.v_loss = 0.5 * agent_1.v_loss_max.mean()
                # else:
                #     agent_1.v_loss = torch.mean((agent_1.returns - agent_1.new_values).pow(2)[mask])
                if args.clip_vloss:
                    # values_pred = agent_1.values + torch.clamp(agent_1.new_values - agent_1.values, -args.clip_coef, args.clip_coef)
                    agent_1.v_loss_unclipped = ((agent_1.new_values.reshape(-1) - agent_1.returns) ** 2)
                    agent_1.v_clipped = agent_1.values + torch.clamp(agent_1.new_values.reshape(-1) - agent_1.values, -args.clip_coef, args.clip_coef)
                    agent_1.v_loss_clipped = (agent_1.v_clipped - agent_1.returns) ** 2
                    agent_1.v_loss_max = torch.max(agent_1.v_loss_unclipped, agent_1.v_loss_clipped)[mask]
                    agent_1.v_loss = 0.5 * agent_1.v_loss_max.mean()
                else:
                    # values_pred = agent_1.values
                    agent_1.v_loss = torch.mean((agent_1.returns - agent_1.new_values.reshape(-1)).pow(2)[mask])
                # agent_1.v_loss = 0.5 * torch.mean(((agent_1.returns - values_pred) ** 2)[mask])
                value_losses.append(agent_1.v_loss.item())

                loss = agent_1.policy_loss + agent_1.entropy * args.ent_coef + agent_1.v_loss

                agent_1.v_optimizer.zero_grad()
                agent_1.v_loss.backward()
                nn.utils.clip_grad_norm_(agent_1.vf.parameters(), args.max_grad_norm)
                agent_1.v_optimizer.step()

            if args.kle_stop:
                if agent_1.approx_kl > args.target_kl:
                    break
            # if args.kle_rollback:
            #     if (agent_1.logprobs[:, agent_1.minibatch_ind] -
            #         agent_1.pg.get_action(
            #             agent_1.obs[agent_1.minibatch_ind],
            #             torch.LongTensor(agent_1.actions[agent_1.minibatch_ind].astype(np.int)).to(device).T,
            #             agent_1.invalid_action_masks[agent_1.minibatch_ind])[1]).mean() > args.target_kl:
            #         agent_1.pg.load_state_dict(agent_1.target_pg.state_dict())
            #         break


        wandb.log({'losses/value_1':np.mean(value_losses),
                   'losses/policy_1':np.mean(pg_losses),
                   'learning_rates/policy_1':agent_1.pg_optimizer.param_groups[0]['lr'],
                   'learning_rates/value_1':agent_1.v_optimizer.param_groups[0]['lr'],
                   'losses/entropy_1':np.mean(entropy_loss),
                   'losses/approx_kl_1':np.mean(approx_kl_divs)}, step = global_step)
        if args.kle_stop or args.kle_rollback:
            wandb.log({'pg_stop_iter':i_epoch_pi})

        # with open("records/p_ln_8.txt", 'a', encoding="utf-8") as file_p_0:
            # file_p_0.write("*******************global_step = %d*****************\n" % (global_step))
            # file_p_0.write("p_ln_8_in\n")
            # np.savetxt(file_p_0, np.array(input_data['p_ln_8'].cpu()), fmt='%.5f')
            # file_p_0.write("p_ln_8_out\n")
            # np.savetxt(file_p_0, np.array(output_data['p_ln_8'].cpu()), fmt='%.5f')
        # with open("records/p_linear_8-128.txt", 'a', encoding="utf-8") as file_p_1:
        #     file_p_1.write("*******************global_step = %d*****************\n" % (global_step))
        #     file_p_1.write("p_linear_8-128\n")
        #     np.savetxt(file_p_1, np.array(output_data['p_linear_8-128'].cpu()), fmt='%.5f')
        # with open("records/p_ln_128.txt", 'a', encoding="utf-8") as file_p_2:
        #     file_p_2.write("*******************global_step = %d*****************\n" % (global_step))
        #     file_p_2.write("p_ln_128\n")
        #     np.savetxt(file_p_2, np.array(output_data['p_ln_128'].cpu()), fmt='%.5f')
        # with open("records/p_leakyreLU_128.txt", 'a', encoding="utf-8") as file_p_3:
        #     file_p_3.write("*******************global_step = %d*****************\n" % (global_step))
        #     file_p_3.write("p_leakyreLU_128\n")
        #     np.savetxt(file_p_3, np.array(output_data['p_leakyreLU_128'].cpu()), fmt='%.5f')
        # with open("records/p_linear_128-256.txt", 'a', encoding="utf-8") as file_p_4:
        #     file_p_4.write("*******************global_step = %d*****************\n" % (global_step))
        #     file_p_4.write("p_linear_128-256\n")
        #     np.savetxt(file_p_4, np.array(output_data['p_linear_128-256'].cpu()), fmt='%.5f')
        # with open("records/p_ln_256.txt", 'a', encoding="utf-8") as file_p_5:
        #     file_p_5.write("*******************global_step = %d*****************\n" % (global_step))
        #     file_p_5.write("p_ln_256\n")
        #     np.savetxt(file_p_5, np.array(output_data['p_ln_256'].cpu()), fmt='%.5f')
        # with open("records/p_leakyreLU_256.txt", 'a', encoding="utf-8") as file_p_6:
        #     file_p_6.write("*******************global_step = %d*****************\n" % (global_step))
        #     file_p_6.write("p_leakyreLU_256\n")
        #     np.savetxt(file_p_6, np.array(output_data['p_leakyreLU_256'].cpu()), fmt='%.5f')
        # with open("records/p_linear_256-512.txt", 'a', encoding="utf-8") as file_p_7:
        #     file_p_7.write("*******************global_step = %d*****************\n" % (global_step))
        #     file_p_7.write("p_linear_256-512\n")
        #     np.savetxt(file_p_7, np.array(output_data['p_linear_256-512'].cpu()), fmt='%.5f')
        # with open("records/p_ln_512.txt", 'a', encoding="utf-8") as file_p_8:
        #     file_p_8.write("*******************global_step = %d*****************\n" % (global_step))
        #     file_p_8.write("p_ln_512\n")
        #     np.savetxt(file_p_8, np.array(output_data['p_ln_512'].cpu()), fmt='%.5f')
        # with open("records/p_leakyreLU_512.txt", 'a', encoding="utf-8") as file_p_9:
        #     file_p_9.write("*******************global_step = %d*****************\n" % (global_step))
        #     file_p_9.write("p_leakyreLU_512\n")
        #     np.savetxt(file_p_9, np.array(output_data['p_leakyreLU_512'].cpu()), fmt='%.5f')
        # with open("records/p_linear_512-1024.txt", 'a', encoding="utf-8") as file_p_10:
        #     file_p_10.write("*******************global_step = %d*****************\n" % (global_step))
        #     file_p_10.write("p_linear_512-1024\n")
        #     np.savetxt(file_p_10, np.array(output_data['p_linear_512-1024'].cpu()), fmt='%.5f')
        # with open("records/p_leakyreLU_1024.txt", 'a', encoding="utf-8") as file_p_11:
        #     file_p_11.write("*******************global_step = %d*****************\n" % (global_step))
        #     file_p_11.write("p_leakyreLU_1024\n")
        #     np.savetxt(file_p_11, np.array(output_data['p_leakyreLU_1024'].cpu()), fmt='%.5f')
        # with open("records/p_linear_1024-2048.txt", 'a', encoding="utf-8") as file_p_12:
        #     file_p_12.write("*******************global_step = %d*****************\n" % (global_step))
        #     file_p_12.write("p_linear_1024-2048\n")
        #     np.savetxt(file_p_12, np.array(output_data['p_linear_1024-2048'].cpu()), fmt='%.5f')
        # with open("records/p_leakyreLU_2048.txt", 'a', encoding="utf-8") as file_p_13:
        #     file_p_13.write("*******************global_step = %d*****************\n" % (global_step))
        #     file_p_13.write("p_leakyreLU_2048\n")
        #     np.savetxt(file_p_13, np.array(output_data['p_leakyreLU_2048'].cpu()), fmt='%.5f')
        # with open("records/p_linear_2048-1331.txt", 'a', encoding="utf-8") as file_p_14:
        #     file_p_14.write("*******************global_step = %d*****************\n" % (global_step))
        #     file_p_14.write("p_linear_2048-1331\n")
        #     np.savetxt(file_p_14, np.array(output_data['p_linear_2048-1331'].cpu()), fmt='%.5f')
        #
        # with open("records/v_ln_8.txt", 'a', encoding="utf-8") as file_v_0:
        #     file_v_0.write("*******************global_step = %d*****************\n" % (global_step))
        #     file_v_0.write("v_ln_8_in\n")
        #     np.savetxt(file_v_0, np.array(input_data['v_ln_8'].cpu()), fmt='%.5f')
        #     file_v_0.write("v_ln_8_out\n")
        #     np.savetxt(file_v_0, np.array(output_data['v_ln_8'].cpu()), fmt='%.5f')
        # with open("records/v_linear_8-128.txt", 'a', encoding="utf-8") as file_v_1:
        #     file_v_1.write("*******************global_step = %d*****************\n" % (global_step))
        #     file_v_1.write("v_linear_8-128\n")
        #     np.savetxt(file_v_1, np.array(output_data['v_linear_8-128'].cpu()), fmt='%.5f')
        # with open("records/v_ln_128.txt", 'a', encoding="utf-8") as file_v_2:
        #     file_v_2.write("*******************global_step = %d*****************\n" % (global_step))
        #     file_v_2.write("v_ln_128\n")
        #     np.savetxt(file_v_2, np.array(output_data['v_ln_128'].cpu()), fmt='%.5f')
        # with open("records/v_leakyreLU_128.txt", 'a', encoding="utf-8") as file_v_3:
        #     file_v_3.write("*******************global_step = %d*****************\n" % (global_step))
        #     file_v_3.write("v_leakyreLU_128\n")
        #     np.savetxt(file_v_3, np.array(output_data['v_leakyreLU_128'].cpu()), fmt='%.5f')
        # with open("records/v_linear_128-256.txt", 'a', encoding="utf-8") as file_v_4:
        #     file_v_4.write("*******************global_step = %d*****************\n" % (global_step))
        #     file_v_4.write("v_linear_128-256\n")
        #     np.savetxt(file_v_4, np.array(output_data['v_linear_128-256'].cpu()), fmt='%.5f')
        # with open("records/v_ln_256.txt", 'a', encoding="utf-8") as file_v_5:
        #     file_v_5.write("*******************global_step = %d*****************\n" % (global_step))
        #     file_v_5.write("v_ln_256\n")
        #     np.savetxt(file_v_5, np.array(output_data['v_ln_256'].cpu()), fmt='%.5f')
        # with open("records/v_leakyreLU_256.txt", 'a', encoding="utf-8") as file_v_6:
        #     file_v_6.write("*******************global_step = %d*****************\n" % (global_step))
        #     file_v_6.write("v_leakyreLU_256\n")
        #     np.savetxt(file_v_6, np.array(output_data['v_leakyreLU_256'].cpu()), fmt='%.5f')
        # with open("records/v_linear_256-512.txt", 'a', encoding="utf-8") as file_v_7:
        #     file_v_7.write("*******************global_step = %d*****************\n" % (global_step))
        #     file_v_7.write("v_linear_256-512\n")
        #     np.savetxt(file_v_7, np.array(output_data['v_linear_256-512'].cpu()), fmt='%.5f')
        # with open("records/v_ln_512.txt", 'a', encoding="utf-8") as file_v_8:
        #     file_v_8.write("*******************global_step = %d*****************\n" % (global_step))
        #     file_v_8.write("v_ln_512\n")
        #     np.savetxt(file_v_8, np.array(output_data['v_ln_512'].cpu()), fmt='%.5f')
        # with open("records/v_leakyreLU_512.txt", 'a', encoding="utf-8") as file_v_9:
        #     file_v_9.write("*******************global_step = %d*****************\n" % (global_step))
        #     file_v_9.write("v_leakyreLU_512\n")
        #     np.savetxt(file_v_9, np.array(output_data['v_leakyreLU_512'].cpu()), fmt='%.5f')
        # with open("records/v_linear_512-1024.txt", 'a', encoding="utf-8") as file_v_10:
        #     file_v_10.write("*******************global_step = %d*****************\n" % (global_step))
        #     file_v_10.write("v_linear_512-1024\n")
        #     np.savetxt(file_v_10, np.array(output_data['v_linear_512-1024'].cpu()), fmt='%.5f')
        # with open("records/v_leakyreLU_1024.txt", 'a', encoding="utf-8") as file_v_11:
        #     file_v_11.write("*******************global_step = %d*****************\n" % (global_step))
        #     file_v_11.write("v_leakyreLU_1024\n")
        #     np.savetxt(file_v_11, np.array(output_data['v_leakyreLU_1024'].cpu()), fmt='%.5f')
        # with open("records/v_linear_1024-2048.txt", 'a', encoding="utf-8") as file_v_12:
        #     file_v_12.write("*******************global_step = %d*****************\n" % (global_step))
        #     file_v_12.write("v_linear_1024-2048\n")
        #     np.savetxt(file_v_12, np.array(output_data['v_linear_1024-2048'].cpu()), fmt='%.5f')
        # with open("records/v_leakyreLU_2048.txt", 'a', encoding="utf-8") as file_v_13:
        #     file_v_13.write("*******************global_step = %d*****************\n" % (global_step))
        #     file_v_13.write("v_leakyreLU_2048\n")
        #     np.savetxt(file_v_13, np.array(output_data['v_leakyreLU_2048'].cpu()), fmt='%.5f')
        # with open("records/v_linear_2048-1331.txt", 'a', encoding="utf-8") as file_v_14:
        #     file_v_14.write("*******************global_step = %d*****************\n" % (global_step))
        #     file_v_14.write("v_linear_2048-1\n")
        #     np.savetxt(file_v_14, np.array(output_data['v_linear_2048-1'].cpu()), fmt='%.5f')
        #
        # TRY NOT TO MODIFY: record rewards for plotting purposes
        # writer.add_scalar("losses/value_loss", v_loss.item(), global_step)
        # writer.add_scalar("charts/policy_learning_rate", pg_optimizer.param_groups[0]['lr'], global_step)
        # writer.add_scalar("charts/value_learning_rate", v_optimizer.param_groups[0]['lr'], global_step)
        # writer.add_scalar("losses/policy_loss", policy_loss.item(), global_step)
        # writer.add_scalar("losses/entropy", np.mean(entropys), global_step)
        # writer.add_scalar("losses/approx_kl", approx_kl.item(), global_step)
        # if args.kle_stop or args.kle_rollback:
        #     writer.add_scalar("debug/pg_stop_iter", i_epoch_pi, global_step)
        #
        # evaluate no mask
        # average_reward, average_invalid_action_stats = evaluate_with_no_mask()
        # writer.add_scalar("evals/charts/episode_reward", average_reward, global_step)
        # print(f"global_step={global_step}, eval_reward={average_reward}")
        # for key, idx in zip(info['invalid_action_stats'], range(len(info['invalid_action_stats']))):
        #     writer.add_scalar(f"evals/stats/{key}", average_invalid_action_stats[idx], global_step)

    # torch.save(agent_1.pg.state_dict(), 'model/pg_3.pt')
    # torch.save(agent_1.vf.state_dict(), 'model/vf_3.pt')

    env.close()


