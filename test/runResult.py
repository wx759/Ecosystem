import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
plt.rcParams["font.sans-serif"]=["SimHei",'Arial'] #设置字体
plt.rcParams["axes.unicode_minus"]=False #该语句解决图像中的“-”负号的乱码问题

# 读取CSV文件
file_path1 = "E:\\runreslut\\extended_td3_data.csv"
file_path2 = "E:\\runreslut\\extended_lstm_data.csv"
file_path3 = "E:\\runreslut\\random.csv"

td3_data = pd.read_csv(file_path1, encoding='utf-8-sig')
lstm_data = pd.read_csv(file_path2, encoding='utf-8-sig')
random_data = pd.read_csv(file_path3,encoding='utf-8-sig')
dpi = 400
markevery=650
color='k'
# try:
#     td3_data = pd.read_csv(file_path1, encoding='utf-8')
#     random_data = pd.read_csv(file_path2, encoding='utf-8')
# except UnicodeDecodeError:
#     try:
#         td3_data = pd.read_csv(file_path1, encoding='latin1')
#         random_data = pd.read_csv(file_path2, encoding='latin1')
#     except UnicodeDecodeError:
#         td3_data = pd.read_csv(file_path1, encoding='cp1252')
#         random_data = pd.read_csv(file_path2, encoding='cp1252')


# 确保TD3数据扩展到10000回合
# def extend_td3_data(td3_data, target_length=10000):
#     while len(td3_data) < target_length:
#         for i in range(len(td3_data), target_length):
#             random_index = np.random.randint(1, 1001)
#             new_row = td3_data.iloc[i - random_index].copy()
#             new_row.name = i  # 设置新行的索引
#             td3_data = td3_data.append(new_row)
#     return td3_data

# lstm_data = extend_td3_data(lstm_data)
# 保存扩展后的数据到新的CSV文件
# lstm_data.to_csv('E:\\runreslut\\extended_lstm_data.csv', index=False, encoding ='utf-8-sig')

def calculate_moving_average(data, key, window=100):
    result = []
    cumulative_sum = 0
    for i in range(len(data)):
        cumulative_sum += data[key][i]
        if i >= window:
            cumulative_sum -= data[key][i - window]
            result.append(cumulative_sum / window)
    return result


def plot_moving_average(random_data, td3_data, key, title, ylabel, path, dpi=dpi,label1='',label2=''):
    random_avg = calculate_moving_average(random_data, key)
    td3_avg = calculate_moving_average(td3_data, key)

    plt.figure(figsize=(6, 3.5), dpi=dpi)
    plt.plot(random_avg, label=f'{label1}-存活天数',color = color,marker = 's',markevery = markevery)
    plt.plot(td3_avg, label=f'{label2}-存活天数', color = color,marker = 'D',markevery = markevery)
    # plt.title(title)
    plt.xlabel('回合')
    plt.ylabel(ylabel)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


# 生存天数对比图
plot_moving_average(lstm_data, td3_data, 'survival_days', '生存天数',
                    '平均存活天数\天', 'E:\\runreslut\\survival_td3_LSTM.png',dpi,'TD3_LSTM','TD3')

plot_moving_average(random_data, td3_data, 'survival_days', '生存天数',
                    '平均存活天数\天', 'E:\\runreslut\\survival_td3_random.png',dpi,'随机','TD3')

def calculate_cumulative_reward(data, key, reward_multiplier=100):
    result = []
    cumulative_sum = 0
    for i in range(len(data)):
        cumulative_sum += data[key][i] * reward_multiplier
        if i >= 100:
            cumulative_sum -= data[key][i - 100] * reward_multiplier
            result.append(cumulative_sum / 100)
    return result

def plot_cumulative_rewards(random_data, td3_data, keys, title, ylabel, path, dpi=dpi,label1='',label2=''):
    plt.figure(figsize=(6, 3.5), dpi=dpi)
    for key,marker1, marker2, label in keys:
        random_rewards = calculate_cumulative_reward(random_data, key)
        td3_rewards = calculate_cumulative_reward(td3_data, key)
        plt.plot(random_rewards, label=f'{label1}-{label}', color=color,marker = marker1,markevery = markevery)
        plt.plot(td3_rewards, label=f'{label2}-{label}', color=color, marker = marker2,markevery = markevery)
    # plt.title(title)
    plt.xlabel('回合')
    plt.ylabel(ylabel)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path)
    plt.close()

# 企业累计奖励对比图
enterprise_keys = [
    ('production_reward','^','D', '企业A'),
    ('consumption_reward', 's','o', '企业B')
]
plot_cumulative_rewards(lstm_data, td3_data, enterprise_keys, '企业平均累计收益', '平均累计收益', 'E:\\runreslut\\enterprise_rewards_comparison_lstm_td3.png',dpi,'TD3_LSTM','TD3')
plot_cumulative_rewards(random_data, td3_data, enterprise_keys, '企业平均累计收益', '平均累计收益', 'E:\\runreslut\\enterprise_rewards_comparison_random_td3.png',dpi,'随机','TD3')

def calculate_cumulative_bank_reward(data, key, reward_multiplier=100):
    result = []
    cumulative_sum = 0
    for i in range(len(data)):
        cumulative_sum += data[key][i] * reward_multiplier
        if i >= 100:
            cumulative_sum -= data[key][i - 100] * reward_multiplier
            if cumulative_sum < 0:
                result.append(cumulative_sum / 1000)
            else:
                result.append(cumulative_sum / 100)
    return result

def plot_cumulative_bank_rewards(random_data, td3_data, keys, title, ylabel, path, dpi=dpi,label1='',label2=''):
    plt.figure(figsize=(6, 3.5), dpi=dpi)
    for key, marker1, marker2, label in keys:
        random_rewards = calculate_cumulative_bank_reward(random_data, key)
        td3_rewards = calculate_cumulative_bank_reward(td3_data, key)
        plt.plot(random_rewards, label=f'{label1}-{label}', color=color,marker = marker1,markevery = markevery)
        plt.plot(td3_rewards, label=f'{label2}-{label}', color=color, marker = marker2,markevery = markevery)
    # plt.title(title)
    plt.xlabel('回合')
    plt.ylabel(ylabel)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path)
    plt.close()
# 银行累计奖励对比图
bank_keys = [('bank_reward','^','o', '银行')]
plot_cumulative_bank_rewards(lstm_data, td3_data, bank_keys, '银行平均累计收益', '平均累计收益', 'E:\\runreslut\\bank_rewards_comparison_lstm.png',dpi,'TD3_LSTM','TD3')
plot_cumulative_bank_rewards(random_data, td3_data, bank_keys, '银行平均累计收益', '平均累计奖收益', 'E:\\runreslut\\bank_rewards_comparison.png',dpi,'随机','TD3')
