import pandas as pd
import matplotlib.pyplot as plt
import wandb

# 安装必要的库
# pip install pandas matplotlib wandb

# 初始化wandb
wandb.init(project="reward_analysis",name='lstm_daily_reward')

# 读取CSV文件
file_path = "E:\\dailyreward\\lstm.csv"
try:
    data = pd.read_csv(file_path, encoding='utf-8')
except UnicodeDecodeError:
    try:
        data = pd.read_csv(file_path, encoding='latin1')
    except UnicodeDecodeError:
        data = pd.read_csv(file_path, encoding='cp1252')


# 计算日均累计奖励
raw_data = data[data['Round'] < 3000]
raw_data['Daily_Average_Reward_production1'] = raw_data['production1_reward_business'] / raw_data['production1_days']
raw_data['Daily_Average_Reward_consumption1'] = raw_data['consumption1_reward_business'] / raw_data['production1_days']
raw_data['Daily_Average_Reward_bank'] = raw_data['bank_reward'] / raw_data['production1_days']

# 过滤数据：在3000回合之后，生存天数至少为20天

filtered_data = data[data['Round'] > 3000]
filtered_data = filtered_data[filtered_data['production1_days'] > 20]

# 计算日均累计奖励
filtered_data['Daily_Average_Reward_production1'] = filtered_data['production1_reward_business'] / filtered_data['production1_days']
filtered_data['Daily_Average_Reward_consumption1'] = filtered_data['consumption1_reward_business'] / filtered_data['production1_days']
filtered_data['Daily_Average_Reward_bank'] = filtered_data['bank_reward'] / filtered_data['production1_days']

for index,row in raw_data.iterrows():
    wandb.log({
        'Round': row['Round'],
        'Daily_Average_Reward_production1': row['Daily_Average_Reward_production1'],
        'Daily_Average_Reward_consumption1': row['Daily_Average_Reward_consumption1'],
        'Daily_Average_Reward_bank': row['Daily_Average_Reward_bank']
    })
# 上传每行数据到wandb
for index, row in filtered_data.iterrows():
    wandb.log({
        'Round': row['Round'],
        'Daily_Average_Reward_production1': row['Daily_Average_Reward_production1'],
        'Daily_Average_Reward_consumption1': row['Daily_Average_Reward_consumption1'],
        'Daily_Average_Reward_bank': row['Daily_Average_Reward_bank']
    })

# 关闭wandb会话
wandb.finish()


# # 记录表格数据到wandb
# wandb.log({"reward_table": wandb.Table(dataframe=data)})
#
# # 绘制日均累计奖励图像
# plt.figure(figsize=(10, 6))
# plt.plot(data['Round'], data['Daily_Average_Reward'], label='Daily Average Reward')
# plt.title("Daily Average Reward per Round")
# plt.xlabel("Round")
# plt.ylabel("Daily Average Reward")
# plt.legend()
# plt.grid(True)
#
# # 将图表保存为文件
# plt.savefig("daily_average_reward.png")
#
# # 上传图表到wandb
# wandb.log({"reward_chart": wandb.Image("daily_average_reward.png")})
#
# # 关闭wandb会话
# wandb.finish()
