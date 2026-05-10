import pandas as pd
import numpy as np
from scipy.stats import t


# ===== 原函数（无需修改）=====

def compute_final_performance(file_path, algo_name, last_n=20):

    df = pd.read_excel(file_path, engine="openpyxl")

    cols = [col for col in df.columns if col.startswith(algo_name)]

    seed_means = []

    for col in cols:

        last_data = df[col].dropna().tail(last_n)

        if len(last_data) == 0:
            continue

        seed_mean = last_data.mean()
        seed_means.append(seed_mean)

    seed_means = np.array(seed_means)

    # 如果没有数据
    if len(seed_means) == 0:
        return np.nan, np.nan

    mean = seed_means.mean()

    # 如果只有1个seed，无法计算CI
    if len(seed_means) < 2:
        return mean, 0.0

    std = seed_means.std(ddof=1)

    # 如果std为0，说明所有seed结果一样
    if std == 0:
        ci = 0.0
    else:
        ci = t.ppf(0.975, len(seed_means)-1) * std / np.sqrt(len(seed_means))

    return mean, ci



# ===== 新增：批量处理多个Excel =====

# 每个指标对应一个Excel文件

files = {

    "Survival Days":r"D:\experiment\data\compare\survival_day\survival_day.xlsx",

    "Reward_production": r"D:\experiment\data\compare\production1\production.xlsx",

    "Reward_consumption":r"D:\experiment\data\compare\consumption1\consumption.xlsx",

    "bank": r"D:\experiment\data\compare\bank\bank.xlsx",

    "bankruptcy_rate": r"D:\experiment\data\compare\bankruptcy\bankruptcy.xlsx"

}


algorithms = ["TD3", "PPO", "Transformer","tf_seq1","tf_seq5"]


# ===== 批量计算 =====

for metric, file_path in files.items():

    print("\n==========================")

    print(metric)

    print("==========================")

    for algo in algorithms:

        mean, ci = compute_final_performance(file_path, algo)

        print(f"{algo}: {mean:.2f} ± {ci:.2f}")