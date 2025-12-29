import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os


def plot_avg_survival_days(excel_path, save_dir=None, filename='bankruptcy_comparison.svg', sheet_name=0):
    """
    绘制TD3, MLP, Transformer三种算法的平均生存天数及方差

    Parameters:
    -----------
    excel_path : str
        Excel文件路径
    save_dir : str
        保存图片的文件夹路径，默认为None（当前目录）
    filename : str
        保存的文件名，默认为'survival_days_comparison.png'
    sheet_name : str or int
        工作表名称或索引，默认为0
    """
    # 读取Excel文件
    df = pd.read_excel(excel_path, sheet_name=sheet_name)

    # 定义各算法对应的列名
    td3_cols = [
        'td_100_5-eval/bankruptcy_rate_step',
        'td_100_4-eval/bankruptcy_rate_step',
        'td3_100_3-eval/bankruptcy_rate_step',
        'td3_100_2-eval/bankruptcy_rate_step',
        'td3_100_1-eval/bankruptcy_rate_step'
    ]

    mlp_cols = [
        'PPO_MLP_seed891_100-eval/bankruptcy_rate_step',
        'PPO_MLP_seed981_100-eval/bankruptcy_rate_step',
        'PPO_MLP_seed936_100-eval/bankruptcy_rate_step'
    ]

    transformer_cols = [
        'ppo_tf_seed891_100-eval/bankruptcy_rate_step',
        'ppo_tf_seed981_100-eval/bankruptcy_rate_step',
        'ppo_tf_seed936_100-eval/bankruptcy_rate_step'
    ]

    # 获取step列
    steps = df['step'].values

    # 计算各算法的均值和标准差
    td3_mean = df[td3_cols].mean(axis=1)
    td3_std = df[td3_cols].std(axis=1)

    mlp_mean = df[mlp_cols].mean(axis=1)
    mlp_std = df[mlp_cols].std(axis=1)

    tf_mean = df[transformer_cols].mean(axis=1)
    tf_std = df[transformer_cols].std(axis=1)

    # 创建图形
    plt.figure(figsize=(12, 8))

    # 绘制TD3 (5次实验)
    plt.plot(steps, td3_mean, label='TD3 (n=5)', color='C1', linewidth=2)
    plt.fill_between(steps, td3_mean - td3_std, td3_mean + td3_std,
                     color='C1', alpha=0.2)

    # 绘制MLP (3次实验)
    plt.plot(steps, mlp_mean, label='PPO_MLP (n=3)', color='C0', linewidth=2)
    plt.fill_between(steps, mlp_mean - mlp_std, mlp_mean + mlp_std,
                     color='C0', alpha=0.2)

    # 绘制Transformer (3次实验)
    plt.plot(steps, tf_mean, label='PPO_Transformer (n=3)', color='C2', linewidth=2)
    plt.fill_between(steps, tf_mean - tf_std, tf_mean + tf_std,
                     color='C2', alpha=0.2)

    # 设置图形属性
    plt.xlabel('Training Steps', fontsize=14)
    plt.ylabel('Average bankruptcy_rate', fontsize=14)
    plt.title('Comparison of Average bankruptcy_rate across Algorithms', fontsize=16)
    plt.legend(loc='best', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()

    # 处理保存路径
    if save_dir is not None:
        # 如果文件夹不存在，则创建
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        save_path = os.path.join(save_dir, filename)
    else:
        save_path = filename

    # 保存图片
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"图片已保存至: {os.path.abspath(save_path)}")

    plt.show()

    return plt.gcf()


# 使用示例
if __name__ == "__main__":
    excel_path = r"D:\experiment\data\compare\bankruptcy\bankruptcy.xlsx"

    # 方式3: 使用完整路径
    plot_avg_survival_days(excel_path, save_dir="D:\experiment\image\compare")
