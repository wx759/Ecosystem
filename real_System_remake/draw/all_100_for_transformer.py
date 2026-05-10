import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os


def get_columns_by_pattern(df, pattern):
    """根据模式模糊匹配列名"""
    return [col for col in df.columns if pattern.lower() in col.lower()]


def plot_combined_survival_days(
        excel_paths,  # 改为列表，支持多个文件
        save_dir=None,
        filename='combined_comparison',
        save_format='svg',
        titles=None,  # 每个子图的标题列表
        xlabels=None,  # 每个子图的x轴标签列表
        ylabels=None,  # 每个子图的y轴标签列表
        use_log_scale=None,  # 每个子图是否使用log纵坐标列表
        use_std=None,  # 每个子图是否显示标准差列表
        seq1_pattern='td',
        seq3_pattern='mlp',
        seq5_pattern='tf',
        ncols=2  # 每行显示几个子图
):
    """
    将多个实验的结果组合在一张图中

    Parameters:
    -----------
    excel_paths : list
        Excel文件路径列表
    titles : list
        每个子图的标题列表
    xlabels : list
        每个子图的x轴标签列表
    ylabels : list
        每个子图的y轴标签列表
    use_log_scale : list
        每个子图是否使用log纵坐标的布尔值列表
    ncols : int
        每行显示几个子图
    """

    n_plots = len(excel_paths)
    nrows = (n_plots + ncols - 1) // ncols  # 计算需要几行


    # 创建子图
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(8 * ncols, 6 * nrows))

    # 确保 axes 是一维数组（方便索引）
    if n_plots == 1:
        axes = [axes]
    else:
        axes = axes.flatten()

    for idx, excel_path in enumerate(excel_paths):
        df = pd.read_excel(excel_path)
        steps = df['step'].values

        # 使用模糊匹配获取列名
        seq1_cols = get_columns_by_pattern(df, seq1_pattern)
        seq3_cols = get_columns_by_pattern(df, seq3_pattern)
        seq5_cols = get_columns_by_pattern(df, seq5_pattern)

        # 打印匹配到的列
        print(f"\n{titles[idx]} 匹配到的列：")
        print(f"  seq1 ({len(seq1_cols)}列): {seq1_cols}")
        print(f"  seq3 ({len(seq3_cols)}列): {seq3_cols}")
        print(f"  seq5 ({len(seq5_cols)}列): {seq5_cols}")

        # 计算均值和标准差
        seq1_mean = df[seq1_cols].mean(axis=1)
        seq3_mean = df[seq3_cols].mean(axis=1)
        seq5_mean = df[seq5_cols].mean(axis=1)
        from scipy.stats import t

        n_seq1 = len(seq1_cols)
        n_seq3 = len(seq3_cols)
        n_seq5 = len(seq5_cols)

        seq1_std = df[seq1_cols].std(axis=1)
        seq3_std = df[seq3_cols].std(axis=1)
        seq5_std = df[seq5_cols].std(axis=1)

        seq1_ci = t.ppf(0.975, n_seq1 - 1) * seq1_std / np.sqrt(n_seq1)
        seq3_ci = t.ppf(0.975, n_seq3 - 1) * seq3_std / np.sqrt(n_seq3)
        seq5_ci = t.ppf(0.975, n_seq5 - 1) * seq5_std / np.sqrt(n_seq5)

        ax = axes[idx]

        # 绘制TD3
        ax.plot(steps, seq1_mean, label=f'seq1 (n={len(seq1_cols)})', color='C0', linewidth=2)
        if use_std[idx]:
            ax.fill_between(steps, seq1_mean - seq1_ci, seq1_mean + seq1_ci,
                        color='C0', alpha=0.1)

        # 绘制MLP
        ax.plot(steps, seq3_mean, label=f'seq3 (n={len(seq3_cols)})', color='C1', linewidth=2)
        if use_std[idx]:
            ax.fill_between(steps, seq3_mean - seq3_ci, seq3_mean + seq3_ci,
                        color='C1', alpha=0.1
                            )

        # 绘制Transformer
        ax.plot(steps, seq5_mean, label=f'seq5 (n={len(seq5_cols)})', color='C2', linewidth=2)
        if use_std[idx]:
            ax.fill_between(steps, seq5_mean - seq5_ci, seq5_mean + seq5_ci,
                        color='C2', alpha=0.1)

        # 设置log坐标（如果需要）
        if use_log_scale[idx]:
            ax.set_yscale('log')

        # 设置自定义的标签和标题
        ax.set_xlabel(xlabels[idx], fontsize=12)
        ax.set_ylabel(ylabels[idx], fontsize=12)
        ax.set_title(titles[idx], fontsize=14)
        ax.legend(loc='lower right', fontsize=10)
        ax.grid(True, linestyle='--', alpha=0.7)

    # 隐藏多余的子图（如果子图数量不能填满网格）
    for idx in range(n_plots, len(axes)):
        axes[idx].set_visible(False)

    plt.tight_layout()

    # 保存图片
    if save_dir is not None:
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        save_path = os.path.join(save_dir, f"{filename}.{save_format}")
    else:
        save_path = f"{filename}.{save_format}"

    plt.savefig(save_path, dpi=300, bbox_inches='tight', format=save_format)
    print(f"\n图片已保存至: {os.path.abspath(save_path)}")

    plt.show()
    return fig


# 使用示例
if __name__ == "__main__":
    plot_combined_survival_days(
        excel_paths=[
            r"D:\experiment\data\compare\survival_day\survival_day.xlsx",
            r"D:\experiment\data\compare\production1\production.xlsx",
            r"D:\experiment\data\compare\consumption1\consumption.xlsx",
            r"D:\experiment\data\compare\bank\bank.xlsx",
        ],
        save_dir=r"D:\experiment\results",
        filename="序列长度比较",
        save_format="png",

        titles=[
            "(a) Survival Days",
            "(b) Enterprise A Performance",
            "(c) Enterprise B Performance",
            "(d) Bank Performance"
        ],
        xlabels=[
            "Evaluation Index (k)",
            "Evaluation Index (k)",
            "Evaluation Index (k)",
            "Evaluation Index (k)"
        ],
        ylabels=[
            "Days",
            "Performance",
            "Performance",
            "Performance"
        ],
        use_log_scale=[
            False,  # 第一张图不用log
            # False,
            False,
            False,
            False
        ],
        use_std=[
            True,
            True,
            True,
            True
        ],
        # 匹配模式
        seq1_pattern='tf_seq1',
        seq3_pattern='Transformer',
        seq5_pattern='tf_seq5',

        ncols=2  # 每行2个子图
    )