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
        td3_pattern='td',
        mlp_pattern='mlp',
        tf_pattern='tf',
        ncols=2,  # 每行显示几个子图
        method='median'  # 计算方法：'mean'或'median'
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
        td3_cols = get_columns_by_pattern(df, td3_pattern)
        mlp_cols = get_columns_by_pattern(df, mlp_pattern)
        tf_cols = get_columns_by_pattern(df, tf_pattern)

        # 打印匹配到的列
        print(f"\n{titles[idx]} 匹配到的列：")
        print(f"  TD3 ({len(td3_cols)}列): {td3_cols}")
        print(f"  MLP ({len(mlp_cols)}列): {mlp_cols}")
        print(f"  Transformer ({len(tf_cols)}列): {tf_cols}")

        # 计算均值和标准差
        if method == 'mean':
            td3_solid = df[td3_cols].mean(axis=1)
            td3_std = df[td3_cols].std(axis=1)
            td3_filling_up = td3_solid + td3_std
            td3_filling_down = td3_solid - td3_std

            mlp_solid = df[mlp_cols].mean(axis=1)
            mlp_std = df[mlp_cols].std(axis=1)
            mlp_filling_up = mlp_solid + mlp_std
            mlp_filling_down = mlp_solid - mlp_std

            tf_solid = df[tf_cols].mean(axis=1)
            tf_std = df[tf_cols].std(axis=1)
            tf_filling_up = tf_solid + tf_std
            tf_filling_down = tf_solid - tf_std

        elif method == 'median':
        # 计算中位数和四分位数
            td3_solid= df[td3_cols].median(axis=1)
            td3_filling_down= df[td3_cols].quantile(0.25, axis=1)
            td3_filling_up= df[td3_cols].quantile(0.75, axis=1)

            mlp_solid = df[mlp_cols].median(axis=1)
            mlp_filling_down = df[mlp_cols].quantile(0.25, axis=1)
            mlp_filling_up = df[mlp_cols].quantile(0.75, axis=1)

            tf_solid = df[tf_cols].median(axis=1)
            tf_filling_down = df[tf_cols].quantile(0.25, axis=1)
            tf_filling_up = df[tf_cols].quantile(0.75, axis=1)
        ax = axes[idx]

        # 绘制TD3
        ax.plot(steps, td3_solid, label=f'TD3 (n={len(td3_cols)})', color='C0', linewidth=2)
        if use_std[idx]:
            ax.fill_between(steps, td3_filling_down,td3_filling_up,
                        color='C0', alpha=0.2)

        # 绘制MLP
        ax.plot(steps, mlp_solid, label=f'PPO_MLP (n={len(mlp_cols)})', color='C1', linewidth=2)
        if use_std[idx]:
            ax.fill_between(steps, mlp_filling_down,mlp_filling_up,
                        color='C1', alpha=0.2)

        # 绘制Transformer
        ax.plot(steps, tf_solid, label=f'PPO_Transformer (n={len(tf_cols)})', color='C2', linewidth=2)
        if use_std[idx]:
            ax.fill_between(steps, tf_filling_down,tf_filling_up,
                        color='C2', alpha=0.2)

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
        save_path = os.path.join(save_dir, f"{filename}_{method}.{save_format}")
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
            r"D:\experiment\data\compare\survival_day\300.xlsx",
            # r"D:\experiment\data\compare\bankruptcy\bankruptcy.xlsx",
            r"D:\experiment\data\compare\production1\300.xlsx",
            r"D:\experiment\data\compare\consumption1\300.xlsx",
            r"D:\experiment\data\compare\bank\300.xlsx",
        ],
        save_dir=r"D:\experiment\results",
        filename="300",
        save_format="svg",

        # 每个子图的自定义设置
        titles=[
            "Comparison of Average Survival Days across Algorithms",
            # "Comparison of Average bankruptcy_rate across Algorithms",
            "Comparison of Average production1_Comprehensive_income across Algorithms",
            "Comparison of Average consumption1_Comprehensive_income Algorithms",
            "Comparison of Average bank1_profit across Algorithms",
        ],
        xlabels=[
            "Number of evaluation times",
            # "Number of evaluation times",
            "Number of evaluation times",
            "Number of evaluation times",
            "Number of evaluation times",
        ],
        ylabels=[
            "Average Survival Days",
            # "Average bankruptcy_rate",
            "Average production1_Comprehensive_income",
            "Average consumption1_Comprehensive_income",
            "Average bank1_profit",

        ],
        use_log_scale=[
            False,  # 第一张图不用log
            # False,
            True,
            True,
            True
        ],
        use_std=[
            True,
            True,
            True,
            True
        ],
        # 匹配模式
        td3_pattern='td',
        mlp_pattern='mlp',
        tf_pattern='tf',

        ncols=2,  # 每行2个子图,
        method='mean'  # 绘图方式: 'mean' or 'median'
    )