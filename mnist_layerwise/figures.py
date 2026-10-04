#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
画图模块：四宫格表示图（PCA / t-SNE 两版）、训练曲线、逐层指标变化图。

版式说明：四宫格刻意做成 PPT 原页的样子——
  2×2 排布、每个面板一圈黑色粗边框、白底、去掉坐标刻度（因为这些二维坐标只是投影坐标，
  没有物理单位，标出数字反而会让人误以为横轴有含义）、子图标题用 PPT 上的英文原文
  input / 1-st hidden / 2-nd hidden / 3-rd hidden，并在每个颜色团的中心标上数字。
面板左下角会额外压一行小字，写清这一层的维度、轮廓系数和 1-NN 准确率，
让图和量化指标表能当场对上——看图的人不必翻到 README 才知道数字是否一致。
"""

from __future__ import annotations
# 打开延迟注解求值，类型标注里可以直接写 np.ndarray

import os
# 用来判断 Windows 自带的中文字体文件是否存在于本机

import matplotlib
# 先拿到 matplotlib 顶层包，为下面强制指定无界面后端做准备

matplotlib.use("Agg")
# 强制使用 Agg 后端：本脚本在后台无人值守运行，不需要弹出窗口，只要能把图存成 PNG
# 这一行必须在 import pyplot 之前执行，否则后端已经被选定，再设就晚了

import matplotlib.pyplot as plt
# 画图主接口
import numpy as np
# 数值计算
from matplotlib import font_manager
# 字体管理：把系统中文字体注册进 matplotlib
from matplotlib.lines import Line2D
# 用来手工构造图例里的彩色圆点（散点图用的是数值映射配色，不方便直接生成图例）

from analysis import class_centers_2d
# 复用 analysis.py 里“算每个数字类在二维平面的中心”的工具，避免同一段逻辑写两遍

COLOR_MAP = "tab10"
# 十类颜色表：tab10 恰好有 10 种区分度很高的颜色，与 0~9 十个数字一一对应
CLASS_COUNT = 10
# 手写数字类别数
POINT_SIZE = 16
# 散点大小；1797 个点用 16 左右既不糊成一坨，也能看出团的形状
POINT_ALPHA = 0.75
# 点的透明度；不透明的话重叠区域看不出密度，太透明又会显得脏
FRAME_WIDTH = 2.6
# 面板黑色边框的线宽，PPT 那种“黑框四宫格”的感觉主要靠它
TITLE_SIZE = 17
# 子图标题字号，PPT 上标题很显眼，这里也放大一些


def setup_chinese_font() -> str | None:
    """为 matplotlib 注册一个系统中文字体，让图里的中文标题正常显示。

    参数：无。
    返回：成功时返回字体族名；找不到任何中文字体时返回 None（图仍会生成，中文可能显示为方框）。
    为什么需要：matplotlib 自带字体不含中文字形，不显式指定就会画出一排“豆腐块”。
    """
    candidates = [
        r"C:\Windows\Fonts\msyh.ttc",
        r"C:\Windows\Fonts\simhei.ttf",
        r"C:\Windows\Fonts\Deng.ttf",
        r"C:\Windows\Fonts\simsun.ttc",
    ]
    # 依次尝试微软雅黑、黑体、等线、宋体这几个 Windows 自带中文字体
    for path in candidates:
        # 逐个检查文件是否存在
        if not os.path.exists(path):
            # 不存在就换下一个候选
            continue
        font_manager.fontManager.addfont(path)
        # 把字体文件注册进 matplotlib 的字体管理器，之后就能按名字引用
        name = font_manager.FontProperties(fname=path).get_name()
        # 从字体文件自身读出字体族名，而不是硬编码字符串，避免名字对不上（雅黑的正名是 Microsoft YaHei）
        plt.rcParams["font.family"] = name
        # 设为全局默认字体
        plt.rcParams["axes.unicode_minus"] = False
        # 让负号按普通减号渲染，否则中文字体下坐标轴的负号会变成方块
        return name
        # 注册成功就结束
    return None
    # 一个中文字体都没找到，返回 None 由调用方提示


def _draw_panel(axis, coordinates: np.ndarray, labels: np.ndarray, title: str, note: str) -> None:
    """在给定的坐标轴上画一个面板：散点 + 类中心数字 + PPT 风格黑框。

    参数：
        axis：matplotlib 的 Axes 对象。
        coordinates：形状 (n, 2) 的二维投影坐标。
        labels：形状 (n,) 的真实数字标签。
        title：面板标题（input / 1-st hidden / ...）。
        note：压在面板左下角的指标小字，说明这一层的维度与两个量化指标。
    返回：无（只画在当前 axis 上）。
    """
    scatter = axis.scatter(
        coordinates[:, 0],
        coordinates[:, 1],
        c=labels,
        cmap=COLOR_MAP,
        vmin=0,
        vmax=CLASS_COUNT - 1,
        s=POINT_SIZE,
        alpha=POINT_ALPHA,
        linewidths=0,
    )
    # 用 10 色离散色表按真实数字标签上色；vmin/vmax 固定成 0/9，
    # 保证四个面板里“同一个数字永远是同一种颜色”，否则各面板自动缩放颜色范围会让对比失去意义
    digits, centers = class_centers_2d(coordinates, labels)
    # 算出每个数字类在二维投影里的中心位置（该类所有点坐标的平均），并拿到实际出现的类别编号
    axis.scatter(
        centers[:, 0],
        centers[:, 1],
        s=430,
        facecolors="white",
        edgecolors="black",
        linewidths=1.6,
        zorder=3,
    )
    # 在类中心先铺一个白色实心圆：直接写字会被底下的散点盖住，白色圆盘提供一个干净底衬
    for digit, (x_position, y_position) in zip(digits, centers):
        # 逐个类中心标注数字
        axis.text(
            x_position,
            y_position,
            str(digit),
            fontsize=15,
            fontweight="bold",
            ha="center",
            va="center",
            color="black",
            zorder=4,
        )
        # 数字写在白色圆盘正中央，zorder 最大保证压在最上层
    axis.set_title(title, fontsize=TITLE_SIZE, fontweight="bold", pad=10)
    # 面板标题就是 PPT 上的原文层名
    axis.set_xticks([])
    # 去掉横轴刻度：这两个坐标只是投影坐标，没有物理单位
    axis.set_yticks([])
    # 去掉纵轴刻度，理由同上
    for spine in axis.spines.values():
        # 遍历四条边框
        spine.set_linewidth(FRAME_WIDTH)
        # 加粗到 PPT 那种明显黑框的效果
        spine.set_color("black")
        # 边框用纯黑
    axis.text(
        0.025,
        0.025,
        note,
        transform=axis.transAxes,
        fontsize=9.5,
        ha="left",
        va="bottom",
        color="#222222",
        bbox={"boxstyle": "round,pad=0.32", "facecolor": "white", "edgecolor": "#999999", "alpha": 0.9},
        zorder=5,
    )
    # 左下角压一行指标小字：白底圆角框保证在密集散点上仍然可读
    return scatter
    # 返回散点对象（调用方目前不用，保留是为了将来想加 colorbar 时方便）


def figure_representation_grid(
    panels: list[dict],
    out_path: str,
    suptitle: str,
    subtitle: str,
) -> None:
    """画四宫格主图：同一批样本在四种表示下的二维投影。

    参数：
        panels：长度 4 的列表，每项是
                {"title": 层名, "coordinates": (n,2) 坐标, "labels": (n,) 真实标签, "note": 指标小字}。
        out_path：图片保存路径。
        suptitle：总标题（含图号）。
        subtitle：副标题，说明降维方法与结论。
    返回：无（图片直接写盘）。
    """
    if len(panels) != 4:
        # 四宫格必须是 4 个面板，多一个少一个都说明调用方搞错了层数
        raise ValueError(f"四宫格需要 4 个面板，收到 {len(panels)} 个")
        # 直接报错，不静默裁剪
    figure, axes = plt.subplots(2, 2, figsize=(12.6, 11.0))
    # 2×2 布局；画布比例接近 PPT 页面，方便和原页对照
    flat_axes = axes.ravel()
    # 把 2×2 的轴数组拉平成一维，方便按下标遍历
    for axis, panel in zip(flat_axes, panels):
        # 逐个面板绘制
        _draw_panel(axis, panel["coordinates"], panel["labels"], panel["title"], panel["note"])
        # 把坐标、标签、标题、指标小字交给通用面板绘制函数
    handles = [
        Line2D([], [], marker="o", linestyle="none", markersize=8, color=plt.get_cmap(COLOR_MAP)(index / 9.0))
        for index in range(CLASS_COUNT)
    ]
    # 手工造 10 个彩色圆点作为图例项；用 index/9 与散点的 vmin=0/vmax=9 配色保持完全一致
    figure.legend(
        handles,
        [str(index) for index in range(CLASS_COUNT)],
        loc="lower center",
        ncol=CLASS_COUNT,
        frameon=False,
        fontsize=12,
        title="颜色 = 真实数字标签",
        title_fontsize=11,
        bbox_to_anchor=(0.5, 0.005),
    )
    # 底部排一排 0~9 的色标图例，替代四张图各配一条 colorbar，更省空间也更像 PPT
    figure.suptitle(f"{suptitle}\n{subtitle}", fontsize=13.5, y=0.985)
    # 总标题 + 副标题
    figure.tight_layout(rect=(0, 0.045, 1, 0.955))
    # 自动布局，并给底部图例和顶部总标题留出空白
    figure.savefig(out_path, dpi=150, bbox_inches="tight")
    # 以 150 dpi 存成 PNG
    plt.close(figure)
    # 关闭画布，释放内存（批量出图时必须做，否则几十张图会把内存吃满）


def figure_training_curve(history: list, out_path: str) -> None:
    """画训练曲线：左边损失、右边准确率，训练集与测试集各一条。

    参数：
        history：model.EpochLog 的列表。
        out_path：图片保存路径。
    返回：无。
    """
    epochs = [record.epoch for record in history]
    # 横轴：epoch 序号
    figure, axes = plt.subplots(1, 2, figsize=(12.8, 4.8))
    # 左右两个面板
    axes[0].plot(epochs, [record.train_loss for record in history], color="#1f77b4", linewidth=1.8, label="训练集")
    # 左图的训练损失曲线
    axes[0].plot(epochs, [record.eval_loss for record in history], color="#d62728", linewidth=1.8, label="测试集")
    # 左图的测试损失曲线
    axes[0].set_xlabel("epoch")
    # 横轴标签
    axes[0].set_ylabel("交叉熵损失")
    # 纵轴标签
    axes[0].set_title("损失：网络在多少个 epoch 内收敛")
    # 标题
    axes[0].grid(alpha=0.25)
    # 淡网格便于读数
    axes[0].legend()
    # 图例
    axes[1].plot(epochs, [record.train_accuracy for record in history], color="#1f77b4", linewidth=1.8, label="训练集")
    # 右图的训练准确率曲线
    axes[1].plot(epochs, [record.eval_accuracy for record in history], color="#d62728", linewidth=1.8, label="测试集")
    # 右图的测试准确率曲线
    axes[1].axhline(0.95, color="#7f7f7f", linestyle="--", linewidth=1.2)
    # 画出 95% 这条验收线，一眼就能看出有没有达标
    axes[1].text(epochs[-1], 0.952, "验收线 95%", fontsize=9, color="#7f7f7f", ha="right", va="bottom")
    # 在验收线右端标注它的含义
    axes[1].set_xlabel("epoch")
    # 横轴标签
    axes[1].set_ylabel("分类准确率")
    # 纵轴标签
    axes[1].set_title("准确率：训练集与测试集贴在一起说明没有过拟合")
    # 标题
    axes[1].grid(alpha=0.25)
    # 淡网格
    axes[1].legend(loc="lower right")
    # 图例放右下角，避免压住上升的曲线
    figure.suptitle("图 3  3 隐藏层 MLP 的训练过程（64->128->64->32->10，Adam，交叉熵）", fontsize=12.5)
    # 总标题写明网络结构与优化配置
    figure.tight_layout(rect=(0, 0, 1, 0.93))
    # 自动布局并给总标题留空间
    figure.savefig(out_path, dpi=150, bbox_inches="tight")
    # 保存
    plt.close(figure)
    # 关闭画布


def figure_layer_metrics(layer_names: list[str], rows: list[dict], out_path: str) -> None:
    """画“指标随层数变化”图：轮廓系数与 1-NN 准确率各自随层数怎么走。

    参数：
        layer_names：四层的名字，作为横轴刻度。
        rows：每层一个字典，需含 "silhouette"、"one_nn_loo"、"one_nn_test" 三个键。
        out_path：图片保存路径。
    返回：无。

    为什么值得单独画一张：README 里那张指标表是数字，这张图把“随层数单调上升”这件事
    画成两条折线，递进到底成不成立、哪一层有起伏，一眼就能看出来。
    """
    positions = np.arange(len(layer_names))
    # 四层的横坐标位置
    silhouette_values = [row["silhouette"] for row in rows]
    # 各层轮廓系数
    loo_values = [row["one_nn_loo"] for row in rows]
    # 各层留一法 1-NN 准确率
    test_values = [row["one_nn_test"] for row in rows]
    # 各层“训练集当参考”的 1-NN 测试准确率
    figure, axes = plt.subplots(1, 2, figsize=(12.8, 4.9))
    # 左右两个面板
    axes[0].plot(positions, silhouette_values, marker="o", markersize=9, linewidth=2.0, color="#2ca02c")
    # 左图：轮廓系数随层数变化
    for position, value in zip(positions, silhouette_values):
        # 逐点标数值，省得读者拿尺子量
        axes[0].annotate(f"{value:.3f}", (position, value), textcoords="offset points", xytext=(0, 9), ha="center", fontsize=10)
        # 标在该点正上方
    axes[0].set_xticks(positions, layer_names)
    # 横轴刻度写层名
    axes[0].set_ylabel("silhouette score（越大越成团）")
    # 纵轴标签
    axes[0].set_title("轮廓系数：按真实标签算的类内紧致 / 类间分离程度")
    # 标题
    axes[0].grid(alpha=0.25)
    # 淡网格
    axes[1].plot(positions, loo_values, marker="o", markersize=9, linewidth=2.0, color="#1f77b4", label="1-NN 留一法（全部 1797 张）")
    # 右图第一条：留一法 1-NN
    axes[1].plot(positions, test_values, marker="s", markersize=8, linewidth=2.0, color="#d62728", label="1-NN 训练集当参考（测试集 450 张）")
    # 右图第二条：训练集当参考、测试集当考点的 1-NN
    for position, value in zip(positions, loo_values):
        # 标出留一法的数值
        axes[1].annotate(f"{value:.3f}", (position, value), textcoords="offset points", xytext=(0, 9), ha="center", fontsize=9)
        # 标在点正上方
    axes[1].set_xticks(positions, layer_names)
    # 横轴刻度写层名
    axes[1].set_ylabel("1-NN 分类准确率")
    # 纵轴标签
    axes[1].set_ylim(0.85, 1.005)
    # 纵轴从 0.85 起：四层的 1-NN 都在 0.93 以上，从 0 起会把差异压成一条平线
    axes[1].set_title("最近邻分类器在该层表示上的准确率")
    # 标题
    axes[1].grid(alpha=0.25)
    # 淡网格
    axes[1].legend(loc="lower right", fontsize=9)
    # 图例
    figure.suptitle("图 4  量化指标随层数变化：表示越深，越容易被最简单的分类器分开", fontsize=12.5)
    # 总标题点明结论
    figure.tight_layout(rect=(0, 0, 1, 0.93))
    # 自动布局
    figure.savefig(out_path, dpi=150, bbox_inches="tight")
    # 保存
    plt.close(figure)
    # 关闭画布
