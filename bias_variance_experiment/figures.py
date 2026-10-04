#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
画图模块：把四个模块的结论各画成一张 PNG。

四张图分别是：
    01 打靶图（2×2）          —— 弹着点云、靶心、云心、bias 箭头与同心圆环
    02 三色曲线叠图（1×3）      —— 黑=真函数，红=100 个宇宙的 f*，蓝=平均模型 f̄
    03 误差分解三曲线图（1×2）  —— bias²、variance、观察总误差，以及恒等式的堆叠柱验证
    04 模型选择流程对照（2×2）  —— 正确流程 / 错误流程 / 公榜大小扫描 / 报告分数与真分数

版式上的几个统一约定：
    · 所有图都用同一个中文字体（Microsoft YaHei），负号按普通减号渲染；
    · 颜色含义在所有图里保持一致——蓝色 = 平均模型 f̄，红色 = 单个宇宙的 f* 或 bias²，
      绿色 = variance，灰色 = 噪声 σ²（不可降的那部分）；
    · 每张图都会把关键数字直接标在图上，让"看图"和"看 metrics.txt"能当场对上。
"""

from __future__ import annotations
# 打开延迟注解求值

import os
# 用来判断 Windows 自带的中文字体文件是否存在

import matplotlib
# 先拿顶层包，为指定无界面后端做准备

matplotlib.use("Agg")
# 后台无人值守运行，不弹窗口，只把图写成 PNG；这一行必须在 import pyplot 之前

import matplotlib.pyplot as plt
# 画图主接口
import numpy as np
# 数值计算
from matplotlib import font_manager
# 把系统中文字体注册进 matplotlib
from matplotlib.patches import FancyArrowPatch
# 画 bias 箭头（比 annotate 的箭头更好控制起点终点）

from model import NOISE_SIGMA, TARGET_DEGREES, X_MAX, X_MIN, true_function
# 真函数与几个常量；图里的黑线、靶心位置都要用真函数算

COLOR_TRUE = "#111111"
# 真函数 f^ 的颜色：纯黑，和 PPT 上的黑线一致
COLOR_AVG = "#1f4fd8"
# 平均模型 f̄ 的颜色：蓝色，与"弹着点云心"同色
COLOR_SINGLE = "#e8442f"
# 单个宇宙的模型 f* 的颜色：红色
COLOR_BIAS = "#c1121f"
# bias² 的颜色：深红（和红色曲线同色系，表示"偏"这一类误差）
COLOR_VAR = "#2a9d3c"
# variance 的颜色：绿色
COLOR_TOTAL = "#1f4fd8"
# 观察总误差的颜色：蓝色
COLOR_NOISE = "#8a8a8a"
# 噪声 σ² 的颜色：灰色，表示"谁也压不下去的地板"
POINT_ALPHA = 0.55
# 弹着点透明度：200 个点叠在一起，太实会糊成一团
POINT_SIZE = 42
# 弹着点大小


def setup_chinese_font() -> str | None:
    """为 matplotlib 注册一个系统中文字体，让图里的中文正常显示。

    参数：无。
    返回：成功时返回字体族名；找不到任何中文字体时返回 None（图仍会生成，中文可能显示为方框）。
    为什么需要：matplotlib 自带字体不含中文字形，不显式指定就会画出一排"豆腐块"。
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
        # 把字体文件注册进字体管理器，之后就能按名字引用
        name = font_manager.FontProperties(fname=path).get_name()
        # 从字体文件本身读出字体族名，而不是硬编码字符串，避免名字对不上
        plt.rcParams["font.family"] = name
        # 设为全局默认字体
        plt.rcParams["axes.unicode_minus"] = False
        # 让负号按普通减号渲染，否则中文字体下坐标轴的负号会变成方块
        return name
        # 注册成功就结束
    return None
    # 一个中文字体都没找到，返回 None 由调用方提示


def _annotate_box(axis, x: float, y: float, text: str, font_size: int = 11) -> None:
    """在数据坐标 (x, y) 处放一个带浅色底框的注释块。

    参数：
        axis：Axes 对象。
        x / y：注释框左上角的数据坐标。
        text：要写的多行文字。
        font_size：字号。
    返回：无（直接画在 axis 上）。
    """
    axis.text(
        x,
        y,
        text,
        fontsize=font_size,
        va="top",
        ha="left",
        bbox=dict(boxstyle="round,pad=0.45", facecolor="white", edgecolor="#999999", alpha=0.92),
        zorder=6,
    )
    # zorder 给大一点，保证注释压在散点之上不会被盖住


def figure_target_diagram(result: dict, path: str) -> str:
    """画 01 打靶图：四个次数各一格，展示"紧而偏"与"散而中"。

    参数：
        result：run_parallel_universes 的返回值。
        path：输出 PNG 的路径。
    返回：写盘路径。

    每格里画了什么：
        · 200 个浅蓝点 = 200 个宇宙的弹着点 (f*_i(200), f*_i(500))；
        · 深蓝实心大点 = 云心（也就是 f̄ 在两个锚点上的取值）；
        · 红色十字 + 同心圆环 = 靶心 (f^(200), f^(500)) 与距离刻度；
        · 红色箭头 = bias：云心到靶心的距离；
        · 左下角注释框 = bias 距离、variance 半径、两者的比值。
    四个格子故意用同一套坐标范围和同一套圆环刻度，这样"点云的胖瘦"和"离靶心的远近"
    才能横向比较——如果每格各自缩放，看起来会全都差不多，反而看不出规律。
    """
    degrees = list(TARGET_DEGREES)
    # 要画的四个次数：1、2、3、5
    target = np.array([float(true_function(200.0)), float(true_function(500.0))])
    # 靶心：真函数在 x_a=200 与 x_b=500 处的取值（无噪声）
    clouds = {}
    # 每个次数的弹着点矩阵
    half_width = 0.0
    # 所有格子共用的坐标半径，先取 0 再逐格放大
    for degree in degrees:
        # 逐次数取弹着点
        cloud = np.column_stack([result["anchor_a"][degree], result["anchor_b"][degree]])
        # 两列拼起来就是 (f*(200), f*(500)) 的二维点
        clouds[degree] = cloud
        # 存起来
        spread = np.linalg.norm(cloud - target, axis=1).max()
        # 这一格里离靶心最远的那个弹着点有多远
        half_width = max(half_width, float(spread))
        # 取所有格子的最大值
    half_width *= 1.18
    # 留 18% 的空白边距，免得最外面的点贴着边框

    figure, axes = plt.subplots(2, 2, figsize=(13.0, 11.6), dpi=140)
    # 2×2 四宫格
    figure.suptitle(
        f"打靶图：{result['n_universes']} 个平行宇宙、每个宇宙 {result['n_train']} 个训练点\n"
        "弹着点 = 各宇宙模型的预测对 $(f^*(200), f^*(500))$；靶心 = 真值 $(\\hat{f}(200), \\hat{f}(500))$",
        fontsize=15,
        y=0.985,
    )
    # 总标题写清每个点代表什么，避免读者把点误解成"某个测试样本"

    for index, degree in enumerate(degrees):
        # 逐格画
        axis = axes[index // 2][index % 2]
        # 2×2 的下标换算
        cloud = clouds[degree]
        # 这一格的弹着点
        center = cloud.mean(axis=0)
        # 云心 = 平均模型在两个锚点上的取值
        bias_distance = float(np.linalg.norm(center - target))
        # bias 距离
        radius = result["anchor_variance_radius"][degree]
        # variance 半径（均方根）
        mean_distance = result["anchor_mean_distance"][degree]
        # 到云心的平均距离（另一种口径）

        for fraction in (0.25, 0.5, 0.75, 1.0):
            # 同心圆环：不管点云多大，都用相对半径画四个环，保证任何一格都看得见刻度
            ring_radius = half_width * fraction
            ring = plt.Circle(target, ring_radius, fill=False, color="#b8b8b8", linewidth=1.0, linestyle="--", zorder=1)
            # 每个环都是一个不填充的圆
            axis.add_patch(ring)
            # 加到坐标轴上
            axis.text(
                target[0] + ring_radius * 0.7071,
                target[1] + ring_radius * 0.7071,
                f"{ring_radius:.0f}",
                fontsize=8.5,
                color="#8f8f8f",
                ha="left",
                va="bottom",
                zorder=2,
            )
            # 在 45° 方向上标出这个环的距离数值，作为一把"尺子"
        axis.scatter(
            cloud[:, 0],
            cloud[:, 1],
            s=POINT_SIZE,
            color="#4a7de0",
            alpha=POINT_ALPHA,
            edgecolor="white",
            linewidth=0.4,
            zorder=3,
        )
        # 200 个弹着点
        axis.scatter(
            [center[0]], [center[1]], s=190, color=COLOR_AVG, edgecolor="black", linewidth=1.4, zorder=5, label="云心 $\\bar{f}$（平均模型）"
        )
        # 云心：一个大蓝点
        axis.scatter([target[0]], [target[1]], s=200, marker="X", color=COLOR_BIAS, edgecolor="black", linewidth=1.0, zorder=6, label="靶心 $\\hat{f}$（真函数）")
        # 靶心：一个红色 X
        arrow = FancyArrowPatch(
            center,
            target,
            arrowstyle="-|>",
            mutation_scale=17,
            color=COLOR_BIAS,
            linewidth=2.0,
            zorder=4,
        )
        # 从云心指向靶心的箭头，长度就是 bias 距离
        axis.add_patch(arrow)
        # 加到图上
        axis.set_xlim(target[0] - half_width, target[0] + half_width)
        # 四格共用同样的横轴范围
        axis.set_ylim(target[1] - half_width, target[1] + half_width)
        # 四格共用同样的纵轴范围
        axis.set_aspect("equal", adjustable="box")
        # 等比例坐标：两个锚点的单位都是 y 的值，必须 1:1 才不会被拉扁
        axis.grid(alpha=0.18, linestyle=":")
        # 淡网格帮助定位
        ratio = bias_distance / radius if radius > 0 else float("inf")
        # "偏"与"散"的比值：小于 1 说明散主导，大于 1 说明偏主导
        axis.set_title(
            f"{degree} 次多项式："
            + ("点云紧、整体偏离靶心（偏主导）" if ratio > 1.0 else "点云散、中心压在靶心（散主导）"),
            fontsize=13.5,
        )
        # 小标题点出这一格的定性结论，判据是 bias 距离与 variance 半径的比值，不写死次数
        if index // 2 == 1:
            # 最后一行才写横轴标签，省得中间两格挤
            axis.set_xlabel("宇宙 i 的预测 $f^*_i(200)$", fontsize=12)
            # 横轴 = 第一个锚点上的预测
        if index % 2 == 0:
            # 第一列才写纵轴标签
            axis.set_ylabel("宇宙 i 的预测 $f^*_i(500)$", fontsize=12)
            # 纵轴 = 第二个锚点上的预测
        ratio = bias_distance / radius if radius > 0 else float("inf")
        # "偏"与"散"的比值：小于 1 说明散主导，大于 1 说明偏主导
        _annotate_box(
            axis,
            target[0] - half_width * 0.96,
            target[1] - half_width * 0.52,
            f"bias 距离 = {bias_distance:.1f}\n"
            f"variance 半径 = {radius:.1f}\n"
            f"平均距离 = {mean_distance:.1f}\n"
            f"bias / variance = {ratio:.2f}",
        )
        # 注释框放在左下区域，通常不会压到点云
        if index == 0:
            # 只在第一格放图例，避免四份重复
            axis.legend(loc="upper right", fontsize=10, framealpha=0.92)
            # 图例说明三类标记
    # 四格画完

    figure.tight_layout(rect=(0.0, 0.0, 1.0, 0.95))
    # 给总标题留出上边距
    figure.savefig(path, dpi=140, bbox_inches="tight")
    # 写盘
    plt.close(figure)
    # 关掉画布，防止在后台累积内存
    return path
    # 返回路径


def figure_parallel_universes(result: dict, degrees: tuple[int, ...], path: str, curve_count: int = 100) -> str:
    """画 02 三色曲线叠图：黑=真函数，红=若干宇宙的 f*，蓝=平均模型 f̄。

    参数：
        result：run_parallel_universes 的返回值。
        degrees：要画的三个次数（本实验取 1、3、5）。
        path：输出 PNG 路径。
        curve_count：画多少条红曲线（任务要求 100 条）。
    返回：写盘路径。

    看图要点：
        低次（1 次）：红曲线彼此挨得很紧（variance 小），但集体停在黑线下方/上方（bias 大）
                     —— "红紧蓝偏"；
        高次（5 次）：红曲线在两端乱翘（variance 大），但蓝线几乎贴着黑线（bias 小）
                     —— "红乱蓝贴"。
        灰色带是 f^ ± σ，表示"就算模型完全正确，观测点也会落在这个带里"，
        用来提醒：误差里有一部分是噪声，不是模型造成的。
    """
    x_grid = result["x_grid"]
    # 测试网格
    truth = result["truth"]
    # 真函数在网格上的取值
    figure, axes = plt.subplots(1, len(degrees), figsize=(16.5, 5.6), dpi=140, sharey=True)
    # 一行三格，共用纵轴范围才能看出"谁更偏、谁更散"
    figure.suptitle(
        "同一批平行宇宙、不同复杂度：黑=真函数 $\\hat{f}$，红=各宇宙模型 $f^*$，蓝=平均模型 $\\bar{f}$",
        fontsize=15,
        y=1.0,
    )
    # 总标题

    for axis, degree in zip(np.atleast_1d(axes), degrees):
        # 逐格画
        curves = result["predictions"][degree]
        # 形状 (M, G) 的全部预测曲线
        mean_curve = result["f_bar"][degree]
        # 平均曲线 f̄
        axis.fill_between(
            x_grid,
            truth - NOISE_SIGMA,
            truth + NOISE_SIGMA,
            color=COLOR_NOISE,
            alpha=0.20,
            zorder=1,
            label=f"真函数 ± σ（σ={NOISE_SIGMA:.0f}）",
        )
        # 灰色噪声带：f^ 上下各一个 σ
        for index in range(min(curve_count, curves.shape[0])):
            # 逐条画红曲线
            axis.plot(x_grid, curves[index], color=COLOR_SINGLE, alpha=0.16, linewidth=0.9, zorder=2)
            # 透明度压得很低，100 条叠起来才能看出"这一束有多宽"
        axis.plot(x_grid, mean_curve, color=COLOR_AVG, linewidth=3.0, zorder=4, label="平均模型 $\\bar{f}$")
        # 蓝线：所有宇宙的平均
        axis.plot(x_grid, truth, color=COLOR_TRUE, linewidth=2.6, zorder=5, label="真函数 $\\hat{f}$")
        # 黑线：真函数
        for anchor in (200.0, 500.0):
            # 打靶图用的两个锚点位置
            axis.axvline(anchor, color="#666666", linestyle=":", linewidth=1.2, zorder=3)
            # 画竖虚线，让读者知道打靶图的点是从这里取的
        axis.set_title(
            f"{degree} 次多项式\n"
            f"bias² = {result['bias2'][degree]:.0f}   variance = {result['variance'][degree]:.0f}",
            fontsize=13,
        )
        # 标题直接带上两个数字，图和指标表当场对上
        axis.set_xlabel("x", fontsize=12)
        # 横轴
        axis.set_xlim(X_MIN, X_MAX)
        # 定义域
        axis.grid(alpha=0.18, linestyle=":")
        # 淡网格
    np.atleast_1d(axes)[0].set_ylabel("y", fontsize=12)
    # 只有最左边那格需要纵轴标签
    np.atleast_1d(axes)[0].set_ylim(-120, 900)
    # 固定纵轴范围：低次的多项式在两端会翘出去，留足空间才不会被裁掉
    np.atleast_1d(axes)[0].legend(loc="upper left", fontsize=10.5, framealpha=0.92)
    # 图例
    figure.tight_layout(rect=(0.0, 0.0, 1.0, 0.94))
    # 留出总标题的位置
    figure.savefig(path, dpi=140, bbox_inches="tight")
    # 写盘
    plt.close(figure)
    # 释放画布
    return path
    # 返回路径


def figure_error_decomposition(result: dict, rows: list[dict], path: str) -> str:
    """画 03 误差分解图：左边三曲线，右边恒等式堆叠柱。

    参数：
        result：run_parallel_universes 的返回值（本函数只用它拿次数）。
        rows：identity_rows 的输出，含每个次数的 bias²、variance、σ²、观察总误差与残差。
        path：输出 PNG 路径。
    返回：写盘路径。

    左图看趋势：红（bias²）随次数下降，绿（variance）随次数上升，
                蓝（观察总误差 = 红 + 绿 + 灰）呈现先降后升的 U 形；
                灰虚线是 σ² = 2500，它是"地板"，蓝色永远不可能低于它。
    右图看数值：把 σ²、variance、bias² 叠成一根柱子，柱顶就是恒等式算出来的总误差；
                黑菱形是实际观测到的总误差，两者几乎重合（残差标在柱顶）。
    """
    degrees = [row["degree"] for row in rows]
    # 次数列表，当横轴
    bias2 = np.array([row["bias2"] for row in rows])
    # 每个次数的 bias²
    variance = np.array([row["variance"] for row in rows])
    # 每个次数的 variance
    observed = np.array([row["observed"] for row in rows])
    # 每个次数的观察总误差
    sigma2 = float(rows[0]["sigma2"])
    # 噪声方差，所有次数都一样

    figure, (axis_left, axis_right) = plt.subplots(1, 2, figsize=(15.0, 5.8), dpi=140)
    # 左右两格
    figure.suptitle(
        f"误差分解：总误差 = bias² + variance + σ²（{result['n_universes']} 个宇宙 × 独立测试网格 {result['x_grid'].size} 点）",
        fontsize=15,
        y=1.0,
    )
    # 总标题写清样本量

    axis_left.plot(degrees, bias2, marker="o", markersize=9, color=COLOR_BIAS, linewidth=2.6, label="bias²（偏）")
    # 红：bias² 随次数下降
    axis_left.plot(degrees, variance, marker="s", markersize=9, color=COLOR_VAR, linewidth=2.6, label="variance（散）")
    # 绿：variance 随次数上升
    axis_left.plot(degrees, observed, marker="D", markersize=9, color=COLOR_TOTAL, linewidth=2.8, label="观察到的总误差")
    # 蓝：实际测到的总误差
    axis_left.plot(
        degrees,
        bias2 + variance + sigma2,
        marker="x",
        markersize=9,
        color="#222222",
        linewidth=1.8,
        linestyle="--",
        label="bias² + variance + σ²",
    )
    # 黑虚线：用恒等式算出来的总误差，用来验证它和蓝线重合
    axis_left.axhline(sigma2, color=COLOR_NOISE, linestyle=":", linewidth=2.2)
    # 灰虚线：σ² 地板
    axis_left.text(1.05, sigma2 + 180, f"σ² = {sigma2:.0f}（噪声地板，压不下去）", fontsize=11, color="#5f5f5f")
    # 标注这条地板线
    best_index = int(np.argmin(observed))
    # 总误差最低的那个点
    axis_left.annotate(
        f"最低点：{degrees[best_index]} 次\n总误差 = {observed[best_index]:.0f}",
        xy=(degrees[best_index], observed[best_index]),
        xytext=(degrees[best_index] + 0.55, observed[best_index] + 1150),
        fontsize=11,
        arrowprops=dict(arrowstyle="->", color="#333333"),
    )
    # 标出甜区
    for x, y in zip(degrees, bias2):
        # 把每个 bias² 的数值写在点旁边，否则低次的点挤在一起看不出差多少
        axis_left.annotate(f"{y:.0f}", (x, y), textcoords="offset points", xytext=(0, -18), ha="center", fontsize=9.5, color=COLOR_BIAS)
        # bias² 数值
    for x, y in zip(degrees, variance):
        # variance 的数值也标出来
        axis_left.annotate(f"{y:.0f}", (x, y), textcoords="offset points", xytext=(0, 10), ha="center", fontsize=9.5, color=COLOR_VAR)
        # variance 数值
    axis_left.set_xticks(degrees)
    # 横轴刻度就是次数
    axis_left.set_xlabel("多项式次数 d（模型复杂度）", fontsize=12)
    # 横轴
    axis_left.set_ylabel("误差（y 的平方）", fontsize=12)
    # 纵轴
    axis_left.set_ylim(0, float(observed.max()) * 1.22)
    # 纵轴范围留点空
    axis_left.grid(alpha=0.2, linestyle=":")
    # 网格
    axis_left.legend(fontsize=10.5, loc="upper center")
    # 图例
    axis_left.set_title("三曲线：红降、绿升、蓝先降后升", fontsize=13.5)
    # 左图标题

    bar_width = 0.55
    # 柱宽
    axis_right.bar(degrees, [sigma2] * len(degrees), bar_width, color=COLOR_NOISE, alpha=0.5, label="σ²（噪声）")
    # 底段：σ²
    axis_right.bar(degrees, variance, bar_width, bottom=[sigma2] * len(degrees), color=COLOR_VAR, alpha=0.85, label="variance")
    # 中段：variance
    axis_right.bar(
        degrees,
        bias2,
        bar_width,
        bottom=sigma2 + variance,
        color=COLOR_BIAS,
        alpha=0.9,
        label="bias²",
    )
    # 顶段：bias²
    axis_right.plot(
        degrees,
        observed,
        linestyle="none",
        marker="D",
        markersize=11,
        color="#111111",
        markeredgecolor="white",
        markeredgewidth=1.4,
        label="实际观测到的总误差",
    )
    # 黑菱形：实际值，用来和柱顶比较
    for row in rows:
        # 逐次数标注相对残差
        axis_right.annotate(
            f"残差 {row['relative'] * 100:+.2f}%",
            (row["degree"], row["rhs"]),
            textcoords="offset points",
            xytext=(0, 16),
            ha="center",
            fontsize=9.5,
            color="#333333",
        )
        # 柱顶写残差百分比
    axis_right.set_xticks(degrees)
    # 横轴刻度
    axis_right.set_xlabel("多项式次数 d", fontsize=12)
    # 横轴
    axis_right.set_ylabel("误差（堆叠高度 = 恒等式右边）", fontsize=12)
    # 纵轴
    axis_right.set_ylim(0, float(observed.max()) * 1.22)
    # 与左图同高，方便对照
    axis_right.grid(alpha=0.2, axis="y", linestyle=":")
    # 只留横向网格
    axis_right.legend(fontsize=10.5, loc="upper center")
    # 图例
    axis_right.set_title("恒等式验证：柱顶（bias²+variance+σ²）与黑点（实测）重合", fontsize=13.5)
    # 右图标题

    figure.tight_layout(rect=(0.0, 0.0, 1.0, 0.94))
    # 留出总标题
    figure.savefig(path, dpi=140, bbox_inches="tight")
    # 写盘
    plt.close(figure)
    # 释放画布
    return path
    # 返回路径


def figure_model_selection(selection: dict, board: dict, homogeneous: dict, training_rows: list[dict], path: str) -> str:
    """画 04 模型选择对照图：四格分别对应"正确流程""乐观偏差来源""翻车现场""报告 vs 真实"。

    参数：
        selection：repeated_selection_experiment 的返回值。
        board：board_size_sweep 的返回值。
        homogeneous：homogeneous_group_experiment 的返回值。
        training_rows：training_size_sweep 的返回值。
        path：输出 PNG 路径。
    返回：写盘路径。

    四格：
        (a) 主设定下，验证集误差曲线 vs private 榜真实误差曲线，并标出验证集选中的次数；
        (b) 乐观偏差随公榜大小变化（横轴对数），外加"候选数扩到 30 个"那一个点；
        (c) 公榜选的模型：横轴是公榜分数、纵轴是 private 榜分数，点几乎全在对角线上方；
        (d) 三种选法的"报告分数"与"真实分数"成对柱状图，外加"理论最优"参考线。
    """
    degrees = list(selection["degrees"])
    # 候选次数
    figure, axes = plt.subplots(2, 2, figsize=(15.5, 11.6), dpi=140)
    # 2×2 四格
    figure.suptitle(
        f"模型选择流程对照：训练集 {int(board['n_train'])} 点、公榜 50 点、private 榜 5000 点、"
        f"每种流程重复 {selection['n_repeats']} 次",
        fontsize=15,
        y=0.985,
    )
    # 总标题写清设定，避免读者把不同设定的数字混着看

    axis = axes[0][0]
    # (a) 主设定：验证集曲线 vs 真实曲线
    axis.plot(
        degrees,
        selection["per_degree_val"],
        marker="o",
        markersize=9,
        linewidth=2.6,
        color="#e07b00",
        label="验证集误差（1970 点，正确流程用它选）",
    )
    # 正确流程看的曲线
    axis.plot(
        degrees,
        selection["per_degree_public"],
        marker="^",
        markersize=9,
        linewidth=2.2,
        linestyle="--",
        color="#b06ad0",
        label="公榜误差（50 点，错误流程用它选）",
    )
    # 错误流程看的曲线
    axis.plot(
        degrees,
        selection["per_degree_private"],
        marker="D",
        markersize=9,
        linewidth=2.8,
        color=COLOR_TOTAL,
        label="private 榜误差（5000 点，真值）",
    )
    # 真值曲线
    for offset, (name, color) in enumerate((("holdout", "#e07b00"), ("public", "#b06ad0"))):
        # 把两种选法各自选中的次数用竖线标出来（取众数）
        counts = selection[name]["counts"]
        chosen = degrees[int(np.argmax(counts))]
        axis.axvline(chosen, color=color, linestyle=":", linewidth=2.0, alpha=0.85)
        # 竖直虚线标出这个次数
        axis.text(
            0.02,
            0.30 - 0.075 * offset,
            f"{'正确流程（验证集）' if name == 'holdout' else '错误流程（公榜）'}最常选 {chosen} 次"
            f"（{counts.max() / counts.sum() * 100:.0f}% 的重复）",
            transform=axis.transAxes,
            fontsize=10,
            color=color,
        )
        # 用轴内相对坐标写文字，位置不会随数据范围变化而跑偏
    axis.set_xticks(degrees)
    # 横轴刻度
    axis.set_xlabel("多项式次数 d", fontsize=12)
    # 横轴
    axis.set_ylabel("MSE", fontsize=12)
    # 纵轴
    axis.grid(alpha=0.2, linestyle=":")
    # 网格
    axis.legend(fontsize=10, loc="upper center")
    # 图例
    axis.set_title("(a) 小数据设定：甜区在 3~5 次，验证集与大测试集基本平行", fontsize=13)
    # 标题

    axis = axes[0][1]
    # (b) 乐观偏差 vs 公榜大小
    sizes = board["sizes"]
    # 公榜大小列表
    optimism = [board["optimism"][size] for size in sizes]
    # 对应的乐观偏差
    axis.plot(sizes, optimism, marker="o", markersize=9, linewidth=2.6, color=COLOR_BIAS, label="异质候选（1~5 次多项式）")
    # 主曲线
    axis.axhline(0.0, color="#444444", linestyle="-", linewidth=1.2)
    # 0 参考线：乐观偏差为 0 表示报告分数诚实
    axis.plot(
        [homogeneous["board_size"]],
        [homogeneous["optimism"]],
        marker="*",
        markersize=20,
        color="#7a1fa2",
        linestyle="none",
        label=f"候选扩到 {homogeneous['group_size']} 个（都是 {homogeneous['degree']} 次，只是训练数据不同），公榜 {homogeneous['board_size']} 点",
    )
    # 候选数扩到 30 个之后的那个点单独标出来
    axis.set_xscale("log")
    # 横轴取对数：乐观偏差大致按 1/sqrt(公榜点数) 衰减，对数坐标下更像一条直线
    axis.set_xticks(sizes)
    # 刻度就是扫描的点数
    axis.set_xticklabels([str(size) for size in sizes], rotation=30)
    # 写成整数，不要科学计数法
    axis.set_xlabel("公榜（小测试集）点数", fontsize=12)
    # 横轴
    axis.set_ylabel("乐观偏差 = private 分数 − 报告分数", fontsize=12)
    # 纵轴
    axis.grid(alpha=0.2, linestyle=":")
    # 网格
    axis.legend(fontsize=10, loc="upper right")
    # 图例
    axis.set_title("(b) 公榜越小、偏差越大；公榜够大时乐观偏差趋于 0", fontsize=13)
    # 标题

    axis = axes[1][0]
    # (c) 公榜 vs private 散点
    reported = selection["public"]["reported_all"]
    # 每次重复里公榜选中的模型，其公榜分数
    realized = selection["public"]["realized_all"]
    # 同一个模型的 private 分数
    low = float(min(reported.min(), realized.min()))
    # 画对角线的下界
    high = float(max(reported.max(), realized.max()))
    # 上界
    axis.plot([low, high], [low, high], color="#444444", linewidth=1.6, linestyle="--", label="y = x（报告 = 真实）")
    # 对角线：点落在上面说明报告虚高
    axis.scatter(reported, realized, s=42, color="#b06ad0", alpha=0.65, edgecolor="white", linewidth=0.4, label="一次完整重复")
    # 每次重复一个点
    axis.scatter(
        [reported.mean()],
        [realized.mean()],
        s=220,
        marker="X",
        color="#111111",
        edgecolor="white",
        linewidth=1.2,
        zorder=5,
        label=f"平均：报告 {reported.mean():.0f}，真实 {realized.mean():.0f}",
    )
    # 平均值那个大点
    axis.set_xlabel("公榜（50 点）上看到的分数", fontsize=12)
    # 横轴
    axis.set_ylabel("同一模型在 private 榜（5000 点）上的分数", fontsize=12)
    # 纵轴
    axis.grid(alpha=0.2, linestyle=":")
    # 网格
    axis.legend(fontsize=10, loc="upper left")
    # 图例
    axis.set_title(
        f"(c) 公榜选出的模型：平均乐观偏差 {selection['public']['optimism']:+.0f}（点数越靠对角线上方，越虚）",
        fontsize=13,
    )
    # 标题带数字

    axis = axes[1][1]
    # (d) 报告分数 vs 真实分数
    labels = ["留出法\n（验证集 1970 点）", "3 折交叉验证\n（每折验证 10 点）", "公榜法\n（50 点）"]
    # 三种选法的名字
    keys = ["holdout", "cv3", "public"]
    # 对应的字典键
    positions = np.arange(len(keys))
    # 柱子的横坐标
    reported_values = [selection[key]["mean_reported"] for key in keys]
    # 每种选法平均报出去的分数
    realized_values = [selection[key]["mean_private"] for key in keys]
    # 每种选法选中的模型在 private 榜上的平均分数
    axis.bar(positions - 0.19, reported_values, 0.36, color="#b9c4e8", edgecolor="#3a4a80", label="报告出去的分数")
    # 浅蓝柱：你报出去的
    axis.bar(positions + 0.19, realized_values, 0.36, color="#e8b9b2", edgecolor="#803a30", label="private 榜真实分数")
    # 浅红柱：真正的成绩
    for index, key in enumerate(keys):
        # 逐根柱子标数值
        axis.annotate(
            f"{selection[key]['optimism']:+.0f}",
            (positions[index], max(reported_values[index], realized_values[index])),
            textcoords="offset points",
            xytext=(0, 8),
            ha="center",
            fontsize=11,
            color=COLOR_BIAS,
        )
        # 柱顶写乐观偏差
    axis.axhline(selection["best_possible"], color="#111111", linestyle="--", linewidth=1.6)
    # 参考线：理论最优（每次都挑到最好的候选）
    axis.text(
        len(keys) - 0.45,
        selection["best_possible"] + 15,
        f"理论最优 = {selection['best_possible']:.0f}",
        fontsize=10.5,
        ha="right",
        color="#111111",
    )
    # 参考线标注
    axis.set_xticks(positions)
    # 刻度
    axis.set_xticklabels(labels, fontsize=10.5)
    # 刻度文字
    axis.set_ylabel("MSE", fontsize=12)
    # 纵轴
    axis.grid(alpha=0.2, axis="y", linestyle=":")
    # 横向网格
    axis.legend(fontsize=10.5, loc="upper left")
    # 图例
    axis.set_title("(d) 三种选法：报告分数 vs 真实分数（柱顶数字是乐观偏差）", fontsize=13)
    # 标题

    figure.tight_layout(rect=(0.0, 0.0, 1.0, 0.95))
    # 留出总标题
    figure.savefig(path, dpi=140, bbox_inches="tight")
    # 写盘
    plt.close(figure)
    # 释放画布
    return path
    # 返回路径
