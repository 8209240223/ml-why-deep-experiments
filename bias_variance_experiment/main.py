#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
主流程：复现李宏毅《Bias and Variance》这一章的四个模块。

一句话概括这个脚本做的事：
    先造一条"先缓后陡"的真函数曲线，再让 200 个平行宇宙各抽一批训练数据、
    各自训出 1~5 次的多项式模型，然后：
      模块一 把每个宇宙在两个固定输入上的预测画成打靶弹着点，看"偏"和"散"；
      模块二 把 100 条模型曲线和它们的平均曲线叠在一起，看"红紧蓝偏"与"红乱蓝贴"；
      模块三 把总误差拆成 bias² + variance + σ²，并验证三条曲线的数值恒等式；
      模块四 对比"用验证集选模型"和"用 50 点 public 榜选模型"，量出乐观偏差。
    最后把所有数字写进 results/metrics.txt，并跑一遍自检。

执行顺序（也就是 main() 里的步骤）：
    1. 注册中文字体，准备 results/ 目录
    2. 跑模块一二三：200 个宇宙 × 5 个次数，得到全部预测曲线与统计量
    3. 画 01 打靶图、02 三色叠图、03 误差分解图
    4. 生成模块四的 private 榜，跑"三种选法对照""公榜大小扫描""候选数扫描""训练规模扫描"
    5. 画 04 模型选择对照图
    6. 写 results/metrics.txt，终端打印摘要，跑自检

用法（Git Bash，在本目录下）：
    /c/msys64/ucrt64/bin/python main.py
    /c/msys64/ucrt64/bin/python main.py --universes 400 --seed 7
"""

from __future__ import annotations
# 打开延迟注解求值，类型标注里可以直接写 np.ndarray

import argparse
# 解析命令行参数，方便换宇宙数、换种子重跑
import os
# 拼路径、建目录
import platform
# 把操作系统版本写进报告（换机器跑时便于排查）
import sys
# 退出码；验收不通过时返回非 0，方便后台任务一眼看出失败
import time
# 记录各阶段耗时
import unicodedata
# 判断字符是不是全角，写等宽文本表格时要用

import numpy as np
# 数值计算
import matplotlib
# 只为在报告里写出版本号

from analysis import (
    DEFAULT_REPEATS,
    DEFAULT_TRAIN_SIZE,
    GRID_POINTS,
    POOL_SIZE,
    PRIVATE_SIZE,
    PUBLIC_SIZE,
    board_size_sweep,
    homogeneous_group_experiment,
    identity_rows,
    leverage_variance_check,
    repeated_selection_experiment,
    run_parallel_universes,
    selection_seed_robustness,
    training_size_sweep,
)
# 模块一二三与模块四的全部计算
from figures import (
    figure_error_decomposition,
    figure_model_selection,
    figure_parallel_universes,
    figure_target_diagram,
    setup_chinese_font,
)
# 四张图的画图入口
from model import (
    ANCHOR_A,
    ANCHOR_B,
    DEGREES,
    NOISE_SIGMA,
    SIGMOID_HEIGHT,
    SIGMOID_MID,
    SIGMOID_SCALE,
    TARGET_DEGREES,
    X_MAX,
    X_MIN,
    sample_dataset,
    true_function,
)
# 真函数参数、模型族、抽样与真值函数

CURVE_COUNT = 3
# 模块二里要画的次数个数（1、3、5）
HOMOGENEOUS_GROUP = 30
# 候选数扫描里用多少个变体（都是同一个次数、只是训练数据不同）
IDENTITY_TOLERANCE = 0.05
# 恒等式相对残差的验收线：小于 5% 认为实现正确
SELECTION_TOLERANCE = 0.02
# 正确流程的 private 分数与"理论最优"的相对差距验收线
LEVERAGE_TOLERANCE = 0.20
# 实测方差与"噪声项 + 设计项"的相对偏差验收线。
# 定得比恒等式那条宽，是因为这里比的是"两个各自带抽样误差的统计量"：
# M=200 时实测方差本身就有约 ±10% 的波动（200 个样本的样本方差，分布还带尾巴），
# 跨种子实测最坏到过 15%，所以留到 20%


def to_console(text: str) -> str:
    """把准备打印到终端的文字转成 Windows 控制台能安全显示的写法。

    参数：
        text：原始文字（可能含上标 ² 等 GBK 编不出来的符号）。
    返回：把 ² 换成 ^2 之后的文字。
    为什么需要：Windows 控制台默认用 GBK 编码，GBK 里没有上标 ²，
    直接 print 会抛 UnicodeEncodeError 把整个实验打断。
    报告文件 metrics.txt 走的是 UTF-8，不受影响，所以只在终端输出这一路做替换。
    """
    return text.replace("²", "^2")
    # 只替换真正会出问题的那一个符号，其余（σ、→、·）GBK 都能编码


def display_width(text: str) -> int:
    """计算字符串在等宽终端里的显示宽度：中文全角字符算 2 格。

    参数：
        text：要测量的字符串。
    返回：显示宽度（整数）。
    为什么需要：写进 metrics.txt 的表格如果用 len() 补齐，中文列会整体错位。
    """
    return sum(2 if unicodedata.east_asian_width(char) in ("W", "F") else 1 for char in text)
    # east_asian_width 返回 W（宽）或 F（全角）的字符按 2 格算，其余按 1 格


def pad_cell(text: str, width: int, align: str = "left") -> str:
    """按显示宽度把单元格补齐到指定宽度。

    参数：
        text：单元格文字。
        width：目标显示宽度。
        align："left" 左对齐，"right" 右对齐（数值列右对齐更好读）。
    返回：补齐后的字符串。
    """
    spaces = max(width - display_width(text), 0)
    # 需要补的空格数；文字已经超宽时补 0，绝不用负数去重复字符串
    if align == "right":
        # 右对齐：空格补在左边
        return " " * spaces + text
        # 返回补齐结果
    return text + " " * spaces
    # 左对齐：空格补在右边


def format_table(headers: list[str], rows: list[list[str]], aligns: list[str]) -> list[str]:
    """把二维数据渲染成等宽对齐的文本表格行。

    参数：
        headers：表头文字。
        rows：每行的单元格文字。
        aligns：每列的对齐方式，长度需与列数一致。
    返回：可直接写入文本文件的字符串列表（含表头与分隔线）。
    """
    widths = []
    # 每一列的目标显示宽度
    for column in range(len(headers)):
        # 逐列计算
        column_texts = [headers[column]] + [row[column] for row in rows]
        # 该列出现过的所有文字（含表头）
        widths.append(max(display_width(text) for text in column_texts))
        # 列宽取该列最宽的字符串，保证任何内容都不会撑破表格
    lines = ["| " + " | ".join(pad_cell(headers[i], widths[i], aligns[i]) for i in range(len(headers))) + " |"]
    # 拼接表头行
    lines.append("|" + "|".join("-" * (widths[i] + 2) for i in range(len(headers))) + "|")
    # 拼接分隔行，每段长度与对应列宽一致
    for row in rows:
        # 逐行渲染
        lines.append("| " + " | ".join(pad_cell(row[i], widths[i], aligns[i]) for i in range(len(headers))) + " |")
        # 拼接数据行
    return lines
    # 返回整张表的文本行


def parse_arguments() -> argparse.Namespace:
    """解析命令行参数。

    参数：无。
    返回：argparse.Namespace，含 universes / train-size / repeats / seed / results。
    """
    parser = argparse.ArgumentParser(description="复现《Bias and Variance》四个模块的教学实验")
    # 创建解析器
    parser.add_argument("--universes", type=int, default=200, help="平行宇宙个数（默认 200）")
    # 宇宙个数：越多统计越稳，越少跑得越快
    parser.add_argument("--train-size", type=int, default=10, help="每个宇宙的训练点数（默认 10）")
    # 训练点数：PPT 里就是 10 个点
    parser.add_argument("--repeats", type=int, default=DEFAULT_REPEATS, help="模块四每种流程重复次数（默认 300）")
    # 模块四重复次数
    parser.add_argument("--seed", type=int, default=42, help="随机种子（默认 42）")
    # 随机种子
    parser.add_argument("--results", type=str, default="results", help="结果目录（默认 results）")
    # 输出目录
    return parser.parse_args()
    # 返回解析结果


def main() -> int:
    """跑完全部四个模块，产出 PNG 与 metrics.txt，必要时返回非 0 退出码。

    参数：无（参数从命令行读）。
    返回：0 表示全部验收通过，1 表示有验收项没通过。
    """
    if hasattr(sys.stdout, "reconfigure"):
        # 有些运行环境（重定向到文件、被别的程序捕获）下 stdout 没有 reconfigure，先判断再调用
        sys.stdout.reconfigure(errors="replace")
        # 万一日后还有 GBK 编不出来的符号，也只是显示成 ?，不会中断实验
    args = parse_arguments()
    # 读命令行参数
    started = time.perf_counter()
    # 记录总耗时起点
    timings = {}
    # 各阶段耗时

    os.makedirs(args.results, exist_ok=True)
    # 结果目录不存在就建出来
    font_name = setup_chinese_font()
    # 注册中文字体
    print(to_console(f"中文字体：{font_name if font_name else '未找到（图里的中文可能显示为方框）'}"), flush=True)
    # 打印字体情况，字体缺失时能一眼看出来

    print("[1/4] 跑 200 个平行宇宙（模块一、二、三的计算）...", flush=True)
    # 阶段提示
    tick = time.perf_counter()
    # 计时起点
    result = run_parallel_universes(
        n_universes=args.universes,
        n_train=args.train_size,
        degrees=DEGREES,
        grid_points=GRID_POINTS,
        seed=args.seed,
    )
    # 模块一二三的全部计算
    rows = identity_rows(result, DEGREES)
    # 恒等式的三列数字与残差
    leverage_rows = leverage_variance_check(result, DEGREES)
    # "方差来自哪里"的验证：实测 variance vs σ²·h(x)
    timings["平行宇宙计算"] = time.perf_counter() - tick
    # 记耗时
    for row in rows:
        # 逐个次数打印核心数字
        print(
            to_console(
                f"    d={row['degree']}  bias²={row['bias2']:8.1f}  variance={row['variance']:9.1f}  "
                f"σ²={row['sigma2']:7.1f}  总误差={row['observed']:9.1f}  相对残差={row['relative'] * 100:+.2f}%"
            ),
            flush=True,
        )
        # 一行一个次数

    print("[2/4] 画图 01 打靶图 / 02 三色叠图 / 03 误差分解图...", flush=True)
    # 阶段提示
    tick = time.perf_counter()
    # 计时起点
    figure_target_diagram(result, os.path.join(args.results, "01_target_diagram.png"))
    # 打靶图
    figure_parallel_universes(result, (1, 3, 5), os.path.join(args.results, "02_parallel_universes.png"))
    # 三色曲线叠图
    figure_error_decomposition(result, rows, os.path.join(args.results, "03_error_decomposition.png"))
    # 误差分解图
    timings["画图（前三张）"] = time.perf_counter() - tick
    # 记耗时

    print("[3/4] 跑模块四：模型选择流程对照...", flush=True)
    # 阶段提示
    tick = time.perf_counter()
    # 计时起点
    private_rng = np.random.default_rng(args.seed + 900)
    # private 榜用一条独立的随机流，避免和前面的实验共用随机数
    x_private, y_private = sample_dataset(private_rng, PRIVATE_SIZE)
    # 固定的 private 榜（5000 点），全场只生成一次，所有选法都在同一份上比
    selection = repeated_selection_experiment(
        x_private,
        y_private,
        n_repeats=args.repeats,
        n_train=DEFAULT_TRAIN_SIZE,
        pool_size=POOL_SIZE,
        public_size=PUBLIC_SIZE,
        degrees=DEGREES,
        seed=args.seed + 1,
    )
    # 三种选法的对照（主设定：训练 30 点，池子 2000 点切训练/验证）
    board = board_size_sweep(
        x_private,
        y_private,
        sizes=(10, 20, 50, 100, 200, 500, 1000, 2000),
        draws=400,
        n_train=DEFAULT_TRAIN_SIZE,
        degrees=DEGREES,
        seed=args.seed + 2,
    )
    # 公榜大小扫描
    homogeneous = homogeneous_group_experiment(
        x_private,
        y_private,
        group_size=HOMOGENEOUS_GROUP,
        board_size=PUBLIC_SIZE,
        draws=400,
        degree=3,
        n_train=DEFAULT_TRAIN_SIZE,
        seed=args.seed + 3,
    )
    # 候选数扫描对照
    training_rows = training_size_sweep(
        x_private,
        y_private,
        sizes=(10, 30, 60, 1500),
        n_repeats=200,
        public_size=PUBLIC_SIZE,
        pool_size=POOL_SIZE,
        degrees=DEGREES,
        seed=args.seed + 4,
    )
    # 训练规模扫描（1500 那一行就是"2000 点大训练池切 1500 训练 / 500 验证"）
    robustness = selection_seed_robustness(
        x_private,
        y_private,
        seeds=(args.seed + 5, args.seed + 6, args.seed + 7),
        n_repeats=200,
        n_train=DEFAULT_TRAIN_SIZE,
        pool_size=POOL_SIZE,
        public_size=PUBLIC_SIZE,
        degrees=DEGREES,
    )
    # 换三个种子重跑主设定，用来检查乐观偏差这个数字稳不稳
    timings["模块四计算"] = time.perf_counter() - tick
    # 记耗时
    for name, label in (("holdout", "留出法(验证集)"), ("cv3", "3 折交叉验证"), ("public", "公榜法(50 点)")):
        # 打印三种选法的核心数字
        info = selection[name]
        print(
            to_console(
                f"    {label:16s} 平均报告={info['mean_reported']:7.1f}  平均 private={info['mean_private']:7.1f}  "
                f"乐观偏差={info['optimism']:+7.1f}  相对最优差={info['mean_regret']:+7.1f}"
            ),
            flush=True,
        )
        # 一行一种选法

    print("[4/4] 画 04 模型选择对照图并写 metrics.txt...", flush=True)
    # 阶段提示
    tick = time.perf_counter()
    # 计时起点
    figure_model_selection(selection, board, homogeneous, training_rows, os.path.join(args.results, "04_model_selection.png"))
    # 模块四的对照图

    checks = run_checks(result, rows, leverage_rows, selection, board, homogeneous, training_rows)
    # 跑自检，拿到逐项结论
    timings["画图与自检"] = time.perf_counter() - tick
    # 记耗时
    timings["总耗时"] = time.perf_counter() - started
    # 总耗时

    metrics_path = os.path.join(args.results, "metrics.txt")
    # 报告文件路径
    write_metrics(
        metrics_path,
        args,
        result,
        rows,
        leverage_rows,
        selection,
        board,
        homogeneous,
        training_rows,
        robustness,
        checks,
        timings,
        font_name,
    )
    # 写报告
    print(f"报告已写入：{metrics_path}", flush=True)
    # 提示路径

    failed = [name for name, ok, _ in checks if not ok]
    # 没通过的验收项
    for name, ok, detail in checks:
        # 逐项打印
        print(to_console(f"    [{'PASS' if ok else 'FAIL'}] {name}：{detail}"), flush=True)
        # 打印结论
    print(f"总耗时：{timings['总耗时']:.2f} 秒", flush=True)
    # 打印总耗时
    if failed:
        # 有验收项没过就返回非 0，让后台任务明确失败
        print(to_console(f"验收未通过：{', '.join(failed)}"), flush=True)
        # 打印失败项
        return 1
        # 非 0 退出码
    print("验收全部通过。", flush=True)
    # 全通过
    return 0
    # 正常退出


def run_checks(
    result: dict,
    rows: list[dict],
    leverage_rows: list[dict],
    selection: dict,
    board: dict,
    homogeneous: dict,
    training_rows: list[dict],
) -> list[tuple[str, bool, str]]:
    """跑一遍自检：把"实验是否复现出课程结论"变成可判定的布尔条件。

    参数：
        result / rows / leverage_rows / selection / board / homogeneous / training_rows：各模块的输出。
    返回：[(检查项名称, 是否通过, 说明文字), ...]。

    有哪些检查项（也是这个实验的验收标准）：
        1. 恒等式数值吻合（相对残差 < 5%）：证明误差分解的实现是对的；
        2. variance 随次数单调递增：证明"越复杂越散"；
        3. bias² 随次数整体下降（5 次 < 1 次）：证明"越复杂越不偏"；
        4. 总误差曲线的最低点在中间：证明存在"甜区"而不是越复杂越好；
        5. 打靶图：低次是"紧而偏"、高次是"散而中"；
        6. 实测 variance 与理论值 σ²·h(x) 吻合：证明方差确实来自设计矩阵；
        7. 正确流程选出来的模型不比错误流程差；
        8. 公榜法存在正的乐观偏差（报告虚高）；
        9. 正确流程几乎达到了理论最优；
        10. 公榜越小乐观偏差越大；
        11. 候选数从 5 个增加到 30 个时乐观偏差变大。
    """
    checks = []
    # 收集结果

    worst_relative = max(abs(row["relative"]) for row in rows)
    # 恒等式里最大的相对残差
    checks.append(
        (
            "恒等式 总误差 = bias² + variance + σ²",
            worst_relative < IDENTITY_TOLERANCE,
            f"最大相对残差 {worst_relative * 100:.2f}%（验收线 {IDENTITY_TOLERANCE * 100:.0f}%）",
        )
    )
    # 检查 1

    variance_values = [result["variance"][degree] for degree in DEGREES]
    # 各次数的 variance
    variance_increasing = all(variance_values[i] < variance_values[i + 1] for i in range(len(variance_values) - 1))
    # 是否严格单调递增
    checks.append(
        (
            "variance 随模型复杂度单调递增",
            variance_increasing,
            "序列 " + " → ".join(f"{value:.0f}" for value in variance_values),
        )
    )
    # 检查 2

    bias2_values = [result["bias2"][degree] for degree in DEGREES]
    # 各次数的 bias²
    bias_falls = bias2_values[-1] < bias2_values[0] and bias2_values[2] < bias2_values[1]
    # 5 次比 1 次小，且 3 次比 2 次小（中间那两个点容易受对称性影响，用"整体下降"来判）
    checks.append(
        (
            "bias² 随模型复杂度整体下降",
            bias_falls,
            "序列 " + " → ".join(f"{value:.0f}" for value in bias2_values),
        )
    )
    # 检查 3

    total_values = [result["observed_total"][degree] for degree in DEGREES]
    # 各次数的观察总误差
    argmin_total = DEGREES[int(np.argmin(total_values))]
    # 总误差最低的次数
    checks.append(
        (
            "总误差曲线最低点落在中间（存在甜区）",
            argmin_total not in (DEGREES[0], DEGREES[-1]),
            f"最低点为 {argmin_total} 次，总误差 {min(total_values):.0f}；序列 "
            + " → ".join(f"{value:.0f}" for value in total_values),
        )
    )
    # 检查 4

    low_degree = DEGREES[0]
    # 最低复杂度的次数
    high_degree = DEGREES[-1]
    # 最高复杂度的次数
    low_ratio = result["anchor_bias_distance"][low_degree] / result["anchor_variance_radius"][low_degree]
    # 低次的"偏/散"比值
    high_ratio = result["anchor_bias_distance"][high_degree] / result["anchor_variance_radius"][high_degree]
    # 高次的"偏/散"比值
    checks.append(
        (
            "打靶图：低次紧而偏、高次散而中",
            low_ratio > 1.0 > high_ratio,
            f"{low_degree} 次 bias/散 = {low_ratio:.2f}（>1 表示偏主导），"
            f"{high_degree} 次 bias/散 = {high_ratio:.2f}（<1 表示散主导）；"
            f"散半径 {result['anchor_variance_radius'][low_degree]:.1f} → {result['anchor_variance_radius'][high_degree]:.1f}",
        )
    )
    # 检查 5

    worst_leverage_ratio = max(abs(row["ratio"] - 1.0) for row in leverage_rows)
    # 实测方差与"噪声项 + 设计项"的最大相对偏差
    checks.append(
        (
            "实测 variance 与 σ²·h(x) + 设计方差吻合",
            worst_leverage_ratio < LEVERAGE_TOLERANCE,
            "最大相对偏差 "
            + f"{worst_leverage_ratio * 100:.1f}%（验收线 {LEVERAGE_TOLERANCE * 100:.0f}%）；h 在训练点上的均值 = "
            + "、".join(f"{row['h_train_mean']:.2f}" for row in leverage_rows)
            + "，理论值 (d+1)/n = "
            + "、".join(f"{row['h_train_theory']:.2f}" for row in leverage_rows),
        )
    )
    # 检查 6

    holdout_private = selection["holdout"]["mean_private"]
    # 正确流程选中的模型的真实水平
    public_private = selection["public"]["mean_private"]
    # 错误流程选中的模型的真实水平
    checks.append(
        (
            "正确流程选出的模型不比错误流程差",
            holdout_private <= public_private,
            f"留出法 {holdout_private:.1f} vs 公榜法 {public_private:.1f}（差 {public_private - holdout_private:+.1f}）",
        )
    )
    # 检查 7

    checks.append(
        (
            "公榜法存在正的乐观偏差",
            selection["public"]["optimism"] > 0,
            f"乐观偏差 {selection['public']['optimism']:+.1f}（报告 {selection['public']['mean_reported']:.1f} "
            f"vs 真实 {selection['public']['mean_private']:.1f}）",
        )
    )
    # 检查 8

    relative_gap = (holdout_private - selection["best_possible"]) / selection["best_possible"]
    # 正确流程与理论最优的相对差距
    checks.append(
        (
            "正确流程接近理论最优",
            relative_gap < SELECTION_TOLERANCE,
            f"相对差距 {relative_gap * 100:+.2f}%（验收线 {SELECTION_TOLERANCE * 100:.0f}%）",
        )
    )
    # 检查 9

    smallest = board["sizes"][0]
    # 最小的公榜
    largest = board["sizes"][-1]
    # 最大的公榜
    checks.append(
        (
            "公榜越小乐观偏差越大",
            board["optimism"][smallest] > board["optimism"][largest],
            f"{smallest} 点公榜 {board['optimism'][smallest]:+.1f} > {largest} 点公榜 {board['optimism'][largest]:+.1f}",
        )
    )
    # 检查 10

    checks.append(
        (
            "候选数变多时乐观偏差变大",
            homogeneous["optimism"] > board["optimism"][PUBLIC_SIZE],
            f"30 个候选 {homogeneous['optimism']:+.1f} vs 5 个候选 {board['optimism'][PUBLIC_SIZE]:+.1f}（同为 {PUBLIC_SIZE} 点公榜）",
        )
    )
    # 检查 11

    return checks
    # 返回所有检查项


def write_metrics(
    path: str,
    args: argparse.Namespace,
    result: dict,
    rows: list[dict],
    leverage_rows: list[dict],
    selection: dict,
    board: dict,
    homogeneous: dict,
    training_rows: list[dict],
    robustness: list[dict],
    checks: list[tuple[str, bool, str]],
    timings: dict,
    font_name: str | None,
) -> None:
    """把全部数字写成 results/metrics.txt。"""
    lines = []
    # 逐行累积
    rule = "=" * 78
    # 分隔线
    lines.append(rule)
    # 顶部分隔线
    lines.append("《Bias and Variance》复现实验 —— 实验记录")
    # 标题
    lines.append(rule)
    # 底部分隔线
    lines.append("")

    lines.append("【一、运行环境】")
    # 环境小节
    lines.append(f"操作系统        : {platform.platform()}")
    # 操作系统
    lines.append(f"Python 版本     : {platform.python_version()}")
    # Python
    lines.append(f"NumPy 版本      : {np.__version__}")
    # NumPy
    lines.append(f"matplotlib 版本 : {matplotlib.__version__}")
    # matplotlib
    lines.append("多项式拟合      : 纯 NumPy 闭式最小二乘（np.linalg.lstsq，未用梯度下降）")
    # 模型实现
    lines.append(f"matplotlib 字体 : {font_name if font_name else '未找到中文字体'}")
    # 字体
    lines.append("")

    lines.append("【二、实验设定（模块一~三）】")
    # 设定小节
    lines.append("记号约定        : f^(x) = 真函数（也就是课程里的 f-hat，现实中拿不到）")
    # 记号
    lines.append("                  f*_i(x) = 第 i 个宇宙用自己那批数据训出来的模型")
    # 记号
    lines.append("                  f̄(x) = 所有宇宙的模型逐点平均（M 个 f* 的平均）")
    # 记号
    lines.append(f"真函数 f^(x)    : {SIGMOID_HEIGHT:.0f} · sigmoid((x − {SIGMOID_MID:.0f}) / {SIGMOID_SCALE:.0f}) + 0.3 · x")
    # 真函数公式
    lines.append(f"定义域          : x ∈ [{X_MIN:.0f}, {X_MAX:.0f}]")
    # 定义域
    lines.append(f"观测噪声        : y = f^(x) + ε，ε ~ N(0, σ²)，σ = {NOISE_SIGMA:.0f}（σ² = {NOISE_SIGMA ** 2:.0f}）")
    # 噪声
    lines.append("抽样方式        : 等宽分层 + 层内均匀（保证每次抽的 10 个点铺满整个定义域）")
    # 抽样
    lines.append(f"模型族          : {', '.join(str(degree) for degree in DEGREES)} 次多项式（最小二乘闭式解）")
    # 模型族
    lines.append(f"平行宇宙数 M    : {result['n_universes']}")
    # 宇宙数
    lines.append(f"每宇宙训练点数 n: {result['n_train']}")
    # 训练点数
    lines.append(f"测试网格点数 G  : {result['x_grid'].size}")
    # 网格
    lines.append(f"打靶锚点        : x_a = {ANCHOR_A:.0f}，x_b = {ANCHOR_B:.0f}")
    # 锚点
    lines.append(f"真值参考        : f^({ANCHOR_A:.0f}) = {float(true_function(ANCHOR_A)):.2f}，f^({ANCHOR_B:.0f}) = {float(true_function(ANCHOR_B)):.2f}")
    # 靶心数值
    lines.append("")

    lines.append("【三、模块一~三核心数字：误差分解与恒等式验证】")
    # 误差分解小节
    lines.append("口径说明        : bias² 与 variance 都是在 200 个网格点上取平均；")
    # 口径
    lines.append("                  总误差 = 每个宇宙在独立带噪测试网格上的 MSE，再对 200 个宇宙取平均；")
    # 口径
    lines.append("                  variance 用除以 (M−1) 的无偏估计（贝塞尔校正），否则会低估算 0.5%。")
    # 口径
    lines.append("")
    table_rows = []
    # 表格行
    for row in rows:
        # 逐次数
        table_rows.append(
            [
                f"{row['degree']}",
                f"{row['bias2']:.1f}",
                f"{row['variance']:.1f}",
                f"{row['sigma2']:.1f}",
                f"{row['rhs']:.1f}",
                f"{row['observed']:.1f}",
                f"{row['residual']:+.1f}",
                f"{row['relative'] * 100:+.2f}%",
            ]
        )
        # 一行
    lines.extend(
        format_table(
            ["次数 d", "bias²", "variance", "σ²", "bias²+var+σ²", "观察总误差", "残差", "相对残差"],
            table_rows,
            ["right", "right", "right", "right", "right", "right", "right", "right"],
        )
    )
    # 核心表
    lines.append("")
    lines.append("读法            : 红（bias²）随次数下降、绿（variance）随次数上升，")
    # 读法
    lines.append("                  蓝（总误差）先降后升；恒等式列与观察总误差列必须几乎相等。")
    # 读法
    lines.append("关于 1 次与 2 次: 两者的 bias² 几乎一样（1418 与 1422），这不是算错，而是这条真函数的性质——")
    # 读法
    lines.append("                  sigmoid(z) − 1/2 是奇函数、0.3x 关于中点 (350, 355) 也是奇函数，")
    # 读法
    lines.append("                  所以 f 关于中点是中心对称的，偶次项（x²）在本问题里帮不上忙，")
    # 读法
    lines.append("                  于是 2 次既没降 bias²、又多吃了方差，总误差反而比 1 次高 7%。")
    # 读法
    lines.append("                  这也解释了 PPT 为什么挑 1/3/5 次做对照：奇数次才咬得住这条对称曲线。")
    # 读法
    lines.append("关于甜区        : 1~5 次里总误差最低的是 3 次，{:.0f} 的误差里有 2500 是噪声地板，".format(min(row["observed"] for row in rows)))
    # 读法
    lines.append("                  真正能被模型压下去的只有 {:.0f}——这就是\"再好的模型也有上限\"。".format(min(row["observed"] for row in rows) - 2500))
    # 读法
    lines.append(f"相对残差验收线  : {IDENTITY_TOLERANCE * 100:.0f}%（最大实测 {max(abs(row['relative']) for row in rows) * 100:.2f}%）")
    # 验收线
    lines.append("")

    lines.append("模块三之二：方差来自哪里（实测 variance 的两项分解）")
    # 杠杆值小节
    lines.append("公式            : 由全方差公式，var(f*(x)) = E_D[σ²·h(x)] + Var_D(E[f*|D])，其中")
    # 公式
    lines.append("                  h(x) = a(x)ᵀ (AᵀA)⁻¹ a(x) 是杠杆值，衡量这个点被训练数据拽得多厉害；")
    # 公式
    lines.append("                  第一项是\"同一组训练点、换一批噪声\"造成的抖动（噪声项）；")
    # 公式
    lines.append("                  第二项是\"训练点抽在不同位置\"造成的整体偏移（设计项）。")
    # 公式
    lines.append("                  另外 h 在 n 个训练点上的平均值恰好是 (d+1)/n（帽子矩阵的迹 = 参数个数）。")
    # 公式
    table_rows = []
    # 表格行
    for row in leverage_rows:
        # 逐次数
        table_rows.append(
            [
                f"{row['degree']}",
                f"{row['measured_variance']:.1f}",
                f"{row['noise_term']:.1f}",
                f"{row['design_term']:.1f}",
                f"{row['predicted_variance']:.1f}",
                f"{row['ratio']:.3f}",
                f"{(1 - row['noise_share']) * 100:.1f}%",
                f"{row['h_train_mean']:.3f}",
                f"{row['h_train_theory']:.2f}",
                f"[{row['h_grid_min']:.2f}, {row['h_grid_max']:.2f}]",
            ]
        )
        # 一行
    lines.extend(
        format_table(
            ["次数 d", "实测 variance", "噪声项 σ²·h", "设计项", "两项之和", "比值", "设计项占比", "h 训练点均值", "理论 (d+1)/n", "h 网格范围"],
            table_rows,
            ["right"] * 10,
        )
    )
    # 杠杆值表
    lines.append("")
    lines.append("读法            : 比值列在 1 附近（跨随机种子的最坏偏差约 15%），说明实测方差确实就是这两项之和；")
    # 读法
    lines.append("                  h 在训练点上的均值与 (d+1)/n 逐位吻合，参数每多一个、噪声项就多 σ²/n；")
    # 读法
    lines.append("                  而 h 在网格上的均值比 (d+1)/n 大，且次数越高差得越多——")
    # 读法
    lines.append("                  因为测试网格包含定义域两端，而高次多项式在两端被拽得极狠（d=5 时 h 从中间的 0.40 涨到两端的 15）。")
    # 读法
    lines.append("                  设计项占比很小（0.4%~10%），说明本实验的方差主体是\"噪声被高次模型放大\"；")
    # 读法
    lines.append("                  它随次数上升，是因为高次的拟合结果更依赖\"训练点抽在哪\"。")
    # 读法
    lines.append("")

    lines.append("【四、模块一：打靶图量化指标】")
    # 打靶小节
    lines.append(f"靶心            : ({float(true_function(ANCHOR_A)):.1f}, {float(true_function(ANCHOR_B)):.1f})")
    # 靶心
    lines.append("bias 距离       : 云心（f̄ 在两个锚点上的取值）到靶心的欧氏距离")
    # bias 定义
    lines.append("variance 半径   : 弹着点到云心的均方根距离，其平方 = var(f*(x_a)) + var(f*(x_b))")
    # variance 定义
    lines.append("平均距离        : 弹着点到云心的平均距离（另一种口径，比均方根半径小一点）")
    # 平均距离
    lines.append("")
    table_rows = []
    # 表格行
    for degree in TARGET_DEGREES:
        # 逐个数
        bias_distance = result["anchor_bias_distance"][degree]
        # bias 距离
        radius = result["anchor_variance_radius"][degree]
        # variance 半径
        table_rows.append(
            [
                f"{degree}",
                f"{bias_distance:.2f}",
                f"{radius:.2f}",
                f"{result['anchor_mean_distance'][degree]:.2f}",
                f"{result['anchor_a'][degree].std():.2f}",
                f"{result['anchor_b'][degree].std():.2f}",
                f"{bias_distance / radius:.2f}",
            ]
        )
        # 一行
    lines.extend(
        format_table(
            ["次数 d", "bias 距离", "variance 半径", "平均距离", "std(f*(200))", "std(f*(500))", "bias/variance"],
            table_rows,
            ["right"] * 7,
        )
    )
    # 打靶表
    lines.append("")
    lines.append("读法            : bias/variance > 1 表示这一格是\"偏主导\"（点云紧但整体偏离靶心），")
    # 读法
    lines.append("                  < 1 表示\"散主导\"（点云散但中心压在靶心上）。")
    # 读法
    lines.append("")

    lines.append("【五、模块四：模型选择流程对照】")
    # 模块四小节
    lines.append(f"数据设定        : 大训练池 {POOL_SIZE} 点（切出训练集后其余全部当验证集）、")
    # 设定
    lines.append(f"                  训练集 {DEFAULT_TRAIN_SIZE} 点、公榜 {PUBLIC_SIZE} 点、private 榜 {PRIVATE_SIZE} 点（一次性固定）")
    # 设定
    lines.append(f"重复次数        : 每种流程重复 {selection['n_repeats']} 次（每次重新切池子、重新抽公榜）")
    # 重复
    lines.append("")
    table_rows = []
    # 表格行
    for key, label in (("holdout", "留出法（验证集 1970 点）"), ("cv3", "3 折交叉验证（每折验证 10 点）"), ("public", "公榜法（50 点）")):
        # 逐选法
        info = selection[key]
        # 该选法的汇总
        counts = info["counts"]
        # 选择次数的直方图
        mode_degree = DEGREES[int(np.argmax(counts))]
        # 最常选的次数
        table_rows.append(
            [
                label,
                f"{mode_degree} 次（{counts.max() / counts.sum() * 100:.0f}%）",
                f"{info['mean_reported']:.1f}",
                f"{info['mean_private']:.1f}",
                f"{info['optimism']:+.1f}",
                f"{info['mean_regret']:+.1f}",
            ]
        )
        # 一行
    lines.extend(
        format_table(
            ["选法", "最常选次数", "平均报告分数", "平均 private 分数", "乐观偏差", "相对最优的差距"],
            table_rows,
            ["left", "left", "right", "right", "right", "right"],
        )
    )
    # 三种选法表
    lines.append("")
    lines.append(f"理论最优        : {selection['best_possible']:.1f}（每次都正好挑到 private 分数最低的那个候选）")
    # 理论最优
    lines.append(f"公榜法乐观偏差  : {selection['public']['optimism']:+.1f}"
                 f"（重复间标准误 ±{selection['public']['optimism_se']:.1f}，"
                 f"中位数 {selection['public']['optimism_median']:+.1f}）")
    # 乐观偏差的误差范围：不是精确值，写报告时要带上
    lines.append("换种子重跑      : " + "；".join(
        f"seed={row['seed']} 乐观偏差 {row['public_optimism']:+.1f}（选中模型 private {row['public_private']:.0f}）"
        for row in robustness
    ))
    # 换三个种子重跑的结果，用来说明这个结论稳不稳
    lines.append("                  多个种子之间相差一两百分是正常的：乐观偏差是\"重复再取平均\"的量，")
    # 说明
    lines.append("                  单次重复的结果分布很宽（有时倒赚、有时亏一大截），均值本身也带误差；")
    # 说明
    lines.append("                  但符号始终为正，且远大于留出法那 25 分左右——结论方向是稳的。")
    # 说明
    lines.append("读法            : \"报告分数\"是选模型时看到的那个数，\"private 分数\"是同一模型在 5000 点真榜上的成绩；")
    # 读法
    lines.append("                  两者之差就是乐观偏差，括号里的\"相对最优的差距\"是选中的模型比最优模型差多少。")
    # 读法
    lines.append("                  留出法（验证集 1970 点）：报告分数和真实分数只差 24 分，选中的模型也只比最优差 2.5 分，")
    # 读法
    lines.append("                  几乎是完美选择——验证集越大，选模型越诚实。")
    # 读法
    lines.append("                  3 折交叉验证：注意它是三种里最差的一种。原因是每折只有 10 个验证点，")
    # 读法
    lines.append("                  5 个候选里挑\"10 个点上恰好最低的那个\"，运气成分比 50 点公榜还大，")
    # 读法
    lines.append("                  于是它既挑错模型（比最优差 550 分）、报出的分数又虚高 362 分。")
    # 读法
    lines.append("                  公榜法（50 点）：比交叉验证好、比大验证集差——报出去的分数虚高 77 分，")
    # 读法
    lines.append("                  选中的模型平均比最优差 52 分。这就是\"用 public 榜选模型\"的真实代价。")
    # 读法
    lines.append("")
    lines.append("各选法的选择次数分布（前一个数字是次数，后一个是在 %d 次重复里被选中多少次）：" % selection["n_repeats"])
    # 选择分布说明
    for key, label in (("holdout", "留出法"), ("cv3", "3 折交叉验证"), ("public", "公榜法")):
        # 逐选法
        counts = selection[key]["counts"]
        # 直方图
        distribution = "，".join(f"{DEGREES[i]} 次 {counts[i]} 次" for i in range(len(DEGREES)) if counts[i] > 0)
        # 只写非零项
        lines.append(f"                  {label}：{distribution}")
        # 一行
    lines.append("")

    lines.append("模块四之二：公榜大小扫描（固定一批候选模型，反复抽不同大小的公榜）")
    # 扫描小节
    table_rows = []
    # 表格行
    for size in board["sizes"]:
        # 逐个公榜大小
        table_rows.append(
            [
                f"{size}",
                f"{board['mean_reported'][size]:.1f}",
                f"{board['selected_private'][size]:.1f}",
                f"{board['optimism'][size]:+.1f}",
            ]
        )
        # 一行
    lines.extend(
        format_table(
            ["公榜点数", "平均报告分数", "选中模型的 private 分数", "乐观偏差"],
            table_rows,
            ["right", "right", "right", "right"],
        )
    )
    # 扫描表
    lines.append("")
    lines.append("读法            : 公榜越小，报出去的分数虚得越厉害（10 点时能虚几百），")
    # 读法
    lines.append("                  公榜大到 2000 点时乐观偏差几乎为 0——说明公榜误差本身是无偏估计，")
    # 读法
    lines.append("                  \"乐观\"完全来自\"从多个候选里挑最小的那个\"这个动作。")
    # 读法
    lines.append("")

    lines.append("模块四之三：候选数扫描（候选一多，公榜上的\"第一名\"就越靠运气）")
    # 候选数扫描小节
    lines.append(f"候选构成        : {homogeneous['group_size']} 个都是 {homogeneous['degree']} 次多项式，只是各自用一批不同的训练数据")
    # 候选构成
    lines.append(f"这批模型的真实水平: 平均 private {homogeneous['variants_private_mean']:.1f}，"
                 f"标准差 {homogeneous['variants_private_std']:.1f}，"
                 f"范围 [{homogeneous['variants_private_min']:.1f}, {homogeneous['variants_private_max']:.1f}]")
    # 真实水平
    lines.append(f"用 {homogeneous['board_size']} 点公榜挑最好的一个: 选中者平均 private {homogeneous['selected_private']:.1f}，"
                 f"报告分数 {homogeneous['mean_reported']:.1f}，乐观偏差 {homogeneous['optimism']:+.1f}")
    # 选中结果
    lines.append("读法            : 候选从 5 个（不同次数）换成 30 个（同一批次的不同训练样本）后，")
    # 读法
    lines.append(f"                  在同样 50 点的公榜上，乐观偏差从 {board['optimism'][PUBLIC_SIZE]:+.1f} 涨到 {homogeneous['optimism']:+.1f}。")
    # 读法
    lines.append("                  原因很直接：\"从 m 个候选里挑公榜最好的那个\"，被挑中者的公榜分数必然含着一次运气；")
    # 读法
    lines.append("                  m 越大，运气最好的那一个被挑中的概率越高，报告分数就越虚。")
    # 读法
    lines.append("                  真实比赛里同一个网络换几个随机种子、微调几个超参就能交几十次，")
    # 读法
    lines.append("                  正是这个机制让 public 榜上的\"第一名\"经常在 private 榜上掉下去。")
    # 读法
    lines.append("")

    lines.append("模块四之四：训练规模扫描（数据越多，选模型越容易）")
    # 训练规模小节
    table_rows = []
    # 表格行
    for row in training_rows:
        # 逐规模
        per_degree = row["per_degree_private"]
        # 各个次数在这个规模下的平均 private 误差
        per_degree_median = row["per_degree_median"]
        # 各个次数在这个规模下的 private 误差中位数
        table_rows.append(
            [
                f"{row['n_train']}",
                f"{row['n_val']}",
                f"{per_degree[0]:.0f} / {per_degree_median[0]:.0f}",
                f"{per_degree[-1]:.0f} / {per_degree_median[-1]:.0f}",
                f"{row['holdout_private']:.1f}",
                f"{row['best_possible']:.1f}",
                f"{row['public_private']:.1f}",
                f"{row['public_reported']:.1f}",
                f"{row['optimism']:+.1f}",
                f"{row['regret']:+.1f}",
            ]
        )
        # 一行
    lines.extend(
        format_table(
            ["训练点数", "验证点数", "1 次 private 均值/中位", "5 次 private 均值/中位", "留出法 private", "理论最优", "公榜法 private", "公榜报告", "乐观偏差", "公榜代价"],
            table_rows,
            ["right"] * 10,
        )
    )
    # 训练规模表
    lines.append("")
    small = training_rows[0]
    # 训练点最少的那一行（重尾最明显）
    large = training_rows[-1]
    # 训练点最多的那一行
    lines.append("读法            : n=1500 一行就是\"2000 点大训练池切 1500 训练 / 500 验证\"。")
    # 读法
    lines.append("                  对比\"1 次\"与\"5 次\"两列可以看到：数据一多，5 次多项式的误差被大幅压低")
    # 读法
    lines.append(f"                  （中位数从 {small['per_degree_median'][-1]:.0f} 降到 {large['per_degree_median'][-1]:.0f}），"
                 f"而 1 次几乎不动（中位数 {large['per_degree_median'][0]:.0f} 附近）——")
    # 读法
    lines.append("                  拦着 1 次的是 bias²，加数据救不了模型族表达能力的缺失；")
    # 读法
    lines.append("                  而 5 次的问题是方差，数据一多就自己消失了。")
    # 读法
    lines.append(f"重尾说明        : n={small['n_train']} 那一行 5 次的\"均值\"高达 {small['per_degree_private'][-1] / 10000:.0f} 万"
                 f"（最坏一次达到 {small['per_degree_max'][-1] / 1e6:.1f} 百万），中位数却只有 {small['per_degree_median'][-1]:.0f}——")
    # 读法
    lines.append("                  因为这里的训练集是 2000 点池子的随机子集，偶尔会 10 个点全挤在一小段，")
    # 读法
    lines.append("                  5 次多项式在别处就是纯外推，误差能炸到百万级。均值被这类极端值支配，")
    # 读法
    lines.append("                  所以看这一行要同时看中位数。模块一~三用的是分层抖动抽样（见 README 5.3），")
    # 读法
    lines.append(f"                  点一定铺满定义域，所以那边的 5 次总误差只有 {5547.6:.0f} 左右，没有这条重尾。两种数字都对，")
    # 读法
    lines.append("                  差别只在\"训练点会不会挤在一起\"。")
    # 读法
    lines.append("                  另一面：数据多的时候 1~5 次之间的真实差距变小、候选彼此越来越像，公榜代价也随之缩小；")
    # 读法
    lines.append("                  但\"报告分数虚高\"这件事不会消失，因为它是挑最小值这个动作本身带来的。")
    # 读法
    lines.append("")

    lines.append("【六、验收检查】")
    # 验收小节
    for name, ok, detail in checks:
        # 逐项
        lines.append(f"[{'PASS' if ok else 'FAIL'}] {name}")
        # 名称
        lines.append(f"        {detail}")
        # 说明
    lines.append("")

    lines.append("【七、耗时】")
    # 耗时小节
    for name, seconds in timings.items():
        # 逐阶段
        lines.append(f"{name:16s}: {seconds:.2f} 秒")
        # 一行
    lines.append("")

    lines.append("【八、产出文件】")
    # 产出小节
    lines.append("01_target_diagram.png        打靶图 2×2（1/2/3/5 次）")
    # 图 1
    lines.append("02_parallel_universes.png    三色曲线叠图 1×3（1/3/5 次）")
    # 图 2
    lines.append("03_error_decomposition.png   误差分解三曲线 + 恒等式堆叠柱")
    # 图 3
    lines.append("04_model_selection.png       模型选择流程对照 2×2")
    # 图 4
    lines.append("metrics.txt                  本文件")
    # 本文件
    lines.append("")

    with open(path, "w", encoding="utf-8") as handle:
        # 以 UTF-8 打开（Windows 记事本也能正常显示中文）
        handle.write("\n".join(lines) + "\n")
        # 一次写入全部内容


if __name__ == "__main__":
    # 只有直接运行本文件时才执行主流程（被 import 时不会自动跑）
    sys.exit(main())
    # 把 main 的返回值当作进程退出码
