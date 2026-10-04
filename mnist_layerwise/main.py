#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
主流程：复现李宏毅《Why Deep》PPT 里 MNIST 四宫格那一页。

一句话概括这个脚本做的事：
    训练一个 3 隐藏层的小网络 -> 把同一批手写数字样本分别取出
    input / 第1隐藏层 / 第2隐藏层 / 第3隐藏层 四种表示 ->
    各自降到 2 维画成四宫格 -> 再给每一层算轮廓系数和 1-NN 准确率 ->
    看“十类混杂”是不是逐层变成了“十类各抱一团”。

执行顺序（也就是 main() 里的步骤）：
    1. 载入 sklearn 自带的 digits（8x8 手写数字，MNIST 的迷你替代品，免联网）
    2. 分层切成训练集 1347 张 / 测试集 450 张
    3. 用纯 NumPy 的 3 隐藏层 MLP 训练（Adam + 交叉熵），打印最终测试准确率
    4. 取出四层表示，各自 PCA 与 t-SNE 降到 2 维，并算全部量化指标
    5. 画图并写盘：四宫格 PCA、四宫格 t-SNE、训练曲线、指标随层数变化
    6. 把全部数字写进 results/metrics.txt，并在终端打印一份摘要

用法（Git Bash，在本目录下）：
    /c/msys64/ucrt64/bin/python main.py
    /c/msys64/ucrt64/bin/python main.py --epochs 400 --seed 7
"""

from __future__ import annotations
# 打开延迟注解求值，类型标注里可以直接写 np.ndarray

import argparse
# 解析命令行参数，方便换种子、换轮数重跑
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
import sklearn
# 只是为了在报告里写出版本号，便于复现
from sklearn.datasets import load_digits
# sklearn 自带的手写数字数据集：随库安装，不需要联网下载
from sklearn.model_selection import train_test_split
# 分层划分训练/测试集

from analysis import (
    leave_one_out_one_nn,
    pca_2d,
    same_label_nearest_neighbor_rate,
    scatter_ratio,
    silhouette_of,
    split_one_nn,
    tsne_2d,
)
# 指标与降维工具
from figures import (
    figure_layer_metrics,
    figure_representation_grid,
    figure_training_curve,
    setup_chinese_font,
)
# 画图工具
from model import REPRESENTATION_NAMES, DeepMLP
# 网络本体，以及四层表示的名字（与 PPT 上的层名一致）

PIXEL_MAX = 16.0
# load_digits 的像素最大值是 16，用它把像素缩放到 0~1，与姊妹项目保持一致的口径
TEST_SIZE = 0.25
# 测试集比例：1797 * 0.25 ≈ 450 张
SPLIT_SEED = 42
# 划分用的随机种子；刻意与姊妹项目 handcrafted_vs_learned 取同一个值，
# 这样两个实验的训练/测试样本集合完全一致，跨项目对比时不用再解释划分差异
RESULTS_DIR = "results"
# 所有产出（PNG 与 metrics.txt）都放在这个目录下
TARGET_ACCURACY = 0.95
# 任务规定的验收线：测试准确率必须达到 95%


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


def load_dataset() -> tuple[np.ndarray, np.ndarray]:
    """载入 sklearn 自带的 8x8 手写数字数据集，并归一化。

    参数：无。
    返回：(flattened, targets)
        flattened：形状 (1797, 64) 的浮点数组，每行是一张 8x8 图拉平后的 64 维像素，取值 0~1；
        targets：形状 (1797,) 的整数标签，取值 0~9。

    为什么要归一化：像素原始取值是 0~16 的整数。不归一化的话，
    第一层权重的梯度尺度大约是归一化后的 16 倍，Adam 虽然能自适应，
    但初始化时的 He 标准差是按“输入方差约为 1”推导的，除以 16 才能让这个前提成立。
    """
    bunch = load_digits()
    # 载入数据集；这个函数不需要下载，数据直接打包在 scikit-learn 里
    images = bunch.images.astype(np.float64)
    # 原始图像，形状 (1797, 8, 8)，像素 0~16
    flattened = images.reshape(images.shape[0], -1) / PIXEL_MAX
    # 拉平成 (1797, 64) 再整体除以 16，得到 0~1 的输入
    targets = bunch.target.astype(np.int64)
    # 标签转成整型，形状 (1797,)
    return flattened, targets
    # 返回输入与标签


def split_dataset(
    flattened: np.ndarray, targets: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """把数据集分层切成训练集与测试集。

    参数：
        flattened：形状 (n, 64) 的输入。
        targets：形状 (n,) 的标签。
    返回：(X_train, X_test, y_train, y_test, train_index, test_index)
        train_index / test_index 是两组样本在原始数据里的下标，
        后面要“把同一批样本拿出来看表示”，靠这两个下标就能保证四宫格里画的是同一批点。

    为什么用 stratify：每个数字的样本数在 174~183 之间本来就很均衡，
    分层抽样进一步保证训练集、测试集里每个数字的比例都和整体一致，
    避免“某个数字恰好没进测试集”这种让小样本实验结论漂移的情况。
    """
    index = np.arange(flattened.shape[0])
    # 原始样本下标，0 到 1796
    X_train, X_test, y_train, y_test, train_index, test_index = train_test_split(
        flattened,
        targets,
        index,
        test_size=TEST_SIZE,
        stratify=targets,
        random_state=SPLIT_SEED,
    )
    # 一次性切出输入、标签、原始下标三样东西，保证三者的行完全对应
    return X_train, X_test, y_train, y_test, train_index, test_index
    # 返回划分结果


def compute_layer_metrics(
    net: DeepMLP,
    representations: list[np.ndarray],
    labels: np.ndarray,
    train_index: np.ndarray,
    test_index: np.ndarray,
    pca_coordinates: list[np.ndarray],
    tsne_coordinates: list[np.ndarray],
) -> list[dict]:
    """对四种表示逐一计算全部量化指标。

    参数：
        net：训练好的网络（这里只用它的结构信息，比如各层维度）。
        representations：长度 4 的列表 [input, h1, h2, h3]，形状为 (1797, d)。
        labels：形状 (1797,) 的真实标签。
        train_index：训练样本在全集中的下标。
        test_index：测试样本在全集中的下标。
        pca_coordinates：长度 4 的列表，每种表示的 PCA 二维投影坐标。
        tsne_coordinates：长度 4 的列表，每种表示的 t-SNE 二维投影坐标。
    返回：长度 4 的字典列表，每个字典是一层的全部指标。

    指标分三组：
      A 组（全部 1797 张，与四宫格图里的点是同一批）：
        silhouette、1-NN 留一、类内散度/类间中心距；
      B 组（只看测试集 450 张，网络训练时完全没见过）：
        silhouette、1-NN 留一、以训练集为参考的 1-NN；
      C 组（二维投影上重算的指标，也就是“图里肉眼看到的那个版本”）：
        PCA / t-SNE 两种投影各自的 silhouette 与“最近邻同色率”。
      B 组存在的意义：回答“深层团得紧会不会只是网络把训练样本背下来了”这个质疑。
      C 组存在的意义：把“图上看没看出团”变成数字，让图与指标表能互相印证。
    """
    rows = []
    # 每层一条记录
    for layer, representation in enumerate(representations):
        # 逐层计算
        row = {
            "name": REPRESENTATION_NAMES[layer],
            # 层名，与 PPT 标题一致
            "dim": int(representation.shape[1]),
            # 该层表示的维度
            "silhouette": silhouette_of(representation, labels),
            # 高维表示本身的轮廓系数（全部 1797 张）
            "one_nn_loo": leave_one_out_one_nn(representation, labels),
            # 高维表示上的 1-NN 留一准确率（全部 1797 张）
            "scatter_ratio": scatter_ratio(representation, labels),
            # 类内散度 / 类间中心距，越小越紧
            "silhouette_pca2d": silhouette_of(pca_coordinates[layer], labels),
            # PCA 二维投影坐标上的轮廓系数：用来检查“图上看到的递进”和“高维数字”是否同向
            "same_label_pca2d": same_label_nearest_neighbor_rate(pca_coordinates[layer], labels),
            # PCA 二维投影上“最近邻同色”的比例，直接对应肉眼对成团的判断
            "silhouette_tsne2d": silhouette_of(tsne_coordinates[layer], labels),
            # t-SNE 二维投影坐标上的轮廓系数
            "same_label_tsne2d": same_label_nearest_neighbor_rate(tsne_coordinates[layer], labels),
            # t-SNE 二维投影上“最近邻同色”的比例
            "silhouette_test": silhouette_of(representation[test_index], labels[test_index]),
            # 只在测试集 450 张上算的轮廓系数
            "one_nn_loo_test": leave_one_out_one_nn(representation[test_index], labels[test_index]),
            # 只在测试集 450 张内部做留一法的 1-NN 准确率
            "one_nn_test": split_one_nn(
                representation[train_index],
                labels[train_index],
                representation[test_index],
                labels[test_index],
            ),
            # 训练集当参考、测试集当考点的 1-NN 准确率
        }
        # 组装好这一层的指标字典
        rows.append(row)
        # 追加到结果列表
    return rows
    # 返回四层指标


def build_panels(
    coordinates: list[np.ndarray],
    rows: list[dict],
    labels: np.ndarray,
    projection: str,
) -> list[dict]:
    """把二维坐标与指标小字打包成 figure_representation_grid 需要的面板列表。

    参数：
        coordinates：长度 4 的二维坐标列表。
        rows：compute_layer_metrics 返回的四层指标。
        labels：形状 (1797,) 的真实标签。
        projection：这批坐标来自哪种降维，"pca" 或 "tsne"；面板小字写对应投影自己的指标，
                    保证“图上写的数”就是由图里这批点算出来的数，不会出现张冠李戴。
    返回：长度 4 的面板字典列表。
    """
    if projection not in ("pca", "tsne"):
        # 只认识这两种投影，别的拼法说明调用方写错了
        raise ValueError(f"projection 只能是 'pca' 或 'tsne'，收到 {projection!r}")
        # 直接报错，避免小字里写出一个牛头不对马嘴的数字
    panels = []
    # 逐层组装
    for layer in range(len(coordinates)):
        # 本实验固定 4 层
        note = (
            f"维度 {rows[layer]['dim']} → 2 维\n"
            f"{projection.upper()} silhouette {rows[layer][f'silhouette_{projection}2d']:+.3f}"
            f"　同类近邻率 {rows[layer][f'same_label_{projection}2d']:.3f}"
        )
        # 面板左下角的小字：这一层原本多少维，以及在这张图的二维坐标上重算的两个指标
        panels.append(
            {
                "title": REPRESENTATION_NAMES[layer],
                # 子图标题用 PPT 原文层名
                "coordinates": coordinates[layer],
                # 该层的二维投影坐标
                "labels": labels,
                # 颜色用的真实标签（四层共用同一份标签，保证颜色含义一致）
                "note": note,
                # 指标小字
            }
        )
        # 追加该面板
    return panels
    # 返回四个面板


def monotonic_report(rows: list[dict], key: str, bigger_is_better: bool) -> tuple[bool, str]:
    """检查某个指标是否随层数单调递进，并生成一句人话结论。

    参数：
        rows：四层指标。
        key：要检查的指标名。
        bigger_is_better：True 表示越大越好（轮廓系数、1-NN 准确率），False 表示越小越好（类内/类间比值）。
    返回：(是否单调, 结论文本)
        是否单调：True 表示从 input 到第 3 隐藏层全程没有回退；
        结论文本：直接写这一路上每一层相对上一层的升降，供报告和终端摘要使用。
    """
    values = [row[key] for row in rows]
    # 取该指标在四层上的取值
    steps = []
    # 记录每一步的变化
    monotonic = True
    # 先假设单调，发现回退再置 False
    for layer in range(1, len(values)):
        # 从第 1 隐藏层开始，逐层和上一层比
        delta = values[layer] - values[layer - 1]
        # 本层减去上一层
        improved = delta > 0 if bigger_is_better else delta < 0
        # 判断这一步算不算“变好”：越大越好的指标要涨，越小越好的指标要跌
        steps.append(f"{rows[layer]['name']} {delta:+.4f}{'（变好）' if improved else '（回退）'}")
        # 把这个变化写成人话
        if not improved:
            # 只要有一层回退，就不是全程单调
            monotonic = False
            # 记录结论
    summary = (
        f"{rows[0]['name']} {values[0]:.4f} → "
        + " → ".join(f"{rows[layer]['name']} {values[layer]:.4f}" for layer in range(1, len(values)))
        + "；逐层变化：" + "，".join(steps)
    )
    # 拼一句“从多少到多少、每步涨跌如何”的完整描述
    return monotonic, summary
    # 返回结论与描述


def write_metrics_report(
    path: str,
    rows: list[dict],
    history: list,
    config: dict,
    timings: dict,
) -> None:
    """把全部实验数字写进 results/metrics.txt。

    参数：
        path：输出文件路径。
        rows：四层指标。
        history：训练日志（EpochLog 列表）。
        config：本次运行的配置字典（种子、轮数、网络结构、耗时等）。
        timings：各阶段耗时（秒）。
    返回：无。

    为什么要把数字和正文分开写进一个纯文本文件：图会变、README 会改，
    但“这次到底跑出了什么数”需要一个不会被后续编辑污染的快照，
    而且终端里 grep 这个文件比翻截图快得多。
    """
    lines: list[str] = []
    # 逐行拼接报告内容
    lines.append("=" * 78)
    # 分隔线
    lines.append("手写数字四宫格：表示如何随层数变深而“十类成团”—— 实验记录")
    # 报告标题
    lines.append("=" * 78)
    # 分隔线
    lines.append("")
    # 空行

    lines.append("【一、运行环境】")
    # 分节标题
    lines.append(f"操作系统        : {platform.platform()}")
    # 操作系统
    lines.append(f"Python 版本     : {platform.python_version()}")
    # Python 版本
    lines.append(f"NumPy 版本      : {np.__version__}")
    # NumPy 版本
    lines.append(f"scikit-learn 版本: {sklearn.__version__}")
    # sklearn 版本
    lines.append("神经网络实现     : 纯 NumPy 手写 3 隐藏层 MLP（未使用 PyTorch / sklearn MLPClassifier）")
    # 说明实现方式
    lines.append(f"matplotlib 字体  : {config['font'] or '未找到中文字体'}")
    # 中文字体
    lines.append("")

    lines.append("【二、数据集】")
    # 分节标题
    lines.append("来源            : sklearn.datasets.load_digits（随 scikit-learn 内置，无需联网下载）")
    # 数据来源
    lines.append("定位            : MNIST 的迷你替代品——同样是 10 类手写数字，只是分辨率 8x8、规模 1797 张，")
    # 定位说明第一行
    lines.append("                  无法联网时用它复现同一结论，代价是细粒度笔画信息比 28x28 的 MNIST 少")
    # 定位说明第二行
    lines.append("样本总数        : 1797")
    # 样本总数
    lines.append("图像尺寸        : 8 x 8 灰度，像素原始取值 0~16，输入时除以 16 归一化到 0~1")
    # 图像规格
    lines.append("类别数          : 10（数字 0~9）")
    # 类别数
    lines.append(f"训练/测试划分   : test_size={TEST_SIZE}，stratify=y，random_state={SPLIT_SEED}")
    # 划分方式
    lines.append(f"训练集样本数    : {config['n_train']}")
    # 训练集大小
    lines.append(f"测试集样本数    : {config['n_test']}")
    # 测试集大小
    lines.append("划分口径        : 与姊妹项目 handcrafted_vs_learned 使用同一 random_state=42，样本集合完全一致")
    # 与姊妹项目对齐
    lines.append("")

    lines.append("【三、模型配置】")
    # 分节标题
    lines.append(f"网络结构        : {config['architecture']}")
    # 网络结构
    lines.append(f"         说明   : 输入 64 维像素 -> 3 个隐藏层（{config['hidden_dims']}）-> 10 类 softmax")
    # 逐层宽度
    lines.append("激活函数        : 隐藏层 ReLU，输出层 softmax")
    # 激活函数
    lines.append("损失函数        : 交叉熵（与 softmax 组合后梯度简化为 P - Y）")
    # 损失函数
    lines.append(f"优化器          : Adam，学习率 {config['learning_rate']}，L2 权重衰减 {config['l2']}（只作用于权重）")
    # 优化器配置
    lines.append(f"训练轮数        : {config['epochs']} 个 epoch，batch_size = {config['batch_size']}")
    # 训练轮数
    lines.append(f"参数总量        : {config['parameter_count']} 个")
    # 参数量
    lines.append(f"随机种子        : 权重初始化 seed={config['seed']}，样本打乱 seed={config['seed']}")
    # 随机种子
    lines.append("")

    lines.append("【四、最终分类准确率（网络自己的表现，与四宫格是同一个网络）】")
    # 分节标题
    lines.append(f"训练集准确率    : {history[-1].train_accuracy:.4f}（{history[-1].train_accuracy:.2%}）")
    # 训练集准确率
    lines.append(f"测试集准确率    : {history[-1].eval_accuracy:.4f}（{history[-1].eval_accuracy:.2%}）")
    # 测试集准确率
    lines.append(f"训练集交叉熵    : {history[-1].train_loss:.4f}")
    # 训练集损失
    lines.append(f"测试集交叉熵    : {history[-1].eval_loss:.4f}")
    # 测试集损失
    lines.append(f"验收线          : 测试准确率 >= {TARGET_ACCURACY:.2f}，本次{'通过' if history[-1].eval_accuracy >= TARGET_ACCURACY else '未通过'}")
    # 验收结论
    lines.append(f"最佳测试准确率  : {max(record.eval_accuracy for record in history):.4f}（训练过程中出现过的最好值）")
    # 训练过程中的最佳测试准确率
    lines.append("")

    lines.append("【五、核心量化指标：四种表示分别有多“成团”】")
    # 分节标题
    lines.append(f"样本口径        : 全部 1797 张，与四宫格图（图 1、图 2）里的点是同一批")
    # 口径说明
    lines.append("")
    table_a_headers = ["表示", "维度", "silhouette", "较上一层", "1-NN 留一", "类内/类间"]
    # 表 A 表头
    table_a_rows = []
    # 表 A 数据行
    for layer, row in enumerate(rows):
        # 逐层填表
        if layer == 0:
            # 输入层没有“上一层”
            delta_text = "—"
            # 用破折号占位
        else:
            # 其他层给出与上一层的差
            delta_text = f"{row['silhouette'] - rows[layer - 1]['silhouette']:+.4f}"
            # 差值带符号，正数表示更成团
        table_a_rows.append([
            row["name"],
            # 层名
            str(row["dim"]),
            # 维度
            f"{row['silhouette']:.4f}",
            # 轮廓系数
            delta_text,
            # 与上一层的差
            f"{row['one_nn_loo']:.4f}",
            # 1-NN 留一准确率
            f"{row['scatter_ratio']:.4f}",
            # 类内散度 / 类间中心距（越小越好）
        ])
        # 追加该行
    lines.extend(format_table(table_a_headers, table_a_rows, ["left", "right", "right", "right", "right", "right"]))
    # 渲染表 A
    lines.append("")
    lines.append("读法            : silhouette 越大越好（-1 ~ 1）；1-NN 留一越大越好；类内/类间越小越好。")
    # 指标读法
    lines.append("")

    lines.append("【五之二、二维投影上重算的指标（也就是“图里肉眼看到的那个版本”）】")
    # 分节标题
    lines.append("为什么单列：上面那张表算的是高维表示（64/128/32 维），而四宫格画的是压到 2 维之后的图。")
    # 动机第一行
    lines.append("            这里把两种投影各自重算一遍，用来证明“图上看到的团”和“高维数字的递进”是同一件事。")
    # 动机第二行
    lines.append("同类近邻率      : 在二维平面上离某个点最近的另一个点与它同色的比例——")
    # 指标解释第一行
    lines.append("                  颜色互相穿插时这个比例低，十类各抱一团时它接近 1，正是肉眼判断“成团”的依据。")
    # 指标解释第二行
    lines.append("")
    table_c_headers = ["表示", "PCA-2D silhouette", "PCA-2D 同类近邻率", "t-SNE silhouette", "t-SNE 同类近邻率"]
    # 表 C 表头
    table_c_rows = [
        [
            row["name"],
            # 层名
            f"{row['silhouette_pca2d']:.4f}",
            # PCA 二维投影的轮廓系数
            f"{row['same_label_pca2d']:.4f}",
            # PCA 二维投影的同类近邻率
            f"{row['silhouette_tsne2d']:.4f}",
            # t-SNE 二维投影的轮廓系数
            f"{row['same_label_tsne2d']:.4f}",
            # t-SNE 二维投影的同类近邻率
        ]
        for row in rows
    ]
    # 逐层填表
    lines.extend(format_table(table_c_headers, table_c_rows, ["left", "right", "right", "right", "right"]))
    # 渲染表 C
    lines.append("")
    pca2d_monotonic, pca2d_summary = monotonic_report(rows, "same_label_pca2d", bigger_is_better=True)
    # 检查 PCA 二维投影上的同类近邻率是否也逐层上升
    tsne2d_monotonic, tsne2d_summary = monotonic_report(rows, "same_label_tsne2d", bigger_is_better=True)
    # 检查 t-SNE 二维投影上的同类近邻率是否也逐层上升
    lines.append(f"PCA-2D 同类近邻率单调递进 : {'是' if pca2d_monotonic else '否'}")
    # 结论一
    lines.append(f"    {pca2d_summary}")
    # 明细
    lines.append(f"t-SNE 同类近邻率单调递进  : {'是' if tsne2d_monotonic else '否'}")
    # 结论二
    lines.append(f"    {tsne2d_summary}")
    # 明细
    lines.append("")

    lines.append("【六、只看测试集（450 张，网络训练时完全没见过）】")
    # 分节标题
    lines.append("为什么要单独列这一组：回答“深层团得紧，会不会只是网络把训练样本背下来了”。")
    # 动机
    lines.append("")
    table_b_headers = ["表示", "维度", "silhouette(450)", "1-NN 留一(450)", "1-NN 以训练集为参考"]
    # 表 B 表头
    table_b_rows = [
        [
            row["name"],
            # 层名
            str(row["dim"]),
            # 维度
            f"{row['silhouette_test']:.4f}",
            # 测试集内部算的轮廓系数
            f"{row['one_nn_loo_test']:.4f}",
            # 测试集内部留一法的 1-NN
            f"{row['one_nn_test']:.4f}",
            # 训练集当参考的 1-NN
        ]
        for row in rows
    ]
    # 逐层填表
    lines.extend(format_table(table_b_headers, table_b_rows, ["left", "right", "right", "right", "right"]))
    # 渲染表 B
    lines.append("")

    lines.append("【七、递进性检查】")
    # 分节标题
    silhouette_monotonic, silhouette_summary = monotonic_report(rows, "silhouette", bigger_is_better=True)
    # 检查轮廓系数是否逐层上升
    one_nn_monotonic, one_nn_summary = monotonic_report(rows, "one_nn_loo", bigger_is_better=True)
    # 检查 1-NN 留一准确率是否逐层上升
    ratio_monotonic, ratio_summary = monotonic_report(rows, "scatter_ratio", bigger_is_better=False)
    # 检查类内/类间比值是否逐层下降
    lines.append(f"silhouette 单调递进 : {'是' if silhouette_monotonic else '否'}")
    # 结论一
    lines.append(f"    {silhouette_summary}")
    # 明细
    lines.append(f"1-NN 留一单调递进  : {'是' if one_nn_monotonic else '否'}")
    # 结论二
    lines.append(f"    {one_nn_summary}")
    # 明细
    lines.append(f"类内/类间单调下降   : {'是' if ratio_monotonic else '否'}")
    # 结论三
    lines.append(f"    {ratio_summary}")
    # 明细
    lines.append("")
    if silhouette_monotonic and one_nn_monotonic and ratio_monotonic and pca2d_monotonic and tsne2d_monotonic:
        # 五个指标（高维三项 + 两种二维投影各一项）全部同向递进，这是最强的情况
        lines.append("结论：高维表示的三类指标与两种二维投影上的同类近邻率全部随层数单调递进，")
        # 结论第一行
        lines.append("      「input 十类混杂 → 第 3 隐藏层十类各抱一团」在数字上和图上同时成立。")
        # 结论第二行
    elif silhouette_monotonic and one_nn_monotonic and ratio_monotonic:
        # 高维三项单调，但某一种二维投影上的同类近邻率出现回退
        lines.append("结论：高维表示的三类指标全部单调递进，但至少有一种二维投影上的同类近邻率出现回退。")
        # 如实说明
        lines.append("      二维投影是有损的（高维压到 2 维必然丢信息），投影方法本身的癖好可以掩盖掉一部分递进，")
        # 原因一
        lines.append("      所以“图上看到什么”与“高维表示有多好”不必完全同步；两处结论冲突时以高维指标为准。")
        # 原因二
    else:
        # 只要有一项回退，就如实写出是哪一项、哪一层，并给出可能的原因
        lines.append("结论：存在非单调的指标，逐层递进并非全部指标都成立；具体回退位置见上面各节明细。")
        # 如实说明
        lines.append("      常见原因是：① 更深的层用 ReLU，输出非负且稀疏，会在欧氏距离下改变类内散布的尺度；")
        # 原因一
        lines.append("      ② 中间的隐藏层可以保留与分类无关的信息（后面几层还有机会把它去掉）；")
        # 原因二
        lines.append("      ③ 1-NN 只看最近邻一个点，个别离群样本就能改变数值。")
        # 原因三
    lines.append("")

    lines.append("【八、耗时】")
    # 分节标题
    for stage, seconds in timings.items():
        # 逐项列出
        lines.append(f"{pad_cell(stage, 20)}: {seconds:.2f} 秒")
        # 阶段名按显示宽度左对齐（中文按 2 格算，否则整列会错位），耗时保留两位小数
    total_label = "合计"
    # 合计行的标签
    lines.append(f"{pad_cell(total_label, 20)}: {sum(timings.values()):.2f} 秒")
    # 合计
    lines.append("")

    lines.append("【九、输出文件】")
    # 分节标题
    lines.append("results/01_layerwise_pca.png   图 1  四宫格主图（PCA 降到 2 维，复刻 PPT 原页）")
    # 图 1
    lines.append("results/02_layerwise_tsne.png  图 2  四宫格对照（t-SNE 降到 2 维）")
    # 图 2
    lines.append("results/03_training_curve.png  图 3  训练曲线（损失与准确率）")
    # 图 3
    lines.append("results/04_layer_metrics.png   图 4  量化指标随层数变化")
    # 图 4
    lines.append("results/metrics.txt            本文件")
    # 本文件
    lines.append("")

    lines.append("【十、复现命令】")
    # 分节标题
    lines.append("cd /f/kimi/mnist_layerwise")
    # 切到项目目录
    lines.append("/c/msys64/ucrt64/bin/python main.py --epochs 300 --seed 42")
    # 复现命令
    lines.append("")

    with open(path, "w", encoding="utf-8") as handle:
        # 以 UTF-8 打开写模式；Windows 下不指定编码会写成 GBK，别的工具读起来会乱码
        handle.write("\n".join(lines) + "\n")
        # 逐行写入并保证文件末尾有换行
    return
    # 写盘完成


def parse_arguments() -> argparse.Namespace:
    """解析命令行参数。

    参数：无。
    返回：argparse.Namespace，含 epochs / batch-size / seed / learning-rate / l2 / output-dir / seed-tsne。
    """
    parser = argparse.ArgumentParser(description="复现李宏毅《Why Deep》PPT 的 MNIST 四宫格页（sklearn digits 迷你版）")
    # 构造参数解析器
    parser.add_argument("--epochs", type=int, default=300, help="训练轮数，默认 300")
    # 训练轮数
    parser.add_argument("--batch-size", type=int, default=64, help="mini-batch 大小，默认 64")
    # 批大小
    parser.add_argument("--seed", type=int, default=42, help="权重初始化与样本打乱的随机种子，默认 42")
    # 随机种子
    parser.add_argument("--learning-rate", type=float, default=1e-3, help="Adam 学习率，默认 1e-3")
    # 学习率
    parser.add_argument("--l2", type=float, default=1e-4, help="权重衰减系数（只作用于权重），默认 1e-4")
    # 权重衰减
    parser.add_argument("--output-dir", type=str, default=RESULTS_DIR, help="输出目录，默认 results")
    # 输出目录
    parser.add_argument("--perplexity", type=float, default=30.0, help="t-SNE 的 perplexity，默认 30")
    # t-SNE 近邻规模
    return parser.parse_args()
    # 返回解析结果


def main() -> int:
    """主流程：训练、量化、画图、写报告。

    参数：无（配置全部来自命令行参数）。
    返回：进程退出码；测试准确率达标返回 0，未达标返回 1，便于后台任务自动判断成败。
    """
    arguments = parse_arguments()
    # 读取命令行配置
    timings: dict[str, float] = {}
    # 各阶段耗时
    font_name = setup_chinese_font()
    # 注册中文字体，后面的图里会出现中文标题
    if font_name is None:
        # 找不到中文字体只提示，不中断——图还是会生成，只是中文可能显示为方块
        print("警告：未找到系统中文字体，图中的中文可能显示为方块", flush=True)
        # 提示用户
    else:
        # 正常情况打印用的是哪个字体，便于排查
        print(f"中文字体：{font_name}", flush=True)
        # 输出字体名

    start = time.perf_counter()
    # 开始计时
    flattened, targets = load_dataset()
    # 载入数据
    X_train, X_test, y_train, y_test, train_index, test_index = split_dataset(flattened, targets)
    # 划分训练/测试集
    timings["载入与划分"] = time.perf_counter() - start
    # 记录这一阶段耗时
    print(f"数据：{flattened.shape[0]} 张 8x8 手写数字，训练集 {X_train.shape[0]}，测试集 {X_test.shape[0]}", flush=True)
    # 打印数据规模

    network = DeepMLP(seed=arguments.seed, learning_rate=arguments.learning_rate, l2=arguments.l2)
    # 构造 3 隐藏层网络
    print(
        f"网络：{' -> '.join(str(size) for size in network.layer_sizes)}，参数 {network.parameter_count()} 个",
        flush=True,
    )
    # 打印结构
    start = time.perf_counter()
    # 重新开始计时
    history = network.fit(
        X_train,
        y_train,
        epochs=arguments.epochs,
        batch_size=arguments.batch_size,
        eval_set=(X_test, y_test),
        seed=arguments.seed,
        verbose=True,
    )
    # 训练网络，每个 epoch 顺带在测试集上评估一次（只用于观察，不参与训练决策）
    timings["训练"] = time.perf_counter() - start
    # 记录训练耗时
    test_accuracy = network.accuracy(X_test, y_test)
    # 训练结束后再算一次测试集准确率，作为最终读数
    print(f"训练完成：测试集准确率 {test_accuracy:.4f}（{test_accuracy:.2%}），训练耗时 {timings['训练']:.2f} 秒", flush=True)
    # 打印最终准确率

    start = time.perf_counter()
    # 重新开始计时
    representations = network.representations(flattened)
    # 取出全部 1797 张样本在四层里的表示，这就是四宫格的数据源
    pca_coordinates = [pca_2d(representation, seed=0) for representation in representations]
    # 四种表示各自 PCA 降到 2 维
    tsne_coordinates = [
        tsne_2d(representation, seed=arguments.seed, perplexity=arguments.perplexity)
        for representation in representations
    ]
    # 四种表示各自 t-SNE 降到 2 维（放在指标之前算，因为二维投影上的指标要绑定这两组坐标）
    layer_rows = compute_layer_metrics(
        network,
        representations,
        targets,
        train_index,
        test_index,
        pca_coordinates,
        tsne_coordinates,
    )
    # 算全部量化指标（高维 + 两种二维投影）
    timings["降维与指标计算"] = time.perf_counter() - start
    # 记录耗时
    print(f"表示维度：{[representation.shape[1] for representation in representations]}", flush=True)
    # 打印四种表示的维度
    for row in layer_rows:
        # 逐层打印指标，方便后台日志直接读
        print(
            f"  {row['name']:<12} silhouette {row['silhouette']:+.4f} | 1-NN 留一 {row['one_nn_loo']:.4f} "
            f"| 类内/类间 {row['scatter_ratio']:.4f} | PCA-2D 同类近邻率 {row['same_label_pca2d']:.4f} "
            f"| t-SNE 同类近邻率 {row['same_label_tsne2d']:.4f}",
            flush=True,
        )
        # 一行写完一层的关键数字

    os.makedirs(arguments.output_dir, exist_ok=True)
    # 确保输出目录存在（不存在就创建，已存在也不报错）
    start = time.perf_counter()
    # 重新开始计时
    figure_representation_grid(
        build_panels(pca_coordinates, layer_rows, targets, "pca"),
        os.path.join(arguments.output_dir, "01_layerwise_pca.png"),
        "图 1  同一批手写数字样本在四层表示空间中的分布（PCA 降到 2 维）",
        "全部 1797 张，颜色 = 真实数字标签，圆圈里的大数字 = 该类在二维平面上的中心。"
        "按 PPT 原页的版式排列：input 面板十类混杂，到第 3 隐藏层十个数字各自抱团",
    )
    # 画图 1：四宫格主图
    timings["图 1：四宫格 PCA"] = time.perf_counter() - start
    # 记录耗时

    start = time.perf_counter()
    # 重新开始计时
    figure_representation_grid(
        build_panels(tsne_coordinates, layer_rows, targets, "tsne"),
        os.path.join(arguments.output_dir, "02_layerwise_tsne.png"),
        "图 2  同一批样本的 t-SNE 四宫格（与图 1 对照）",
        "t-SNE 是非线性降维，只保留“谁和谁是邻居”，团会比 PCA 更炸开；"
        "它用来排除一种质疑：图 1 的递进会不会只是 PCA 这种线性投影造成的错觉",
    )
    # 画图 2：四宫格 t-SNE 对照
    timings["图 2：四宫格 t-SNE"] = time.perf_counter() - start
    # 记录耗时

    start = time.perf_counter()
    # 重新开始计时
    figure_training_curve(history, os.path.join(arguments.output_dir, "03_training_curve.png"))
    # 画图 3：训练曲线
    figure_layer_metrics(
        [row["name"] for row in layer_rows],
        layer_rows,
        os.path.join(arguments.output_dir, "04_layer_metrics.png"),
    )
    # 画图 4：指标随层数变化
    timings["图 3 / 图 4"] = time.perf_counter() - start
    # 记录两张图的耗时

    config = {
        "font": font_name,
        # 中文字体名
        "n_train": int(X_train.shape[0]),
        # 训练集大小
        "n_test": int(X_test.shape[0]),
        # 测试集大小
        "architecture": "64 -> 128(ReLU) -> 64(ReLU) -> 32(ReLU) -> 10(softmax)",
        # 结构的一行描述
        "hidden_dims": "128, 64, 32",
        # 三个隐藏层的宽度
        "epochs": arguments.epochs,
        # 训练轮数
        "batch_size": arguments.batch_size,
        # 批大小
        "seed": arguments.seed,
        # 随机种子
        "learning_rate": arguments.learning_rate,
        # 学习率
        "l2": arguments.l2,
        # 权重衰减
        "parameter_count": network.parameter_count(),
        # 参数量
    }
    # 汇总配置，写进报告的第三节
    write_metrics_report(os.path.join(arguments.output_dir, "metrics.txt"), layer_rows, history, config, timings)
    # 写 results/metrics.txt
    print(f"已写出 {arguments.output_dir}/metrics.txt 与 4 张 PNG", flush=True)
    # 提示产出位置

    silhouette_monotonic, _ = monotonic_report(layer_rows, "silhouette", bigger_is_better=True)
    # 检查轮廓系数是否逐层上升
    one_nn_monotonic, _ = monotonic_report(layer_rows, "one_nn_loo", bigger_is_better=True)
    # 检查 1-NN 留一是否逐层上升
    pca2d_monotonic, _ = monotonic_report(layer_rows, "same_label_pca2d", bigger_is_better=True)
    # 检查 PCA 二维投影上的同类近邻率是否逐层上升
    tsne2d_monotonic, _ = monotonic_report(layer_rows, "same_label_tsne2d", bigger_is_better=True)
    # 检查 t-SNE 二维投影上的同类近邻率是否逐层上升
    print(
        f"递进性：silhouette 单调={'是' if silhouette_monotonic else '否'}，"
        f"1-NN 留一单调={'是' if one_nn_monotonic else '否'}，"
        f"PCA-2D 同类近邻率单调={'是' if pca2d_monotonic else '否'}，"
        f"t-SNE 同类近邻率单调={'是' if tsne2d_monotonic else '否'}",
        flush=True,
    )
    # 打印递进性结论
    print(f"总耗时：{sum(timings.values()):.2f} 秒", flush=True)
    # 打印总耗时
    if test_accuracy < TARGET_ACCURACY:
        # 没达到验收线就返回非 0，让后台任务明确失败，而不是“看起来跑完了”
        print(f"验收未通过：测试准确率 {test_accuracy:.4f} < {TARGET_ACCURACY}", flush=True)
        # 打印失败原因
        return 1
        # 非 0 退出码
    print(f"验收通过：测试准确率 {test_accuracy:.4f} >= {TARGET_ACCURACY}", flush=True)
    # 达标提示
    return 0
    # 正常退出


if __name__ == "__main__":
    # 只有直接运行本文件时才执行主流程（被 import 时不会自动跑）
    sys.exit(main())
    # 把 main 的返回值当作进程退出码
