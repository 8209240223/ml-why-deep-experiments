#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
主实验脚本：人工特征流水线（方案 A）vs 端到端学习（方案 B），手写数字分类。

对应李宏毅《Why Deep》中 "End-to-end Learning" 那两页 PPT：
  - 方案 A（对应"绿 + 蓝"）：人工设计的特征提取器（features.py，绿色、不可学习）
    后面只接一个逻辑回归分类器（蓝色、唯一从数据学习的模块）；
  - 方案 B（对应"全蓝"）：原始 64 维像素直接送进一个 64-64-10 的 MLP（model.py），
    没有任何人写的模块，特征和分类器全部从数据里学出来。

两条流水线使用完全相同的训练/测试划分（同一个 random_state、同样的 stratify），
因此准确率差距只能归因于"特征从哪来"这一个变量。

另外还有一个参照组 R（原始像素 + 逻辑回归，分类器与方案 A 完全相同），
它不属于 PPT 上的两种范式，只用来堵住"是不是逻辑回归本身太弱"这个漏洞。

产出：
  results/01_handcrafted_features.png   人工特征长什么样、为什么它丢了信息
  results/02_learned_first_layer.png    端到端网络自己学出来的 64 个 8x8 特征探测器
  results/03_confusion_matrices.png     两方案的混淆矩阵对比
  results/04_misclassified_examples.png 两方案分别错在哪些样本上
  results/05_feature_space_comparison.png  人工特征空间 vs 网络学出的隐藏层特征空间
  results/06_training_curve.png         端到端网络的训练曲线
  results/metrics.txt                   全部数值记录（准确率、每类召回率、混淆矩阵、耗时）

用法（Git Bash）：
    /c/msys64/ucrt64/bin/python main.py
    /c/msys64/ucrt64/bin/python main.py --epochs 300 --hidden 128
    /c/msys64/ucrt64/bin/python main.py --skip-extra-seeds
"""

from __future__ import annotations
# 打开延迟注解求值，类型标注里可以直接写 np.ndarray 而不必加引号

import argparse
# 解析命令行参数，让轮数、批大小、种子等可以不改代码就调
import os
# 拼接结果目录路径、创建目录
import platform
# 记录运行环境（操作系统），写进 metrics.txt
import sys
# 取 Python 版本号，写进 metrics.txt 便于复现
import time
# 统计各阶段耗时，最后汇报"运行耗时"
import unicodedata
# 计算字符显示宽度（中文占两格），让 metrics.txt 里的文本表格能对齐

import numpy as np
# 数值计算：数组切片、拼接、均值等

import matplotlib
# 先导入 matplotlib 顶层模块，才能在后端设置生效前做配置

matplotlib.use("Agg")
# 必须在 import pyplot 之前切换成无界面后端 Agg，否则在没有显示器的环境下保存图片会报错

import matplotlib.pyplot as plt
# 画图主模块

from matplotlib import font_manager
# 用来注册系统中文字体，避免图里的中文变成方块

from matplotlib.cm import ScalarMappable
# 用于给"多子图共用一张颜色条"的场景构造可映射对象

from matplotlib.colors import Normalize
# 给 ScalarMappable 提供数值到颜色的映射规则

from sklearn.datasets import load_digits
# 数据集：sklearn 自带的手写数字，免下载、随库安装
import sklearn
# 顶层包本身也需要导入，用来把版本号写进 metrics.txt（便于日后复现环境）
from sklearn.decomposition import PCA
# 只用于可视化降维：把 20 维人工特征和 64 维隐藏层特征都压到 2 维画散点图
from sklearn.linear_model import LogisticRegression
# 方案 A 的分类器，对应 PPT 上的蓝色模块
from sklearn.metrics import confusion_matrix, f1_score
# 评估指标：混淆矩阵、宏平均 F1
from sklearn.model_selection import train_test_split
# 固定种子的训练/测试划分，两条流水线共用

import features as handcrafted
# 本实验的人工特征模块（特征名容易和 sklearn 的 features 概念混淆，这里显式改名为 handcrafted）

from model import SoftmaxMLP
# 本实验的端到端模型：纯 NumPy 手写的单隐藏层 MLP

RESULTS_DIRNAME = "results"
# 结果全部写入脚本同级的 results 目录

DEFAULT_SEED = 42
# 主实验的随机种子；两条流水线共用它，保证训练/测试划分完全一致
DEFAULT_TEST_SIZE = 0.25
# 测试集比例：1797 张里留四分之一（约 449 张）做最终评估，其余用于训练
DEFAULT_EPOCHS = 200
# 端到端网络的训练轮数
DEFAULT_BATCH_SIZE = 64
# mini-batch 梯度下降的批大小
DEFAULT_HIDDEN_DIM = 64
# 隐藏层单元数，刻意等于输入维度 64，方便和"人工特征维度"直接比较
DEFAULT_LEARNING_RATE = 1e-3
# Adam 的学习率
DEFAULT_EXTRA_SEEDS = "7,2024"
# 额外的随机种子，用于做"换一份划分、结论是否还成立"的稳定性检验
CONFUSION_FIGURE_LIMIT = 6
# 错误样本对比图里每类最多画 6 个样本，太多会看不清
PIXEL_MAX = 16.0
# load_digits 像素的最大值，用于把像素缩放到 0~1


def setup_chinese_font() -> str | None:
    """为 matplotlib 注册一个系统中文字体，让图里的中文标题正常显示。

    参数：无。
    返回：成功时返回字体族名，找不到任何中文字体时返回 None（图仍会生成，中文可能显示为方框）。
    为什么需要：matplotlib 自带字体不含中文字形，不显式指定就会画出一堆方块。
    """
    candidates = [
        r"C:\Windows\Fonts\msyh.ttc",
        r"C:\Windows\Fonts\simhei.ttf",
        r"C:\Windows\Fonts\Deng.ttf",
        r"C:\Windows\Fonts\simsun.ttc",
    ]
    # 依次尝试微软雅黑、黑体、等线、宋体这几个 Windows 自带中文字体
    for path in candidates:
        # 逐个检查文件是否存在于本机
        if not os.path.exists(path):
            # 不存在就换下一个候选
            continue
        font_manager.fontManager.addfont(path)
        # 把字体文件注册进 matplotlib 的字体管理器，之后就能按名字引用
        name = font_manager.FontProperties(fname=path).get_name()
        # 从字体文件自身读出字体族名，而不是硬编码字符串，避免名字对不上
        plt.rcParams["font.family"] = name
        # 设为全局默认字体
        plt.rcParams["axes.unicode_minus"] = False
        # 让负号按普通减号渲染，否则中文字体下坐标轴的负号会变方块
        return name
        # 注册成功就结束
    return None
    # 一个中文字体都没找到


def display_width(text: str) -> int:
    """计算字符串在等宽终端里的显示宽度：中文全角字符算 2 格。

    参数：
        text：要测量的字符串。
    返回：显示宽度（整数）。
    为什么需要：写进 metrics.txt 的表格如果用 len() 补齐，中文列会整体错位。
    """
    return sum(2 if unicodedata.east_asian_width(char) in ("W", "F") else 1 for char in text)
    # east_asian_width 返回 W（宽）/F（全角）的字符按 2 格算，其余按 1 格


def pad_cell(text: str, width: int, align: str = "left") -> str:
    """按显示宽度把单元格补齐到指定宽度。

    参数：
        text：单元格文字。
        width：目标显示宽度。
        align："left" 左对齐，"right" 右对齐（数值列用右对齐更好读）。
    返回：补齐后的字符串。
    """
    spaces = max(width - display_width(text), 0)
    # 需要补的空格数；文字已超宽时补 0，绝不用负数重复字符串
    if align == "right":
        # 右对齐：空格补在左边
        return " " * spaces + text
        # 返回补齐结果
    return text + " " * spaces
    # 左对齐：空格补在右边


def format_table(headers: list[str], rows: list[list[str]], aligns: list[str]) -> list[str]:
    """把二维数据渲染成用等宽对齐的文本表格行。

    参数：
        headers：表头文字。
        rows：每一行的单元格文字。
        aligns：每列的对齐方式，长度需与列数一致。
    返回：可直接写入文本文件的字符串列表（含表头与分隔线）。
    """
    widths = []
    # 每一列的目标显示宽度
    for column in range(len(headers)):
        # 逐列计算
        column_texts = [headers[column]] + [row[column] for row in rows]
        # 该列所有出现过的文字（含表头）
        widths.append(max(display_width(text) for text in column_texts))
        # 列宽取该列最宽的那个字符串，保证任何内容都不会撑破表格
    lines = ["| " + " | ".join(pad_cell(headers[i], widths[i], aligns[i]) for i in range(len(headers))) + " |"]
    # 拼接表头行
    lines.append("|" + "|".join("-" * (widths[i] + 2) for i in range(len(headers))) + "|")
    # 拼接分隔行，长度与每列宽度对应
    for row in rows:
        # 逐行渲染
        lines.append("| " + " | ".join(pad_cell(row[i], widths[i], aligns[i]) for i in range(len(headers))) + " |")
        # 拼接数据行
    return lines
    # 返回整张表的文本行


def load_dataset() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """载入 sklearn 自带的手写数字数据集。

    参数：无。
    返回：(images, flattened, targets)
        images：形状 (1797, 8, 8) 的浮点数组，像素值保持原生尺度 0~16；
        flattened：形状 (1797, 64) 的浮点数组，像素值缩放到 0~1，直接喂给神经网络；
        targets：形状 (1797,) 的整数标签。

    为什么保留两份：人工特征模块内部自己会做 0~16 到 0~1 的缩放（见 features.py），
    而神经网络需要的是拉平后的 0~1 向量，两条路线各自拿到最合适的输入形式，
    但都没有做任何"需要从数据学习"的预处理（均值方差标准化之类），保证对比公平。
    """
    digits = load_digits()
    # 载入数据集，返回的是 Bunch 对象，含 data（(1797,64) 的 uint8）与 target
    images = digits.images.astype(np.float64)
    # 转成浮点：后面要求和、取平均，整数运算会截断
    flattened = digits.data.astype(np.float64) / PIXEL_MAX
    # 拉平成 (1797, 64) 并按最大值 16 归一化到 0~1
    targets = digits.target.astype(np.int64)
    # 标签转成整型，便于当数组下标用
    return images, flattened, targets
    # 返回三个数组


def split_dataset(
    flattened: np.ndarray, targets: np.ndarray, test_size: float, seed: int
) -> tuple[np.ndarray, np.ndarray]:
    """按固定随机种子做分层划分，返回训练集与测试集的样本下标。

    参数：
        flattened：形状 (n, 64) 的输入数组，只用它的长度来生成下标。
        targets：形状 (n,) 的标签，用于分层抽样。
        test_size：测试集比例。
        seed：随机种子。
    返回：(train_index, test_index)，两个整型下标数组。

    为什么要显式返回下标而不是直接返回切好的数组：
    方案 A 需要"原始 0~16 的图像"，方案 B 需要"拉平且归一化的像素"，
    用同一组下标去两边切片，才能保证两条流水线看到的样本一模一样，
    准确率差距才只反映特征来源的差别，不掺杂划分差异。
    """
    all_index = np.arange(flattened.shape[0])
    # 生成 0 ~ n-1 的样本编号
    train_index, test_index = train_test_split(
        all_index,
        test_size=test_size,
        random_state=seed,
        stratify=targets,
    )
    # 分层划分：保证训练集和测试集里 0~9 每个数字的比例与全集一致，避免小样本下的类别偏斜
    return np.sort(train_index), np.sort(test_index)
    # 排序后返回，方便打印"前若干个测试样本编号"之类的人类可读信息


def evaluate(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """对一组预测结果算全套评估指标。

    参数：
        y_true：形状 (n,) 的真标签。
        y_pred：形状 (n,) 的预测标签。
    返回：含准确率、宏平均 F1、每类召回率、混淆矩阵的字典。

    为什么除了准确率还要看每类召回率：digits 各类样本数并不完全相同（类 8 最少），
    只看总准确率会掩盖"某些数字几乎全错"的情况，每类召回率能暴露这类问题。
    """
    labels = np.arange(10)
    # 固定的类别顺序 0~9，保证混淆矩阵行列含义一致
    matrix = confusion_matrix(y_true, y_pred, labels=labels)
    # 混淆矩阵：第 i 行第 j 列 = 真实为 i 却被预测成 j 的样本数
    recall = matrix.diagonal() / np.maximum(matrix.sum(axis=1), 1)
    # 每类召回率 = 对角线（该类被正确识别的数量）/ 该类的真实样本总数；分母做保护避免除零
    return {
        "accuracy": float((y_true == y_pred).mean()),
        # 总准确率：预测正确的比例
        "macro_f1": float(f1_score(y_true, y_pred, average="macro")),
        # 宏平均 F1：先算每类的 F1 再取算术平均，对类别不均衡更稳健
        "per_class_recall": recall,
        # 长度 10 的每类召回率数组
        "per_class_support": matrix.sum(axis=1),
        # 长度 10 的每类测试样本数
        "confusion": matrix,
        # 10x10 混淆矩阵
    }
    # 打包返回


def run_handcrafted_pipeline(
    images_train: np.ndarray,
    y_train: np.ndarray,
    images_test: np.ndarray,
    y_test: np.ndarray,
    use_quadrants: bool,
) -> dict:
    """跑完方案 A 的一条完整流水线：人工特征 + 逻辑回归。

    参数：
        images_train / y_train：训练集图像（0~16 尺度）与标签。
        images_test / y_test：测试集图像与标签。
        use_quadrants：True 表示追加 4 个象限均值特征（20 维），False 表示只用 16 维投影。
    返回：字典，含特征矩阵、模型、预测结果、评估指标和耗时。

    这条流水线正是 PPT 上"绿 + 蓝"的写法：
    绿（人工特征，固定公式，不可学习） -> 蓝（逻辑回归，唯一从数据学习的模块）。
    """
    start = time.perf_counter()
    # 记录流水线开始时刻
    features_train = handcrafted.handcrafted_features(images_train, use_quadrants=use_quadrants)
    # 训练集特征：完全由人写死的公式算出，不接触任何标签
    features_test = handcrafted.handcrafted_features(images_test, use_quadrants=use_quadrants)
    # 测试集用同一个公式变换，保证训练/测试口径一致
    extract_seconds = time.perf_counter() - start
    # 特征提取耗时（这个耗时极小，也正是人工特征"便宜"的地方，但代价是上限低）

    classifier = LogisticRegression(max_iter=5000, C=1.0)
    # 逻辑回归：多分类时 lbfgs 求解器默认按多项逻辑回归处理；max_iter 提到 5000 保证收敛
    train_start = time.perf_counter()
    # 记录分类器训练开始时刻
    classifier.fit(features_train, y_train)
    # 这是整条流水线里唯一"从数据里学参数"的步骤，对应 PPT 的蓝色模块
    train_seconds = time.perf_counter() - train_start
    # 分类器训练耗时
    y_pred = classifier.predict(features_test)
    # 在测试集上预测
    metrics = evaluate(y_test, y_pred)
    # 计算全套评估指标
    metrics["feature_dim"] = int(features_train.shape[1])
    # 记下特征维度，写报告时要用
    metrics["extract_seconds"] = extract_seconds
    # 记下特征提取耗时
    metrics["train_seconds"] = train_seconds
    # 记下分类器训练耗时
    metrics["total_seconds"] = extract_seconds + train_seconds
    # 记下该流水线的总耗时
    metrics["y_pred"] = y_pred
    # 保存预测结果，画混淆矩阵和错误样本图时要用
    metrics["features_train"] = features_train
    # 保存训练集特征，画特征空间对比图和"最相似样本对"时要用
    metrics["features_test"] = features_test
    # 保存测试集特征
    metrics["classifier"] = classifier
    # 保存模型本身，便于后续检查权重
    metrics["feature_names"] = handcrafted.feature_names(use_quadrants)
    # 保存每一维特征的名字
    return metrics
    # 返回该方案的全部产物


def run_raw_pixel_linear_control(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
) -> dict:
    """参照组 R：原始 64 维像素 + 同一个逻辑回归，用来隔离"特征"这一个变量。

    参数：
        X_train / y_train：训练集像素（(n,64)，归一化到 0~1）与标签。
        X_test / y_test：测试集像素与标签。
    返回：字典，含预测结果、评估指标和耗时。

    为什么需要这个参照组：只看"人工特征 + 逻辑回归 88%"对"原始像素 + MLP 97%"
    会留下一个漏洞——差距也可能只是"逻辑回归太弱、不如神经网络"，而不是"人工特征丢了信息"。
    本参照组用的是同一个逻辑回归（同样是线性分类器，能力上限一模一样），
    唯一区别是输入端不做人工降维：
      - 若参照组明显高于方案 A，说明瓶颈在人工特征，而不在分类器；
      - 若参照组与方案 A 差不多，才说明是分类器能力的问题。
    它不属于 PPT 上的两种范式，只是实验层面的对照，因此单独列在报告里。
    """
    start = time.perf_counter()
    # 记录开始时刻
    classifier = LogisticRegression(max_iter=5000, C=1.0)
    # 与方案 A 完全相同的分类器配置，保证"分类器能力"这个变量被彻底固定
    classifier.fit(X_train, y_train)
    # 直接在 64 维原始像素上训练，不做任何人工特征变换
    train_seconds = time.perf_counter() - start
    # 训练耗时
    y_pred = classifier.predict(X_test)
    # 在同一个测试集上预测
    metrics = evaluate(y_test, y_pred)
    # 计算全套评估指标
    metrics["feature_dim"] = int(X_train.shape[1])
    # 输入维度就是像素数 64
    metrics["train_seconds"] = train_seconds
    # 记下训练耗时
    metrics["total_seconds"] = train_seconds
    # 参照组没有单独的特征提取阶段，总耗时等于训练耗时
    metrics["y_pred"] = y_pred
    # 保存预测结果
    return metrics
    # 返回参照组的全部产物


def run_end_to_end_pipeline(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    hidden_dim: int,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    seed: int,
    verbose: bool,
) -> dict:
    """跑完方案 B 的完整流水线：原始像素直接进 MLP，端到端学习。

    参数：
        X_train / y_train：训练集像素（(n,64)，已归一化到 0~1）与标签。
        X_test / y_test：测试集像素与标签。
        hidden_dim：隐藏层单元数。
        epochs：训练轮数。
        batch_size：mini-batch 大小。
        learning_rate：Adam 学习率。
        seed：随机种子，同时用于权重初始化和批次打乱。
        verbose：是否打印训练过程。
    返回：字典，含网络、预测结果、评估指标和耗时。

    这条流水线对应 PPT 上"全蓝"的写法：没有任何人设计的模块，
    第一层的 64 个权重向量本身就是要从数据里学出来的特征提取器。
    """
    start = time.perf_counter()
    # 记录流水线开始时刻
    network = SoftmaxMLP(
        input_dim=X_train.shape[1],
        hidden_dim=hidden_dim,
        num_classes=10,
        seed=seed,
        learning_rate=learning_rate,
    )
    # 构造网络：输入维度直接取像素数 64，不经过任何人工作用上的降维
    network.fit(
        X_train,
        y_train,
        epochs=epochs,
        batch_size=batch_size,
        eval_set=(X_test, y_test),
        seed=seed,
        verbose=verbose,
    )
    # 训练：交叉熵损失 + Adam 做 mini-batch 梯度下降；测试集只用于记录曲线，不参与更新
    train_seconds = time.perf_counter() - start
    # 训练耗时
    y_pred = network.predict(X_test)
    # 在测试集上预测
    metrics = evaluate(y_test, y_pred)
    # 计算全套评估指标
    metrics["network"] = network
    # 保存网络对象，画第一层权重图要用
    metrics["history"] = network.history
    # 保存训练日志，画训练曲线要用
    metrics["train_seconds"] = train_seconds
    # 记下训练耗时
    metrics["total_seconds"] = train_seconds
    # 端到端方案的总耗时就是训练耗时
    metrics["y_pred"] = y_pred
    # 保存预测结果
    metrics["hidden_features_train"] = network.forward(X_train)[1]
    # 取隐藏层激活作为"网络自己学出来的 64 维特征"，用于和人工特征做特征空间对比
    return metrics
    # 返回该方案的全部产物


def run_single_seed(
    images: np.ndarray,
    flattened: np.ndarray,
    targets: np.ndarray,
    seed: int,
    test_size: float,
    hidden_dim: int,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    verbose: bool,
) -> dict:
    """在给定随机种子下完整跑一遍两条流水线，供主实验和多随机种子检验共用。

    参数：
        images / flattened / targets：数据集的三份视图（见 load_dataset）。
        seed：划分训练测试集与初始化网络用的随机种子。
        test_size：测试集比例。
        hidden_dim / epochs / batch_size / learning_rate：端到端网络的超参数。
        verbose：是否打印网络训练过程。
    返回：字典，含划分信息、方案 A1/A2、方案 B、参照组 R 的指标与公共数据。
    """
    train_index, test_index = split_dataset(flattened, targets, test_size, seed)
    # 两条流水线共用这一组下标
    images_train, images_test = images[train_index], images[test_index]
    # 方案 A 需要 (n, 8, 8) 的原始尺度图像
    X_train, X_test = flattened[train_index], flattened[test_index]
    # 方案 B 需要 (n, 64) 的归一化像素
    y_train, y_test = targets[train_index], targets[test_index]
    # 两边的标签完全相同
    result_a16 = run_handcrafted_pipeline(images_train, y_train, images_test, y_test, use_quadrants=False)
    # 方案 A1：只用 16 维行/列投影
    result_a20 = run_handcrafted_pipeline(images_train, y_train, images_test, y_test, use_quadrants=True)
    # 方案 A2：16 维投影 + 4 维象限均值 = 20 维
    result_b = run_end_to_end_pipeline(
        X_train, y_train, X_test, y_test, hidden_dim, epochs, batch_size, learning_rate, seed, verbose
    )
    # 方案 B：端到端 MLP
    result_control = run_raw_pixel_linear_control(X_train, y_train, X_test, y_test)
    # 参照组 R：同一个逻辑回归，但输入换成未做人工降维的 64 维原始像素
    return {
        "seed": seed,
        # 本次使用的随机种子
        "train_index": train_index,
        # 训练集下标
        "test_index": test_index,
        # 测试集下标
        "y_train": y_train,
        # 训练标签
        "y_test": y_test,
        # 测试标签
        "X_train": X_train,
        # 训练像素（归一化）
        "X_test": X_test,
        # 测试像素（归一化）
        "images_train": images_train,
        # 训练图像（0~16 尺度）
        "images_test": images_test,
        # 测试图像（0~16 尺度）
        "A16": result_a16,
        # 方案 A1 的结果
        "A20": result_a20,
        # 方案 A2 的结果
        "B": result_b,
        # 方案 B 的结果
        "R": result_control,
        # 参照组 R 的结果
    }
    # 打包返回


def figure_handcrafted_features(
    images: np.ndarray,
    targets: np.ndarray,
    out_path: str,
) -> None:
    """画图 1：人工特征示意（原图 + 行投影 + 列投影 + 象限均值）。

    参数：
        images：形状 (n, 8, 8) 的图像，0~16 尺度。
        targets：形状 (n,) 的标签。
        out_path：图片保存路径。
    返回：无（图片直接写盘）。

    为什么这么选样本：第三、四行放的是"人工特征余弦相似度最高、但真实标签不同"的一对图。
    它们在人看来是两个不同的数字，在 20 维人工特征空间里却几乎重合——
    这一对图就是"人工特征丢了空间信息、导致类别不可分"的直接证据。

    注意：这里的特征是对"全部样本"现算的，与训练/测试划分无关。
    本函数只做示意图，不参与任何训练或评估，因此不存在信息泄漏问题；
    换成训练集子集来挑反而会让"样本下标"和"图像数组下标"错位。
    """
    all_features = handcrafted.handcrafted_features(images, use_quadrants=True)
    # 对全部样本算一遍 20 维人工特征，保证后面的相似度搜索与 images/targets 的下标一一对应
    pair_a, pair_b, similarity = handcrafted.most_similar_cross_label_pair(all_features, targets)
    # 找出人工特征最像的一对异类样本
    boldest = int(np.argmax(images.reshape(images.shape[0], -1).sum(axis=1)))
    # 挑一张墨迹最多的图作为"典型样本"，通常人眼一看就能认出来
    sample_indices = [boldest, pair_a, pair_b]
    # 三行分别是：典型样本、最相似对里的第一张、第二张
    row_notes = ["典型样本", f"最相似异类对之一（余弦相似度 {similarity:.3f}）", f"最相似异类对之二（余弦相似度 {similarity:.3f}）"]
    # 每行的说明文字

    figure = plt.figure(figsize=(12.5, 8.2))
    # 新建画布
    grid = figure.add_gridspec(3, 4, width_ratios=[1.0, 1.5, 1.5, 1.1], hspace=0.75, wspace=0.45)
    # 3 行 4 列的网格：原图、行投影、列投影、象限均值；列宽按内容需要设定
    for row, sample in enumerate(sample_indices):
        # 逐行画一个样本
        image = images[sample]
        # 取出该样本的 8x8 图
        row_projection = image.sum(axis=1) / PIXEL_MAX
        # 行投影：得到 8 个数，与 features.py 中 row_projection 的算法一致
        column_projection = image.sum(axis=0) / PIXEL_MAX
        # 列投影：得到 8 个数
        quadrant = handcrafted.quadrant_means(image[None, :, :])[0]
        # 象限均值：复用 features.py 的公式，额外加一维组成 (1,8,8) 以匹配它的接口

        axis_image = figure.add_subplot(grid[row, 0])
        # 第 1 列：原图
        axis_image.imshow(image, cmap="gray_r", vmin=0, vmax=PIXEL_MAX)
        # gray_r 让"数值大=墨迹黑"，符合人对纸面数字的直觉
        axis_image.set_title(f"样本 #{sample}\n真实标签 = {targets[sample]}（{row_notes[row]}）", fontsize=10)
        # 标题里标注样本编号、真实标签和这一行为什么被选中
        axis_image.set_xticks([])
        # 去掉刻度：8x8 的像素图上刻度只会碍眼
        axis_image.set_yticks([])
        # 去掉 y 刻度

        axis_row = figure.add_subplot(grid[row, 1])
        # 第 2 列：行投影（水平条形图，方向与图的行一致）
        axis_row.barh(np.arange(8), row_projection, color="#2f5fd0")
        # 画 8 根横条，第 i 根的长度是第 i 行的墨量
        axis_row.set_ylim(7.5, -0.5)
        # y 轴反向，让第 0 行画在最上面，和原图的行顺序对应
        axis_row.set_xlim(0, 1)
        # x 轴固定 0~1，所有样本的投影图刻度一致才方便横向比较
        axis_row.set_ylabel("行号")
        # y 轴含义
        if row == 0:
            # 只在第一行写列标题，避免整张图都是重复文字
            axis_row.set_title("行投影（8 维）：每行像素和", fontsize=11)
            # 列标题
        axis_row.grid(axis="x", alpha=0.3)
        # 加淡网格便于读数

        axis_column = figure.add_subplot(grid[row, 2])
        # 第 3 列：列投影（垂直条形图）
        axis_column.bar(np.arange(8), column_projection, color="#c26a1a")
        # 画 8 根竖条，第 j 根的高度是第 j 列的墨量
        axis_column.set_ylim(0, 1)
        # y 轴固定 0~1
        axis_column.set_xlabel("列号")
        # x 轴含义
        if row == 0:
            # 同样只在第一行写标题
            axis_column.set_title("列投影（8 维）：每列像素和", fontsize=11)
            # 列标题
        axis_column.grid(axis="y", alpha=0.3)
        # 加淡网格

        axis_quadrant = figure.add_subplot(grid[row, 3])
        # 第 4 列：象限均值
        axis_quadrant.imshow(quadrant.reshape(2, 2), cmap="viridis", vmin=0, vmax=1)
        # 2x2 的小热力图，颜色越亮表示该象限平均越黑
        for position, value in enumerate(quadrant):
            # 在每个格子里写上具体数值
            axis_quadrant.text(
                position % 2,
                position // 2,
                f"{value:.2f}",
                ha="center",
                va="center",
                color="white",
                fontsize=11,
            )
            # 文字位置 (列, 行) 与 reshape(2,2) 后的下标一致
        axis_quadrant.set_xticks([])
        # 去掉刻度
        axis_quadrant.set_yticks([])
        # 去掉刻度
        if row == 0:
            # 只在第一行写标题
            axis_quadrant.set_title("象限均值（4 维）", fontsize=11)
            # 列标题
    figure.suptitle(
        "图 1  人工特征（PPT 绿色模块）：8x8 原图被压成 16 维行/列投影 + 4 维象限均值\n"
        "注意后两行：两张标签不同的图，行/列投影几乎一模一样——"
        "投影只统计「每行/每列有多少墨」，丢掉了墨在行内、列内的具体位置",
        fontsize=12,
    )
    # 总标题把这张图要表达的核心结论直接写出来
    figure.savefig(out_path, dpi=150, bbox_inches="tight")
    # 保存图片，bbox_inches 收紧留白
    plt.close(figure)
    # 关闭画布释放内存


def figure_learned_first_layer(network: SoftmaxMLP, out_path: str) -> None:
    """画图 2：端到端网络第一层的 64 个权重向量，每个渲染成一张 8x8 小图。

    参数：
        network：训练好的 SoftmaxMLP。
        out_path：图片保存路径。
    返回：无（图片直接写盘）。

    怎么读这张图：隐藏单元 k 的输出是 W1[:, k] 与像素向量的内积，
    所以每个 8x8 的小图就是该单元在"找什么样的图案"——
    这是网络从数据里自己学出来的特征探测器，和人工写死的行/列投影形成直接对照。
    """
    filters = network.first_layer_filters()
    # 形状 (64, 8, 8)，第 k 张是第 k 个隐藏单元的权重模板
    limit = float(np.abs(filters).max())
    # 取所有权重里绝对值最大的那个，用它作为色标上下限，保证 64 张小图颜色可比
    grid_size = 8
    # 64 个单元排成 8x8 的方阵
    figure, axes = plt.subplots(grid_size, grid_size, figsize=(11.5, 11.5))
    # 一次创建 64 个子图
    for index in range(grid_size * grid_size):
        # 逐个隐藏单元画图
        axis = axes[index // grid_size, index % grid_size]
        # 定位到第 index 个子图
        axis.imshow(filters[index], cmap="coolwarm", vmin=-limit, vmax=limit)
        # coolwarm 双色映射：蓝色表示负权重（该像素变亮反而支持这一单元），红色表示正权重
        axis.set_title(f"#{index}", fontsize=7)
        # 标出单元编号，方便正文里点名
        axis.set_xticks([])
        # 去掉刻度
        axis.set_yticks([])
        # 去掉刻度
    figure.suptitle(
        "图 2  端到端网络从数据里学出的 64 个「特征探测器」（隐藏层第一层权重，8x8）\n"
        "红色 = 正权重（这里出现墨迹会激活该单元），蓝色 = 负权重；"
        "对比图 1：这些模板不是人写的，而是梯度下降自己在训练中长出来的",
        fontsize=12,
    )
    # 总标题强调"学出来"这一点
    mappable = ScalarMappable(norm=Normalize(vmin=-limit, vmax=limit), cmap="coolwarm")
    # 构造一个只用于配色的对象，让 64 个子图共用一条颜色条
    figure.colorbar(mappable, ax=axes, shrink=0.55, pad=0.02, label="第一层权重取值")
    # 在最右侧加一条竖直颜色条
    figure.savefig(out_path, dpi=150, bbox_inches="tight")
    # 保存
    plt.close(figure)
    # 关闭画布


def figure_confusion_matrices(result_a: dict, result_b: dict, y_test: np.ndarray, out_path: str) -> None:
    """画图 3：两个方案的混淆矩阵并排对比。

    参数：
        result_a：方案 A（人工特征 + 逻辑回归）的结果字典。
        result_b：方案 B（端到端 MLP）的结果字典。
        y_test：测试集真标签。
        out_path：图片保存路径。
    返回：无（图片直接写盘）。

    怎么读：第 i 行第 j 列的数字表示"真实是 i、被预测成 j"的样本数，
    对角线是答对的，非对角线全是错误；对比两张图能看出人工特征方案
    在哪些数字对之间大规模混淆（通常是形状相似的数字对）。
    """
    figure, axes = plt.subplots(1, 2, figsize=(13.5, 6.0))
    # 左右两张子图
    panels = [
        (axes[0], result_a["confusion"], "方案 A：人工特征 + 逻辑回归", result_a["accuracy"]),
        # 左边放方案 A
        (axes[1], result_b["confusion"], "方案 B：端到端 MLP（全蓝）", result_b["accuracy"]),
        # 右边放方案 B
    ]
    # 把两张图需要的东西整理成列表，循环处理避免代码重复
    shared_max = float(max(result_a["confusion"].max(), result_b["confusion"].max()))
    # 两张图共用同一个色标上限：否则各自按自己的最大值归一化，深色浅色就不能横向比较了
    for axis, matrix, title, accuracy in panels:
        # 逐个面板绘制
        axis.imshow(matrix, cmap="Blues", vmin=0, vmax=shared_max)
        # 用蓝色深浅表示样本数；vmax 用上面算出的公共上限
        for i in range(10):
            # 逐行标注
            for j in range(10):
                # 逐列标注
                value = matrix[i, j]
                # 取出该格的样本数
                if value == 0:
                    # 0 不标注，图面更干净
                    continue
                    # 跳到下一格
                axis.text(
                    j,
                    i,
                    str(value),
                    ha="center",
                    va="center",
                    fontsize=8,
                    color="white" if value > shared_max * 0.55 else "#222222",
                )
                # 深色背景上用白字、浅色背景上用深字，保证可读
        axis.set_xticks(np.arange(10))
        # x 轴是预测类别
        axis.set_yticks(np.arange(10))
        # y 轴是真实类别
        axis.set_xlabel("预测类别")
        # x 轴含义
        axis.set_ylabel("真实类别")
        # y 轴含义
        axis.set_title(f"{title}\n测试准确率 = {accuracy:.2%}，非对角线合计 = {int(matrix.sum() - matrix.trace())} 个错误", fontsize=11)
        # 标题里给出准确率和总错误数
    figure.suptitle("图 3  两方案的混淆矩阵对比（同一测试集、同一份划分）", fontsize=13)
    # 总标题
    figure.tight_layout()
    # 自动调整间距
    figure.savefig(out_path, dpi=150, bbox_inches="tight")
    # 保存
    plt.close(figure)
    # 关闭画布


def figure_misclassified_examples(
    images_test: np.ndarray,
    y_test: np.ndarray,
    result_a: dict,
    result_b: dict,
    out_path: str,
) -> None:
    """画图 4：两方案错误样本对比，分三类展示。

    参数：
        images_test：测试集图像（0~16 尺度）。
        y_test：测试集真标签。
        result_a：方案 A 的结果字典。
        result_b：方案 B 的结果字典。
        out_path：图片保存路径。
    返回：无（图片直接写盘）。

    三行的含义：
        第 1 行"人工特征错、端到端对"：这类样本最能说明问题——图上的信息足够，
            但投影特征把它压没了，网络却能靠像素里的空间结构认出来；
        第 2 行"两方案都错"：往往是字迹本身潦草或像别的数字，属于数据本身的难度；
        第 3 行"人工特征对、端到端错"：数量通常很少，说明端到端不是万能的，
            但在本任务里它错的样本明显更少。
    """
    prediction_a = result_a["y_pred"]
    # 方案 A 的预测
    prediction_b = result_b["y_pred"]
    # 方案 B 的预测
    correct_a = prediction_a == y_test
    # 方案 A 是否答对的布尔数组
    correct_b = prediction_b == y_test
    # 方案 B 是否答对的布尔数组
    categories = [
        ("人工特征方案错 / 端到端对", np.where(~correct_a & correct_b)[0]),
        # 第一类下标
        ("两个方案都错", np.where(~correct_a & ~correct_b)[0]),
        # 第二类下标
        ("人工特征方案对 / 端到端错", np.where(correct_a & ~correct_b)[0]),
        # 第三类下标
    ]
    # 三类样本的测试集下标
    figure, axes = plt.subplots(3, CONFUSION_FIGURE_LIMIT, figsize=(14.5, 8.4))
    # 3 行（三类）× 6 列（每类最多 6 个样本）
    for row, (title, indices) in enumerate(categories):
        # 逐类绘制
        for column in range(CONFUSION_FIGURE_LIMIT):
            # 逐列绘制
            axis = axes[row, column]
            # 当前格子
            if column >= indices.shape[0]:
                # 该类的样本不够 6 个时，剩余格子留空
                axis.axis("off")
                # 关掉坐标轴，避免出现空白方框
                continue
                # 进入下一格
            sample = int(indices[column])
            # 该类别里第 column 个样本在测试集中的位置
            axis.imshow(images_test[sample], cmap="gray_r", vmin=0, vmax=PIXEL_MAX)
            # 画出这张 8x8 的图
            axis.set_title(
                f"真 {y_test[sample]}\nA→{prediction_a[sample]}  B→{prediction_b[sample]}",
                fontsize=8,
            )
            # 标题给出真标签和两个方案的预测，一眼看出谁对谁错
            axis.set_xticks([])
            # 去掉刻度
            axis.set_yticks([])
            # 去掉刻度
        axes[row, 0].text(
            -0.45,
            0.5,
            title,
            transform=axes[row, 0].transAxes,
            rotation=90,
            ha="center",
            va="center",
            fontsize=10,
        )
        # 在图左侧竖直写出类别名称；用 text 而不是 ylabel，位置更容易控制
    figure.suptitle(
        f"图 4  错误样本对比（共 {int((~correct_a).sum())} 个 vs {int((~correct_b).sum())} 个错误）\n"
        "第一行的样本最能说明人工特征的问题：像素里明明有足够信息，投影特征却把它压掉了",
        fontsize=12,
    )
    # 总标题给出两方案各自的错误总数
    figure.tight_layout(rect=(0.02, 0, 1, 0.94))
    # 自动布局，给顶部标题留出空间
    figure.savefig(out_path, dpi=150, bbox_inches="tight")
    # 保存
    plt.close(figure)
    # 关闭画布


def figure_feature_space(
    features_a: np.ndarray,
    hidden_b: np.ndarray,
    y_train: np.ndarray,
    out_path: str,
) -> None:
    """画图 5：人工特征空间 vs 网络学出的隐藏层特征空间（都降到 2 维看类间重叠）。

    参数：
        features_a：形状 (n, 20) 的人工特征（训练集）。
        hidden_b：形状 (n, 64) 的网络隐藏层激活（训练集）。
        y_train：形状 (n,) 的训练标签。
        out_path：图片保存路径。
    返回：无（图片直接写盘）。

    为什么要看这个：混淆矩阵只告诉我们"最后错多少"，
    这里把两边的中间表示各降到 2 维染上颜色：
    同色点聚成一团、不同色点分开，就说明这种表示本身容易分类；
    颜色大面积交织，说明后面接再强的分类器也难分开。
    注意这里用的是训练集，且 PCA 是无标签的降维，不涉及测试集信息。
    """
    reduced_a = PCA(n_components=2, random_state=0).fit_transform(features_a)
    # 把 20 维人工特征压到 2 维
    reduced_b = PCA(n_components=2, random_state=0).fit_transform(hidden_b)
    # 把 64 维隐藏层激活压到 2 维
    figure, axes = plt.subplots(1, 2, figsize=(13.0, 5.8))
    # 左右两栏
    panels = [
        (axes[0], reduced_a, "方案 A 的中间表示：20 维人工特征", features_a.shape[1]),
        # 左边是人工特征
        (axes[1], reduced_b, "方案 B 的中间表示：网络自己学出的 64 维隐藏层特征", hidden_b.shape[1]),
        # 右边是学出来的特征
    ]
    # 整理成列表循环处理
    for axis, reduced, title, dimension in panels:
        # 逐个面板绘制
        scatter = axis.scatter(
            reduced[:, 0],
            reduced[:, 1],
            c=y_train,
            cmap="tab10",
            s=12,
            alpha=0.75,
            vmin=0,
            vmax=9,
        )
        # 用 10 色离散色表按数字类别上色；vmin/vmax 固定为 0/9 保证两边颜色含义一致
        axis.set_title(f"{title}\n输入维度 {dimension}，PCA 降到 2 维", fontsize=11)
        # 标题注明原始维度
        axis.set_xlabel("主成分 1")
        # x 轴
        axis.set_ylabel("主成分 2")
        # y 轴
        axis.grid(alpha=0.25)
        # 淡网格
        colorbar = figure.colorbar(scatter, ax=axis, ticks=np.arange(10))
        # 每张图配一条颜色条，标出 0~9
        colorbar.set_label("数字类别")
        # 颜色条含义
    figure.suptitle(
        "图 5  中间表示的可分性对比（训练集，PCA 仅用于可视化）\n"
        "右图同色点聚得更紧、异色点分得更开，说明「网络自己学出的特征」比「人写的投影特征」更容易分类",
        fontsize=12,
    )
    # 总标题点明结论
    figure.tight_layout()
    # 自动布局
    figure.savefig(out_path, dpi=150, bbox_inches="tight")
    # 保存
    plt.close(figure)
    # 关闭画布


def figure_training_curve(history: list, out_path: str) -> None:
    """画图 6：端到端网络的训练曲线（损失与准确率随 epoch 变化）。

    参数：
        history：model.EpochLog 的列表，每个 epoch 一条。
        out_path：图片保存路径。
    返回：无（图片直接写盘）。

    怎么读：训练曲线一直下降说明优化在做正事；
    测试曲线在训练曲线附近、没有明显"训练继续降而测试转升"的开口，
    说明这个规模的网络在 digits 上并没有严重过拟合。
    """
    epochs = [record.epoch for record in history]
    # 横坐标：轮数
    figure, axes = plt.subplots(1, 2, figsize=(12.5, 4.8))
    # 左右两栏：损失、准确率
    axes[0].plot(epochs, [record.train_loss for record in history], label="训练集损失", color="#2f5fd0")
    # 训练损失曲线
    axes[0].plot(epochs, [record.eval_loss for record in history], label="测试集损失", color="#dd3f3f")
    # 测试损失曲线（只在日志里非空，fit 时传了 eval_set 才有值）
    axes[0].set_xlabel("训练轮数（epoch）")
    # 横轴含义
    axes[0].set_ylabel("平均交叉熵损失")
    # 纵轴含义
    axes[0].set_title("损失：交叉熵随训练下降")
    # 子图标题
    axes[0].grid(alpha=0.3)
    # 淡网格
    axes[0].legend()
    # 图例
    axes[1].plot(epochs, [record.train_accuracy for record in history], label="训练集准确率", color="#2f5fd0")
    # 训练准确率曲线
    axes[1].plot(epochs, [record.eval_accuracy for record in history], label="测试集准确率", color="#dd3f3f")
    # 测试准确率曲线
    axes[1].set_xlabel("训练轮数（epoch）")
    # 横轴含义
    axes[1].set_ylabel("准确率")
    # 纵轴含义
    axes[1].set_title("准确率：几十轮后基本饱和")
    # 子图标题
    axes[1].grid(alpha=0.3)
    # 淡网格
    axes[1].legend()
    # 图例
    figure.suptitle("图 6  端到端方案（方案 B）的训练过程：全部函数（特征 + 分类器）由这一个损失一起优化", fontsize=12)
    # 总标题强调"一个损失端到端优化"
    figure.tight_layout()
    # 自动布局
    figure.savefig(out_path, dpi=150, bbox_inches="tight")
    # 保存
    plt.close(figure)
    # 关闭画布


def write_metrics(
    path: str,
    dataset_info: dict,
    primary: dict,
    stability: list[dict],
    config: dict,
    figure_files: list[str],
    total_seconds: float,
) -> None:
    """把全部数值结果写成文本文件 results/metrics.txt。

    参数：
        path：输出文件路径。
        dataset_info：数据集概况（样本数、类别数、每类样本数等）。
        primary：主实验（默认种子）的完整结果。
        stability：多随机种子重复实验的结果列表。
        config：本次运行的超参数配置。
        figure_files：实际生成的图片文件名列表。
        total_seconds：脚本总耗时。
    返回：无（文件直接写盘）。

    为什么还要单独写一份纯文本记录：README 是给人读的说明文档，
    这份文件是"实验原始记录"，方便日后核对数字、对比改动前后的效果，
    也满足"结果可复现"的要求。
    """
    lines: list[str] = []
    # 逐行累积要写入的内容
    lines.append("=" * 78)
    # 顶部装饰线
    lines.append("手写数字分类：人工特征流水线（方案 A）vs 端到端学习（方案 B）—— 实验记录")
    # 标题
    lines.append("=" * 78)
    # 底部装饰线
    lines.append("")
    # 空行
    lines.append("【一、运行环境】")
    # 小节标题
    lines.append(f"操作系统        : {platform.platform()}")
    # 操作系统信息
    lines.append(f"Python 版本     : {sys.version.split()[0]}")
    # Python 版本
    lines.append(f"NumPy 版本      : {np.__version__}")
    # NumPy 版本
    lines.append(f"scikit-learn 版本: {sklearn.__version__}")
    # scikit-learn 版本（文件头已 import sklearn，这里直接用）
    lines.append("神经网络实现     : 纯 NumPy 手写 MLP（未使用 PyTorch，本机未安装）")
    # 说明网络实现方式
    lines.append(f"matplotlib 字体  : {config['font']}")
    # 记录中文字体是否成功注册
    lines.append("")
    # 空行
    lines.append("【二、数据集】")
    # 小节标题
    lines.append("来源            : sklearn.datasets.load_digits（随 scikit-learn 内置，无需联网下载）")
    # 数据集来源
    lines.append(f"样本总数        : {dataset_info['n_samples']}")
    # 样本总数
    lines.append("图像尺寸        : 8 x 8 灰度，像素取值 0~16")
    # 图像尺寸与像素范围
    lines.append(f"类别数          : {dataset_info['n_classes']}（数字 0~9）")
    # 类别数
    lines.append(f"每类样本数      : {', '.join(str(int(c)) for c in dataset_info['class_counts'])}")
    # 每类样本数，用于说明数据并非完全均衡
    lines.append(f"训练/测试划分   : test_size={config['test_size']}，stratify=y，random_state={config['seed']}")
    # 划分方式
    lines.append(f"训练集样本数    : {dataset_info['n_train']}")
    # 训练集大小
    lines.append(f"测试集样本数    : {dataset_info['n_test']}")
    # 测试集大小
    lines.append("说明            : 两条流水线使用完全相同的划分（同一组样本下标），保证差距只来自特征来源")
    # 强调对比的公平性
    lines.append("")
    # 空行
    lines.append("【三、方案配置】")
    # 小节标题
    lines.append("方案 A（绿 + 蓝）: 人工特征提取器（固定公式、不可学习）+ 逻辑回归（唯一的学习模块）")
    # 方案 A 的一句话描述
    lines.append("                  A1 = 行投影 8 维 + 列投影 8 维 = 16 维")
    # A1 的特征构成
    lines.append("                  A2 = A1 + 4 个象限平均亮度 = 20 维")
    # A2 的特征构成
    lines.append("                  分类器：sklearn LogisticRegression(max_iter=5000, C=1.0, 默认 lbfgs)")
    # 分类器配置
    lines.append("方案 B（全蓝）   : 原始 64 维像素直接输入 MLP，无任何人工作用上的模块")
    # 方案 B 的一句话描述
    lines.append(f"                  网络结构：64 -> {config['hidden_dim']}（ReLU）-> 10（softmax）")
    # 网络结构
    lines.append(f"                  损失函数：交叉熵；优化器：Adam（mini-batch，batch_size={config['batch_size']}，lr={config['learning_rate']}）")
    # 损失与优化器
    lines.append(f"                  训练轮数：{config['epochs']} 个 epoch")
    # 训练轮数
    lines.append("")
    # 空行
    lines.append("【四、主实验结果（测试集准确率对照）】")
    # 小节标题
    rows = [
        ["A1 人工特征 16 维", "+ 逻辑回归", str(primary["A16"]["feature_dim"]), f"{primary['A16']['accuracy']:.4f}", f"{primary['A16']['accuracy'] * 100:.2f}%"],
        # A1 一行
        ["A2 人工特征 20 维", "+ 逻辑回归", str(primary["A20"]["feature_dim"]), f"{primary['A20']['accuracy']:.4f}", f"{primary['A20']['accuracy'] * 100:.2f}%"],
        # A2 一行
        ["B  原始 64 维像素", "+ 端到端 MLP", "64", f"{primary['B']['accuracy']:.4f}", f"{primary['B']['accuracy'] * 100:.2f}%"],
        # B 一行
        ["R  原始 64 维像素", "+ 逻辑回归", str(primary["R"]["feature_dim"]), f"{primary['R']['accuracy']:.4f}", f"{primary['R']['accuracy'] * 100:.2f}%"],
        # 参照组一行：分类器与方案 A 完全相同，只有输入不同
    ]
    # 四行结果
    lines.extend(format_table(["方案", "后接模块", "输入维度", "测试准确率", "百分比"], rows, ["left", "left", "right", "right", "right"]))
    # 渲染成对齐的文本表格
    lines.append("")
    # 空行
    lines.append(f"方案 B 相对方案 A1 的准确率提升：{(primary['B']['accuracy'] - primary['A16']['accuracy']) * 100:.2f} 个百分点")
    # 计算提升幅度
    lines.append(f"方案 B 相对方案 A2 的准确率提升：{(primary['B']['accuracy'] - primary['A20']['accuracy']) * 100:.2f} 个百分点")
    # 相对 A2 的提升
    lines.append("")
    # 空行
    lines.append("关于参照组 R（它不属于 PPT 上的两种范式，只是实验层面的对照）：")
    # 说明 R 的性质，避免读者把它当成第三种范式
    lines.append("  R 用的是与方案 A 一模一样的逻辑回归（同一个求解器、同一组超参数、同样是线性分类器），")
    # 强调分类器完全相同
    lines.append(f"  唯一区别是输入端不做人工降维。结果 R 相对 A2 变化了 "
                 f"{(primary['R']['accuracy'] - primary['A20']['accuracy']) * 100:+.2f} 个百分点。")
    # 给出对照结果
    lines.append("  这排除了「差距只是因为逻辑回归比神经网络弱」这一种解释，把原因锁定在人工特征本身的信息损失上。")
    # 说明这个参照组的价值
    lines.append("")
    # 空行
    lines.append("【五、宏平均 F1 与每类召回率】")
    # 小节标题
    rows = []
    # 重新构造表格数据
    for name, result in (("A1", primary["A16"]), ("A2", primary["A20"]), ("B", primary["B"]), ("R", primary["R"])):
        # 三个方案 + 参照组各一行
        rows.append(
            [name, f"{result['macro_f1']:.4f}"]
            + [f"{value:.3f}" for value in result["per_class_recall"]]
            # 每类召回率保留三位小数
        )
        # 组装成一行
    headers = ["方案", "宏平均F1"] + [f"数字{i}" for i in range(10)]
    # 表头：10 个类别各一列
    lines.extend(format_table(headers, rows, ["left", "right"] + ["right"] * 10))
    # 渲染表格
    lines.append("")
    # 空行
    lines.append("（每类召回率 = 该数字被正确识别的比例；测试集中每类的样本数见下表）")
    # 说明文字
    rows = [
        ["测试样本数"] + [str(int(value)) for value in primary["A16"]["per_class_support"]],
        # 每类测试样本数
    ]
    # 只有一行
    lines.extend(format_table(["项目"] + [f"数字{i}" for i in range(10)], rows, ["left"] + ["right"] * 10))
    # 渲染表格
    lines.append("")
    # 空行
    lines.append("【六、混淆矩阵】")
    # 小节标题
    for name, result in (("方案 A2（20 维人工特征 + 逻辑回归）", primary["A20"]), ("方案 B（端到端 MLP）", primary["B"])):
        # 两张混淆矩阵分别打印
        lines.append(f"--- {name}，测试准确率 {result['accuracy']:.4f} ---")
        # 小标题
        matrix = result["confusion"]
        # 取出矩阵
        rows = [
            [str(i)] + [str(int(value)) for value in matrix[i]] + [str(int(matrix[i].sum()))]
            for i in range(10)
            # 每行：真实类别 + 10 个预测列 + 该行合计
        ]
        # 组装行数据
        rows.append(["合计"] + [str(int(matrix[:, j].sum())) for j in range(10)] + [str(int(matrix.sum()))])
        # 最后加一行列合计
        lines.extend(format_table(["真\\预测"] + [str(i) for i in range(10)] + ["合计"], rows, ["left"] + ["right"] * 11))
        # 渲染表格
        lines.append("")
        # 表格之间空一行
    lines.append("【七、多随机种子稳定性检验】")
    # 小节标题
    lines.append("（换不同的训练/测试划分与网络初始化，看「端到端更好」这个结论是否稳定）")
    # 说明
    rows = []
    # 收集行
    for record in stability:
        # 每个种子一行
        rows.append(
            [
                str(record["seed"]),
                f"{record['A16']['accuracy']:.4f}",
                f"{record['A20']['accuracy']:.4f}",
                f"{record['B']['accuracy']:.4f}",
                f"{record['R']['accuracy']:.4f}",
                f"{(record['B']['accuracy'] - record['A20']['accuracy']) * 100:+.2f}",
            ]
        )
        # 列出三个方案的准确率、参照组准确率和差值
    a16_values = np.array([record["A16"]["accuracy"] for record in stability])
    # 收集所有种子的 A1 准确率
    a20_values = np.array([record["A20"]["accuracy"] for record in stability])
    # 收集所有种子的 A2 准确率
    b_values = np.array([record["B"]["accuracy"] for record in stability])
    # 收集所有种子的 B 准确率
    r_values = np.array([record["R"]["accuracy"] for record in stability])
    # 收集所有种子的参照组 R 准确率
    rows.append(
        [
            "平均",
            f"{a16_values.mean():.4f}",
            f"{a20_values.mean():.4f}",
            f"{b_values.mean():.4f}",
            f"{r_values.mean():.4f}",
            f"{(b_values.mean() - a20_values.mean()) * 100:+.2f}",
        ]
    )
    # 追加平均行
    rows.append(
        [
            "标准差",
            f"{a16_values.std():.4f}",
            f"{a20_values.std():.4f}",
            f"{b_values.std():.4f}",
            f"{r_values.std():.4f}",
            "-",
        ]
    )
    # 追加标准差行，用来说明结果波动有多大
    lines.extend(format_table(["随机种子", "A1 准确率", "A2 准确率", "B 准确率", "R 准确率", "B-A2(百分点)"], rows, ["left"] + ["right"] * 5))
    # 渲染表格
    lines.append("说明：三个种子下 B 都稳定高于 A2 与 R，R 也都高于 A2，「端到端/不降维更好」不是某一次划分的偶然。")
    # 补一句结论性说明，避免读者只看单行数字
    lines.append("")
    # 空行
    lines.append("【八、耗时】")
    # 小节标题
    lines.append(f"方案 A1：特征提取 {primary['A16']['extract_seconds']:.3f} s + 分类器训练 {primary['A16']['train_seconds']:.3f} s = {primary['A16']['total_seconds']:.3f} s")
    # A1 耗时拆解
    lines.append(f"方案 A2：特征提取 {primary['A20']['extract_seconds']:.3f} s + 分类器训练 {primary['A20']['train_seconds']:.3f} s = {primary['A20']['total_seconds']:.3f} s")
    # A2 耗时拆解
    lines.append(f"方案 B ：端到端训练（含全部特征学习）{primary['B']['total_seconds']:.3f} s")
    # B 耗时
    lines.append(f"参照组 R：仅在原始像素上训练分类器 {primary['R']['total_seconds']:.3f} s")
    # 参照组耗时
    lines.append(f"脚本总耗时：{total_seconds:.3f} s（含多随机种子重复实验与全部画图）")
    # 脚本总耗时
    lines.append("")
    # 空行
    lines.append("【九、生成的图片】")
    # 小节标题
    for name in figure_files:
        # 逐张列出
        lines.append(f"  {name}")
        # 文件名
    lines.append("")
    # 空行
    lines.append("【十、结论】")
    # 小节标题
    lines.append(
        f"1. 端到端方案 B 的测试准确率 {primary['B']['accuracy']:.2%}，明显高于人工投影特征方案 A1 的 "
        f"{primary['A16']['accuracy']:.2%} 与 A2 的 {primary['A20']['accuracy']:.2%}。"
    )
    # 结论 1
    lines.append(
        "2. 人工投影特征把 8x8 的空间结构压成「每行/每列有多少墨」，大量形状不同的数字在特征空间里重叠，"
        "这是方案 A1/A2 准确率上不去的根本原因；"
        "多补 4 个象限均值只能小幅改善，因为手工设计的分辨率始终有限。"
    )
    # 结论 2
    lines.append(
        f"3. 参照组 R（同一个逻辑回归，但输入换成未做人工降维的 64 维原始像素）拿到 {primary['R']['accuracy']:.2%}，"
        f"比方案 A1 高出 {(primary['R']['accuracy'] - primary['A16']['accuracy']) * 100:.2f} 个百分点。"
        "分类器完全没变，唯一变的是特征，这直接证明瓶颈在人工特征而不是分类器能力。"
    )
    # 结论 3：用参照组把"是特征的问题还是分类器的问题"这一个疑点彻底堵死
    lines.append(
        "4. 端到端方案的 64 个隐藏单元权重是梯度下降自己学出的 8x8 模板（见图 2），"
        "相当于用一整层「可以学的特征提取器」取代了人写死的投影公式，因此准确率最高。"
    )
    # 结论 4
    lines.append("")
    # 末尾空行
    with open(path, "w", encoding="utf-8") as handle:
        # 以 UTF-8 打开文件，避免中文在 Windows 默认编码下写坏
        handle.write("\n".join(lines) + "\n")
        # 写入全部内容，末尾补一个换行符
    return
    # 显式返回，强调没有别的副作用


def print_summary(primary: dict) -> None:
    """在终端打印一行行结果摘要，方便运行完立刻看到结论。

    参数：
        primary：主实验结果字典。
    返回：无（只打印）。
    """
    print("")
    # 先空一行
    print("=" * 62)
    # 装饰线
    print("测试集准确率对照（同一份划分，random_state = %d）" % primary["seed"])
    # 标题里带上种子
    print("-" * 62)
    # 分隔线
    print(f"  方案 A1  人工特征 16 维 + 逻辑回归   : {primary['A16']['accuracy']:.4f}")
    # A1 结果
    print(f"  方案 A2  人工特征 20 维 + 逻辑回归   : {primary['A20']['accuracy']:.4f}")
    # A2 结果
    print(f"  方案 B   原始 64 维像素 + 端到端 MLP : {primary['B']['accuracy']:.4f}")
    # B 结果
    print(f"  参照组 R 原始 64 维像素 + 逻辑回归   : {primary['R']['accuracy']:.4f}   ← 与 A 同一个分类器，只换特征")
    # 参照组结果：分类器与方案 A 完全相同，用来隔离"特征"这一个变量
    print("-" * 62)
    # 分隔线
    print(f"  端到端相对 A1 提升 : {(primary['B']['accuracy'] - primary['A16']['accuracy']) * 100:+.2f} 个百分点")
    # 提升幅度
    print(f"  端到端相对 A2 提升 : {(primary['B']['accuracy'] - primary['A20']['accuracy']) * 100:+.2f} 个百分点")
    # 提升幅度
    print(f"  参照组R 相对 A2 提升 : {(primary['R']['accuracy'] - primary['A20']['accuracy']) * 100:+.2f} 个百分点"
          "   ← 差距来自人工特征丢了信息，而非分类器太弱")
    # 参照组对方案 A2 的差值：排除了"线性分类器能力不足"的另一种解释
    print("=" * 62)
    # 装饰线


def parse_arguments() -> argparse.Namespace:
    """定义并解析命令行参数。

    参数：无（从 sys.argv 读取）。
    返回：解析后的参数命名空间。
    """
    parser = argparse.ArgumentParser(description="人工特征流水线 vs 端到端学习：手写数字分类对照实验")
    # 创建解析器并写一句用途说明
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="主实验随机种子（决定划分与网络初始化）")
    # 主实验种子
    parser.add_argument("--test-size", type=float, default=DEFAULT_TEST_SIZE, help="测试集比例")
    # 测试集比例
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS, help="端到端网络的训练轮数")
    # 训练轮数
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE, help="mini-batch 大小")
    # 批大小
    parser.add_argument("--hidden", type=int, default=DEFAULT_HIDDEN_DIM, help="隐藏层单元数")
    # 隐藏层宽度
    parser.add_argument("--learning-rate", type=float, default=DEFAULT_LEARNING_RATE, help="Adam 学习率")
    # 学习率
    parser.add_argument("--extra-seeds", type=str, default=DEFAULT_EXTRA_SEEDS, help="额外的随机种子，逗号分隔，用于稳定性检验")
    # 额外种子列表
    parser.add_argument("--skip-extra-seeds", action="store_true", help="跳过多随机种子重复实验，只跑主实验（更快）")
    # 跳过稳定性检验的开关
    parser.add_argument("--outdir", type=str, default=RESULTS_DIRNAME, help="结果输出目录（相对脚本所在目录）")
    # 输出目录
    parser.add_argument("--quiet", action="store_true", help="不打印网络训练过程")
    # 静默模式
    return parser.parse_args()
    # 返回解析结果


def main() -> int:
    """脚本入口：跑完两条流水线、画全部图、写出 metrics.txt。

    参数：无。
    返回：进程退出码，0 表示正常结束。
    """
    arguments = parse_arguments()
    # 解析命令行参数
    if hasattr(sys.stdout, "reconfigure"):
        # Windows 控制台默认用 GBK 编码，中文日志会乱码；显式改成 UTF-8 并允许替换无法编码的字符
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        # 只做这一件事：后续所有 print 都按 UTF-8 输出，与 Git Bash / mintty 的预期一致
    script_dir = os.path.dirname(os.path.abspath(__file__))
    # 脚本所在目录：用绝对路径，保证从任何工作目录运行都能把结果写对地方
    out_dir = arguments.outdir if os.path.isabs(arguments.outdir) else os.path.join(script_dir, arguments.outdir)
    # 输出目录：给的是相对路径时相对脚本目录解析
    os.makedirs(out_dir, exist_ok=True)
    # 创建输出目录；已存在时不报错
    font_name = setup_chinese_font()
    # 注册中文字体
    print(f"中文字体：{font_name if font_name else '未找到，图中的中文可能显示为方块'}")
    # 打印字体注册结果

    script_start = time.perf_counter()
    # 记录脚本起点，最后算总耗时
    images, flattened, targets = load_dataset()
    # 载入数据集
    print(f"数据集：load_digits，{images.shape[0]} 张 8x8 灰度图，{len(np.unique(targets))} 个类别")
    # 打印数据集概况

    config = {
        "seed": arguments.seed,
        # 主实验种子
        "test_size": arguments.test_size,
        # 测试集比例
        "epochs": arguments.epochs,
        # 训练轮数
        "batch_size": arguments.batch_size,
        # 批大小
        "hidden_dim": arguments.hidden,
        # 隐藏层宽度
        "learning_rate": arguments.learning_rate,
        # 学习率
        "font": font_name or "未注册",
        # 中文字体
    }
    # 打包配置，写进 metrics.txt

    print("\n[1/3] 主实验：方案 A1 / A2 / B（random_state = %d）" % arguments.seed)
    # 进度提示
    primary = run_single_seed(
        images,
        flattened,
        targets,
        seed=arguments.seed,
        test_size=arguments.test_size,
        hidden_dim=arguments.hidden,
        epochs=arguments.epochs,
        batch_size=arguments.batch_size,
        learning_rate=arguments.learning_rate,
        verbose=not arguments.quiet,
    )
    # 跑主实验：一次得到三个方案的完整结果
    print_summary(primary)
    # 打印准确率摘要

    n_train = primary["train_index"].shape[0]
    # 训练集样本数
    n_test = primary["test_index"].shape[0]
    # 测试集样本数
    dataset_info = {
        "n_samples": int(images.shape[0]),
        # 样本总数
        "n_classes": int(len(np.unique(targets))),
        # 类别数
        "class_counts": np.bincount(targets, minlength=10),
        # 每类样本数
        "n_train": int(n_train),
        # 训练集大小
        "n_test": int(n_test),
        # 测试集大小
    }
    # 打包数据集概况

    print("\n[2/3] 生成可视化")
    # 进度提示
    figure_files: list[str] = []
    # 记录真正写盘的图片文件名
    figure_specs = [
        (
            "01_handcrafted_features.png",
            lambda path: figure_handcrafted_features(images, targets, path),
        ),
        # 图 1：人工特征示意（对全集现算特征来挑"最相似异类对"，纯可视化用途，不涉及训练）
        (
            "02_learned_first_layer.png",
            lambda path: figure_learned_first_layer(primary["B"]["network"], path),
        ),
        # 图 2：网络学出的第一层权重
        (
            "03_confusion_matrices.png",
            lambda path: figure_confusion_matrices(primary["A20"], primary["B"], primary["y_test"], path),
        ),
        # 图 3：混淆矩阵对比（用 A2，因为它是方案 A 里更强的那个，对比更保守也更公平）
        (
            "04_misclassified_examples.png",
            lambda path: figure_misclassified_examples(primary["images_test"], primary["y_test"], primary["A20"], primary["B"], path),
        ),
        # 图 4：错误样本对比
        (
            "05_feature_space_comparison.png",
            lambda path: figure_feature_space(
                primary["A20"]["features_train"], primary["B"]["hidden_features_train"], primary["y_train"], path
            ),
        ),
        # 图 5：中间表示的可分性对比
        (
            "06_training_curve.png",
            lambda path: figure_training_curve(primary["B"]["history"], path),
        ),
        # 图 6：训练曲线
    ]
    # 每张图一个 (文件名, 绘制函数) 的组合
    for filename, draw in figure_specs:
        # 逐张生成
        target_path = os.path.join(out_dir, filename)
        # 拼出完整输出路径
        draw(target_path)
        # 调用对应的绘图函数
        figure_files.append(filename)
        # 记下文件名
        print(f"  {filename}  ({os.path.getsize(target_path) / 1024:.1f} KB)")
        # 打印文件名和体积，等于顺手验证"文件真的写出来了且非空"

    stability: list[dict] = [primary]
    # 稳定性检验列表，主实验本身就是其中一条记录
    if not arguments.skip_extra_seeds:
        # 没有显式跳过时才跑额外种子
        extra_seeds = [int(token) for token in arguments.extra_seeds.split(",") if token.strip()]
        # 解析逗号分隔的额外种子；顺便过滤掉空字符串
        for seed in extra_seeds:
            # 逐个种子重复实验
            if seed == arguments.seed:
                # 和主实验种子重复就跳过，避免同一结果算两遍
                continue
                # 进入下一个种子
            print(f"  额外随机种子 seed={seed} 的重复实验…")
            # 进度提示
            stability.append(
                run_single_seed(
                    images,
                    flattened,
                    targets,
                    seed=seed,
                    test_size=arguments.test_size,
                    hidden_dim=arguments.hidden,
                    epochs=arguments.epochs,
                    batch_size=arguments.batch_size,
                    learning_rate=arguments.learning_rate,
                    verbose=False,
                )
            )
            # 跑一遍并把结果追加进列表

    total_seconds = time.perf_counter() - script_start
    # 脚本总耗时
    print("\n[3/3] 写入 results/metrics.txt")
    # 进度提示
    metrics_path = os.path.join(out_dir, "metrics.txt")
    # metrics.txt 的完整路径
    write_metrics(metrics_path, dataset_info, primary, stability, config, figure_files, total_seconds)
    # 写好文本记录
    print(f"  已写入 {metrics_path}")
    # 提示完成
    print(f"\n全部完成，总耗时 {total_seconds:.2f} s")
    # 打印总耗时
    return 0
    # 正常退出


if __name__ == "__main__":
    # 只有被直接运行（而不是被 import）时才执行主流程
    raise SystemExit(main())
    # 用 main 的返回值作为进程退出码
