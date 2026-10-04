#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
人工特征提取器：对应李宏毅《Why Deep》PPT 中标记为"绿色 hand-crafted"的那个模块。

本模块只做"人设计好的、不含任何可学习参数"的固定变换：
把一张 8x8 的手写数字图压缩成一个低维向量——
  - 行投影：每行的像素和，8 维；
  - 列投影：每列的像素和，8 维；
  - 象限均值：把 8x8 切成四个 4x4 小块，各取平均亮度，4 维。
前 16 维构成方案 A1 的特征，16+4 维构成方案 A2 的特征。

为什么叫"手工"：这些公式（求和、取平均）完全由人拍脑袋定下来，
训练数据不会改变它们中间的任何数字，这就是 PPT 里"绿色"的含义。
如果把任务换成语音，PPT 上对应的绿色模块就是 MFCC；换成图像就是 SIFT / HOG。

用法：
    from features import handcrafted_features
    feats = handcrafted_features(images)
"""

from __future__ import annotations
# 打开延迟注解求值：类型标注写成 np.ndarray 即可，不必用字符串包起来，也不影响运行速度

import numpy as np
# 只用 NumPy 做逐元素运算；本模块刻意不依赖 sklearn，保证"特征提取是纯手工、与分类器解耦"

GRID_SIZE = 8
# 每张图是 8x8 的灰度矩阵；行投影、列投影各得到 GRID_SIZE 维，合计 16 维

PIXEL_MAX = 16.0
# load_digits 的像素取值范围是 0~16 的整数（16 表示最黑的墨迹）
# 用它做除数可以把投影值缩放到 0~1，属于"人定的固定缩放"，同样没有可学习参数

QUADRANT_SIZE = GRID_SIZE // 2
# 象限边长：8x8 切成四个 4x4 的小块，所以每块边长是 4

PROJECTION_DIM = 2 * GRID_SIZE
# 行投影 + 列投影的总维度：8 + 8 = 16

QUADRANT_DIM = 4
# 象限均值特征的维度：左上、右上、左下、右下共 4 个

FEATURE_DIM_PROJECTION_ONLY = PROJECTION_DIM
# 方案 A1 的特征维度：只用 16 维投影特征

FEATURE_DIM_WITH_QUADRANTS = PROJECTION_DIM + QUADRANT_DIM
# 方案 A2 的特征维度：16 维投影 + 4 维象限均值 = 20 维


def row_projection(images: np.ndarray) -> np.ndarray:
    """计算每张图的行投影：把每一行的像素加起来，得到一个 8 维向量。

    参数：
        images：形状 (n, 8, 8) 的浮点数组，第 i 张图是 images[i]。
    返回：形状 (n, 8) 的浮点数组，第 j 列表示"第 j 行一共有多少墨"。

    为什么这么设计：人观察手写数字时，会觉得"这个字上半部分笔画重、下半部分轻"，
    行投影正是把这个直觉写成了一个固定公式。代价是它丢掉了"同一行里墨迹出现在左边还是右边"，
    这个信息损失是后面分析方案 A 准确率偏低的关键原因。
    """
    return images.sum(axis=2) / PIXEL_MAX
    # axis=2 表示对每一行内部的列方向求和（把一行 8 个像素压成 1 个数）
    # 再统一除以 PIXEL_MAX，让特征落在 0~1，便于后续逻辑回归的优化器收敛


def column_projection(images: np.ndarray) -> np.ndarray:
    """计算每张图的列投影：把每一列的像素加起来，得到一个 8 维向量。

    参数：
        images：形状 (n, 8, 8) 的浮点数组。
    返回：形状 (n, 8) 的浮点数组，第 j 列表示"第 j 列一共有多少墨"。

    与行投影的关系：两者互为转置意义上的对称操作，合起来 16 维，
    正好等于 PPT 上"人工设计特征"这一绿色方块的输出维度（本实验自行设定的低维版本）。
    """
    return images.sum(axis=1) / PIXEL_MAX
    # axis=1 表示对每张图内部的"行方向"求和，等价于把整张图沿竖直方向压扁成 8 个列和


def quadrant_means(images: np.ndarray) -> np.ndarray:
    """计算每张图四个象限的平均亮度，得到 4 维特征。

    参数：
        images：形状 (n, 8, 8) 的浮点数组。
    返回：形状 (n, 4) 的浮点数组，列顺序固定为 [左上, 右上, 左下, 右下]。

    为什么要加这四个数：投影特征只关心"整行/整列的墨量"，完全不管空间分布。
    象限均值是最粗糙的一层空间信息——它能区分"墨偏左上"和"墨偏右下"，
    但分辨率只有 4 个格子，属于典型的"人工设计、信息量有限"的特征。
    """
    top = images[:, :QUADRANT_SIZE, :]
    # 取每张图的上半部分（前 4 行），形状 (n, 4, 8)
    bottom = images[:, QUADRANT_SIZE:, :]
    # 取每张图的下半部分（后 4 行），形状 (n, 4, 8)
    top_left = top[:, :, :QUADRANT_SIZE].mean(axis=(1, 2))
    # 左上块：上半部分再取左 4 列，然后在 (行, 列) 两个轴上取平均，得到 (n,) 的一列
    top_right = top[:, :, QUADRANT_SIZE:].mean(axis=(1, 2))
    # 右上块：上半部分取右 4 列后求均值
    bottom_left = bottom[:, :, :QUADRANT_SIZE].mean(axis=(1, 2))
    # 左下块：下半部分取左 4 列后求均值
    bottom_right = bottom[:, :, QUADRANT_SIZE:].mean(axis=(1, 2))
    # 右下块：下半部分取右 4 列后求均值
    return np.stack([top_left, top_right, bottom_left, bottom_right], axis=1) / PIXEL_MAX
    # 按固定的 [左上, 右上, 左下, 右下] 顺序拼成 (n, 4)，再除以 PIXEL_MAX 缩放到 0~1


def handcrafted_features(images: np.ndarray, use_quadrants: bool = True) -> np.ndarray:
    """把 (n, 8, 8) 的图像批量转换成人工特征矩阵。

    参数：
        images：形状 (n, 8, 8) 的浮点数组，像素值取原始尺度 0~16（load_digits 的原生范围）。
        use_quadrants：True 时输出 20 维（投影 + 象限均值），False 时只输出 16 维投影。
    返回：形状 (n, 16) 或 (n, 20) 的浮点数组，每个元素都在 0~1 之间。

    注意输入是原始 0~16 的像素值：归一化（除以 PIXEL_MAX）是在本函数内部、
    由各个投影子函数顺手完成的，这样调用方只需要把 load_digits 的图直接丢进来即可，
    也保证了"人工特征"和"喂给神经网络"这两条路线的像素预处理口径可以分开对比。

    这个函数整体就是 PPT 上的绿色方块：输入是原始图像，输出是固定的人工特征，
    里面没有任何需要训练的参数；接在它后面的分类器才是 PPT 上的蓝色方块。
    """
    rows = row_projection(images)
    # 先算 8 维行投影
    cols = column_projection(images)
    # 再算 8 维列投影
    parts = [rows, cols]
    # 用列表收集要拼接的各段特征，顺序决定了最终特征向量的每一维含义
    if use_quadrants:
        # 只有需要 20 维版本时才追加象限均值
        parts.append(quadrant_means(images))
        # 追加 4 维象限均值
    return np.concatenate(parts, axis=1)
    # 沿特征维拼接，得到 (n, 16) 或 (n, 20)


def feature_names(use_quadrants: bool = True) -> list[str]:
    """返回人工特征每一维的中文名字，供文档和调试信息使用。

    参数：
        use_quadrants：是否包含象限均值特征，需与 handcrafted_features 的取值一致。
    返回：长度等于特征维度的字符串列表，顺序与 handcrafted_features 的输出一一对应。
    """
    names = [f"行投影_第{i}行" for i in range(GRID_SIZE)]
    # 前 8 维的名字：行投影_第0行 ~ 行投影_第7行
    names += [f"列投影_第{i}列" for i in range(GRID_SIZE)]
    # 接着 8 维：列投影_第0列 ~ 列投影_第7列
    if use_quadrants:
        # 20 维版本再多出 4 个象限的名字
        names += ["象限均值_左上", "象限均值_右上", "象限均值_左下", "象限均值_右下"]
        # 顺序必须与 quadrant_means 内部的 stack 顺序严格一致，否则名字会对错位置
    return names


def normalize_features(features: np.ndarray) -> np.ndarray:
    """把一批特征向量按行做 L2 归一化，只用于算相似度，不用于训练。

    参数：
        features：形状 (n, d) 的浮点数组。
    返回：形状 (n, d) 的浮点数组，每一行都是单位向量（全零行会被替换成零向量）。

    为什么要归一化：接下来要比较"两张图的人工特征有多像"，
    而不同数字的墨迹总量差别很大（比如 8 比 1 黑得多），不归一化的话
    相似度会主要反映"墨的多少"而不是"形状像不像"。
    """
    norms = np.linalg.norm(features, axis=1, keepdims=True)
    # 逐行求 L2 范数，keepdims=True 保证形状是 (n, 1)，才能和 (n, d) 广播相除
    norms = np.where(norms == 0.0, 1.0, norms)
    # 把范数为 0 的行保护成 1，避免除以 0 产生 NaN（digits 里几乎不会出现全黑/全空图）
    return features / norms
    # 每一行除以自己的长度，得到单位向量


def most_similar_cross_label_pair(features: np.ndarray, targets: np.ndarray) -> tuple[int, int, float]:
    """在所有样本里找出"人工特征最像、但真实标签不同"的一对样本。

    参数：
        features：形状 (n, d) 的人工特征矩阵。
        targets：形状 (n,) 的整数标签。
    返回：(下标 a, 下标 b, 余弦相似度)，满足 targets[a] != targets[b] 且相似度最大。

    为什么需要这个函数：它把"人工特征丢信息"这件事从一个抽象说法变成一张具体的图。
    如果两个不同数字的 16~20 维人工特征几乎一样，那就说明这两类在特征空间里
    已经重叠了，无论后面的逻辑回归多聪明都不可能把它们分开，
    这正是 PPT 里"人工设计特征限制了最终上限"的直观证据。
    """
    unit = normalize_features(features)
    # 先把特征按行归一化，这样矩阵乘法得到的就是余弦相似度
    similarity = unit @ unit.T
    # (n, d) 乘 (d, n) 得到 (n, n) 的相似度矩阵，第 (i, j) 个元素是样本 i 与样本 j 的余弦相似度
    same_label = targets[:, None] == targets[None, :]
    # 广播比较，得到 (n, n) 的布尔矩阵，同类别的位置为 True
    similarity = np.where(same_label, -np.inf, similarity)
    # 把同类别（以及自己和自己）的位置全部设成负无穷，这样 argmax 只会挑出跨类别的一对
    best = int(np.argmax(similarity))
    # 在展平后的 (n*n,) 上取最大值的下标
    a, b = divmod(best, similarity.shape[1])
    # 把展平下标还原成矩阵的行号 a 与列号 b，即两个样本的编号
    return a, b, float(similarity[a, b])
    # 返回样本下标和它们的余弦相似度（转成 Python float 便于打印）
