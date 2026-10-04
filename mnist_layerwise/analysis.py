#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
把「四种表示到底有多好」变成可读数字的工具箱。

本文件回答一个问题：四宫格图里的“十类混杂 → 十类成团”是肉眼印象，
能不能用数字把它量出来？于是对每一层表示算三类指标：

  1) silhouette score（轮廓系数）：同一个数字的点彼此有多近、离别的数字有多远。
     取值 -1~1，越大说明“同类抱团、异类分离”，是“是否成团”最直接的量化。
  2) 1-NN 准确率：把该层表示当成特征，用最简单的“最近邻分类器”分类。
     如果表示本身好，连 1-NN 这种几乎不学习的分类器都能分对；
     它衡量的正是 PPT 想说的那个意思——“表示好了，后面的分类器就很轻松”。
  3) 类内散度 / 类间中心距：类内平均离散程度除以十个类中心之间的平均距离。
     越小说明团越紧、彼此越开。它和 silhouette 看的是同一件事，
     但计算方式更原始（不涉及“最近邻类”的选择），可以互相印证。

另外提供把高维表示降到 2 维的两个函数：PCA 与 t-SNE。
PCA 是线性降维，保留的是方差最大的方向，是 PPT 原页用的做法；
t-SNE 是非线性降维，专门保留局部邻域结构，画出来的团通常比 PCA 更“炸开”，
用来做对照可以说明“逐层成团”不是 PCA 这一种投影方法的错觉。
"""

from __future__ import annotations
# 打开延迟注解求值，类型标注里可以直接写 np.ndarray

import numpy as np
# 数值计算
from sklearn.decomposition import PCA
# 线性降维：把 64/128/32 维表示压到 2 维来画散点
from sklearn.manifold import TSNE
# 非线性降维：按局部邻域结构把表示铺到 2 维，作为 PCA 的对照
from sklearn.metrics import silhouette_score
# 轮廓系数：本实验最核心的“成团程度”量化指标
from sklearn.metrics.pairwise import euclidean_distances
# 成对欧氏距离矩阵；1-NN 与类内/类间距离都建立在它上面


def _check_labels(labels: np.ndarray) -> None:
    """检查标签是否满足 silhouette / 1-NN 的基本前提，不满足就直接报错。

    参数：
        labels：形状 (n,) 的整数标签。
    返回：无；不满足条件时抛 ValueError。

    为什么不写 try/except 把异常吞掉：这两个指标只有在“多类且每类都有若干样本”时才有意义。
    如果哪一层表示意外退化（例如某层神经元全被 ReLU 压成 0，导致所有样本重合），
    我们希望程序立刻炸掉并指出原因，而不是打印出一个看似正常、实则毫无意义的数字。
    """
    unique = np.unique(labels)
    # 找出标签里实际出现的类别
    if unique.size < 2:
        # 只有一个类别时，silhouette 无定义、1-NN 也退化成“永远答对”
        raise ValueError(f"标签至少要有 2 个类别，实际只有 {unique.size} 个")
        # 抛出明确错误
    counts = np.bincount(labels)
    # 统计每个类别出现的样本数
    if counts[counts > 0].min() < 2:
        # silhouette 要求每个类别至少有 2 个样本，否则轮廓系数无定义
        raise ValueError("每个类别至少要有 2 个样本，否则 silhouette 无定义")
        # 抛出明确错误
    if labels.shape[0] != counts.sum():
        # 防御性检查：标签里不该出现负数
        raise ValueError("标签中存在非法取值（如负数）")
        # 抛出明确错误


def silhouette_of(representation: np.ndarray, labels: np.ndarray) -> float:
    """计算某一层表示的轮廓系数。

    参数：
        representation：形状 (n, d) 的表示矩阵，可以是 input/h1/h2/h3 中的任意一层。
        labels：形状 (n,) 的真实数字标签（0~9）。
    返回：轮廓系数，取值 -1~1；越大越好。

    公式（单个样本 i 的轮廓系数）：
        s(i) = (b(i) - a(i)) / max(a(i), b(i))
    其中 a(i) 是 i 到“自己这一类”其他点的平均距离（类内凝聚度，越小越好），
    b(i) 是 i 到“离自己最近的那个其他类”所有点的平均距离（类间分离度，越大越好）。
    整个数据集的 silhouette score 就是所有样本 s(i) 的平均值：
      - 接近 1  → 同类点紧紧抱团，且离最近的其他类很远（十类各抱一团）；
      - 接近 0  → 两类边界处互相穿插（十类混杂）；
      - 小于 0  → 点离别的类反而比离自己的类更近（表示已经很糟糕）。
    注意这个指标只用真实标签来“算分”，不参与任何训练，所以可以用来看任意一层的表示。
    """
    _check_labels(labels)
    # 先确认标签合法，避免算出无意义的数值
    return float(silhouette_score(representation, labels, metric="euclidean"))
    # sklearn 默认就是欧氏距离的轮廓系数；转成 Python float 便于写进文本报告


def leave_one_out_one_nn(representation: np.ndarray, labels: np.ndarray) -> float:
    """用“留一法”评估 1-NN（最近邻）分类器在该层表示上的准确率。

    参数：
        representation：形状 (n, d) 的表示矩阵。
        labels：形状 (n,) 的真实标签。
    返回：0~1 的准确率。

    做法：对每个样本，把它自己从参考集中拿掉，在剩下的 n-1 个样本里找最近邻，
    用最近邻的标签作为预测，再和真标签比对。
    为什么用留一法而不是划分训练/测试：n 只有 1797，划分会浪费样本；
    留一法把每个样本都当过考点，指标更稳定，也完全等价于“用 n-1 个样本做参考的 1-NN”。
    实现上直接算成对距离矩阵，再把对角线（自己到自己，距离恒为 0）设成无穷大就能排除自己，
    比调用 sklearn 的交叉验证循环快得多、也更直白。
    """
    _check_labels(labels)
    # 确认标签合法
    distances = euclidean_distances(representation)
    # 算出 (n, n) 的成对欧氏距离矩阵，第 (i, j) 项是样本 i 与 j 的距离
    np.fill_diagonal(distances, np.inf)
    # 把对角线设成无穷大：这一项是“样本到它自己”，不排除掉的话最近邻永远是它自己
    nearest = distances.argmin(axis=1)
    # 每行取距离最小的那一列下标，就是该样本的最近邻编号
    return float((labels[nearest] == labels).mean())
    # 最近邻的标签与真标签一致的比例就是准确率


def same_label_nearest_neighbor_rate(coordinates: np.ndarray, labels: np.ndarray) -> float:
    """在二维投影坐标上算“最近的那个邻居是不是同一个数字”的比例。

    参数：
        coordinates：形状 (n, 2) 的二维投影坐标（PCA 或 t-SNE 的输出）。
        labels：形状 (n,) 的真实标签。
    返回：0~1 的比例，越大说明图上同色点越倾向挨在一起。

    为什么需要这个指标：四宫格图是给人看的，“肉眼可见”本身很难写进表格。
    但在二维平面上，“离我最近的点和我同色”这件事恰好就是眼睛判断“成团”的依据——
    颜色互相穿插时，最近的邻居大概率是别的数字，这个比例就低；
    十类各抱一团时，最近的邻居几乎总是同一个数字，这个比例就接近 1。
    于是它把“图上看起来团不团”翻译成了一个能写进报告的、可复现的数字。
    注意二维投影是有损的（64 维压到 2 维必然丢信息），所以这个数字只用于描述图，
    不能当作模型性能指标——模型性能看的是高维表示上的 1-NN 与最终准确率。
    """
    _check_labels(labels)
    # 确认标签合法
    distances = euclidean_distances(coordinates)
    # 二维坐标的成对距离矩阵
    np.fill_diagonal(distances, np.inf)
    # 排除“自己到自己”
    nearest = distances.argmin(axis=1)
    # 每个点在二维平面上的最近邻
    return float((labels[nearest] == labels).mean())
    # 最近邻同色的比例


def split_one_nn(
    train_representation: np.ndarray,
    train_labels: np.ndarray,
    test_representation: np.ndarray,
    test_labels: np.ndarray,
) -> float:
    """用“训练集当参考、测试集当考点”评估 1-NN 准确率。

    参数：
        train_representation：训练集样本的表示，形状 (n_train, d)。
        train_labels：训练集真标签。
        test_representation：测试集样本的表示，形状 (n_test, d)。
        test_labels：测试集真标签。
    返回：0~1 的测试准确率。

    和留一法的区别：参考集里全是网络训练时见过的样本，考点全是没见过的。
    所以这个数字既可用来衡量表示质量，又顺带检查了“表示的泛化性”——
    如果深层表示在训练样本上抱团、在测试样本上却散开，这里的数字会明显低于留一法。
    这正好回应一个可能的质疑：“深层团得紧，会不会只是网络把训练样本背下来了？”
    """
    _check_labels(test_labels)
    # 确认标签合法
    distances = euclidean_distances(test_representation, train_representation)
    # 形状 (n_test, n_train)：每个测试样本到每个训练样本的距离
    nearest = distances.argmin(axis=1)
    # 每行取最近的那个训练样本
    return float((train_labels[nearest] == test_labels).mean())
    # 用最近邻训练样本的标签预测测试样本，与真标签比对


def scatter_ratio(representation: np.ndarray, labels: np.ndarray) -> float:
    """计算“类内散度 / 类间中心距”，衡量团的紧致程度。

    参数：
        representation：形状 (n, d) 的表示矩阵。
        labels：形状 (n,) 的真实标签。
    返回：比值，越小表示“团越紧、彼此越远”。

    计算方式：
        类内散度  = 所有样本到“自己类中心”的距离的平均值；
        类间中心距 = 十个类中心两两距离的平均值；
        比值 = 类内散度 / 类间中心距。
    为什么再加这个指标：轮廓系数在“类的形状被拉长”时可能偏低，
    而这个比值只看中心与平均散布，非常粗糙但极其稳定，两者同向变化才让人放心。
    """
    _check_labels(labels)
    # 确认标签合法
    digits = np.unique(labels)
    # 实际出现的类别编号（0~9）
    centroids = np.stack([representation[labels == digit].mean(axis=0) for digit in digits])
    # 逐个类别求平均向量，得到 (类别数, d) 的类中心矩阵
    row_of_label = np.searchsorted(digits, labels)
    # 把原始标签映射成 0..9 的连续行号，方便用花式索引取“每个样本自己的类中心”
    within = float(np.linalg.norm(representation - centroids[row_of_label], axis=1).mean())
    # 每个样本减去它所属类的中心再求范数，就是到类中心的距离，最后取平均
    between = float(euclidean_distances(centroids).mean())
    # 类中心两两距离的平均值（含对角线上“自己到自己”的 0，作为统一口径的常数下压，不影响层间比较）
    return within / between
    # 比值越小，说明同类越紧、异类越开


def pca_2d(representation: np.ndarray, seed: int = 0) -> np.ndarray:
    """把一层表示用 PCA 线性降到 2 维。

    参数：
        representation：形状 (n, d) 的表示矩阵。
        seed：PCA 内部随机性种子（本实现用完整 SVD，实际是确定性的，保留参数是为了口径统一）。
    返回：形状 (n, 2) 的二维坐标。

    为什么先做这个：PPT 原页用的就是 PCA。
    PCA 找的是方差最大的两个正交方向——方差大不等于“类别分得开”，
    所以 input 面板上虽然铺得很开，颜色却是混的；这恰恰是 PPT 想展示的对比。
    """
    return PCA(n_components=2, random_state=seed).fit_transform(representation)
    # 拟合 2 个主成分并把数据投影过去


def tsne_2d(representation: np.ndarray, seed: int = 42, perplexity: float = 30.0) -> np.ndarray:
    """把一层表示用 t-SNE 非线性降到 2 维。

    参数：
        representation：形状 (n, d) 的表示矩阵。
        seed：随机种子，固定后同一层的图可复现。
        perplexity：近邻规模，可以粗略理解为“每个点大概关心多少个邻居”，默认 30。
    返回：形状 (n, 2) 的二维坐标。

    和 PCA 的区别：t-SNE 不保留全局方差，只努力让“原本就是邻居的点”在 2 维里依然是邻居。
    所以它画出来的团更明显，代价是坐标轴没有物理含义、团与团之间的距离不可解释。
    本实验用它做对照：如果某一层的 t-SNE 图也没能分出团，那就说明表示真的不行，
    而不是 PCA 这种线性投影把它压坏了。
    注意 n=1797 属于小规模，t-SNE 直接跑完全可行（本机每层约 5~15 秒）。
    """
    return TSNE(
        n_components=2,
        perplexity=perplexity,
        init="pca",
        learning_rate="auto",
        max_iter=1000,
        random_state=seed,
    ).fit_transform(representation)
    # 用 PCA 结果初始化可以让每次运行结果更稳定，也避免随机初始化的“假团”


def class_centers_2d(coordinates: np.ndarray, labels: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """在 2 维投影里算出每个数字类的中心位置，供图上标注数字用。

    参数：
        coordinates：形状 (n, 2) 的二维坐标。
        labels：形状 (n,) 的真实标签。
    返回：(digits, centers)
        digits：出现的类别，形状 (k,)；
        centers：对应类别在 2 维里的中心坐标，形状 (k, 2)。
    """
    digits = np.unique(labels)
    # 实际出现的类别编号，按从小到大排序
    centers = np.stack([coordinates[labels == digit].mean(axis=0) for digit in digits])
    # 对每个类别取二维坐标的平均值，作为画数字标记的位置
    return digits, centers
    # 返回类别编号与对应的中心坐标
