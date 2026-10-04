#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
李宏毅《Why Deep Learning?》棋盘格（checkerboard）实验的 NumPy 复现。

本脚本用纯 NumPy 手写一个多层感知机（MLP），在人工合成的"菱形棋盘格"二分类任务上，
对比 {1 个隐藏层} 与 {3 个隐藏层} 两种网络在 {100000, 20000} 个训练样本下的表现，
输出 4 组测试准确率、训练日志，以及 4 张"标准棋盘格 vs 模型预测"的对比图。

为什么不用 PyTorch：本机 /c/msys64/ucrt64/bin/python 未安装 torch；本实验网络很小
（输入 2 维、64 单元隐藏层、输出 1 维），用 NumPy 手写前向/反向传播代码量可控，
且没有框架版本依赖风险，结果完全可复现。

用法（Git Bash）：
    /c/msys64/ucrt64/bin/python checkerboard.py
    /c/msys64/ucrt64/bin/python checkerboard.py --samples 100000 20000 --layers 1 3 --epochs 200
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
import matplotlib

# 必须在 import pyplot 之前指定无界面后端，否则在无显示环境下保存图片会失败
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.colors import ListedColormap

# ---------------------------------------------------------------------------
# 0. 全局常量与中文字体设置
# ---------------------------------------------------------------------------

# 数据所在正方形区域的半边长：训练/测试点在 [-1, 1] x [-1, 1] 上均匀采样
DOMAIN = 1.0
# 默认棋盘格"格距"：两个旋转坐标 (x1+x2) 与 (x1-x2) 各自的区间长度
# 该值越小格子越密，水平方向可见的菱形列数约为 2 / cell_size = 4 列
DEFAULT_CELL_SIZE = 0.5
# 隐藏层神经元个数的默认值，1 隐层与 3 隐层网络都使用同样的宽度
DEFAULT_HIDDEN = 64
# 独立测试集大小：用足够大的新样本估计泛化准确率，避免测试集本身的随机波动
DEFAULT_TEST_SIZE = 20000
# 训练轮数上限；配合"收敛早停"使用，正常情况下会在收敛后自动提前停止
DEFAULT_EPOCHS = 2500
# 批次大小；0 表示全批量梯度下降
DEFAULT_BATCH = 0
# Adam 优化器的初始学习率
DEFAULT_LR = 0.01
# 第一层权重的初始化放大倍数，等价于把输入尺度放大同样倍数
DEFAULT_INIT_SCALE = 4.0
# 画决策区域时在 [-1,1]^2 上取的网格分辨率（resolution x resolution 个点）
DEFAULT_RESOLUTION = 320

# 决策区域配色：标签 0 -> 蓝色，标签 1 -> 红色，与 PPT 上的红蓝棋盘一致
BLUE = "#2f5fd0"
RED = "#dd3f3f"
# 离散标签使用的颜色表（用于"标准棋盘格"那一栏）
CMAP_TRUTH = ListedColormap([BLUE, RED])


def setup_chinese_font() -> str | None:
    """为 matplotlib 注册一个系统中文字体，避免图中的中文变成方块。

    参数：无。
    返回：成功时返回字体名，找不到任何中文字体时返回 None（图仍会生成，只是中文可能显示为方框）。
    为什么需要：matplotlib 自带字体不含 CJK 字形，图注里的中文必须显式指定字体才不会乱码。
    """
    # 依次尝试常见的 Windows 中文字体，取第一个存在的
    candidates = [
        r"C:\Windows\Fonts\msyh.ttc",
        r"C:\Windows\Fonts\simhei.ttf",
        r"C:\Windows\Fonts\Deng.ttf",
    ]
    for path in candidates:
        # 路径不存在就跳过，换下一个候选字体
        if not os.path.exists(path):
            continue
        # 把字体文件注册进 matplotlib 的字体管理器，之后就能按名字引用它
        font_manager.fontManager.addfont(path)
        # 从字体文件自身读出字体族名（如 "Microsoft YaHei"），而不是硬编码字符串
        name = font_manager.FontProperties(fname=path).get_name()
        plt.rcParams["font.family"] = name
        # 让负号按普通减号渲染，否则中文默认字体下坐标轴的负号会变成方块
        plt.rcParams["axes.unicode_minus"] = False
        return name
    return None


# ---------------------------------------------------------------------------
# 1. 数据集：按题设公式生成"菱形棋盘格"标签
# ---------------------------------------------------------------------------


def checkerboard_label(X: np.ndarray, cell_size: float = DEFAULT_CELL_SIZE) -> np.ndarray:
    """按菱形棋盘格函数给一批二维点打 0/1 标签。

    参数：
        X：形状 (n, 2) 的浮点数组，每行是一个点 (x1, x2)。
        cell_size：格距 s，控制棋盘格的疏密，也等价于"格子尺寸"这个可配置参数。
    返回：形状 (n,) 的 0/1 浮点数组，0 表示蓝类、1 表示红类。

    标签函数的数学形式（本实验的核心定义）：
        y(x1, x2) = ( floor((x1 + x2) / s) + floor((x1 - x2) / s) ) mod 2
    几何含义：先把坐标轴旋转 45 度，得到 u = (x1+x2)/s 与 v = (x1-x2)/s；
    再在 (u, v) 平面上按"整数格子的行列下标之和的奇偶"上色，
    于是每个格子在原 (x1, x2) 平面上是一个菱形（正方形转了 45 度）。
    """
    # 旋转 45 度并归一化格距：u 沿"左下-右上"方向，即原坐标系的正对角线方向
    u = (X[:, 0] + X[:, 1]) / cell_size
    # v 沿"左上-右下"方向，与 u 垂直，两者共同构成旋转后的坐标系
    v = (X[:, 0] - X[:, 1]) / cell_size
    # 对 u、v 分别取下整得到"第几个格子"的整数下标；注意 floor 对负数向下取整仍然正确
    iu = np.floor(u).astype(np.int64)
    # 同上，得到 v 方向的格子下标
    iv = np.floor(v).astype(np.int64)
    # 两个下标之和的奇偶决定颜色：和为偶数取蓝(0)，为奇数取红(1)，这就是"棋盘格"交替
    parity = (iu + iv) % 2
    # 转成 float 便于直接和交叉熵损失里的浮点标签运算
    return parity.astype(np.float64)


def sample_points(n: int, rng: np.random.Generator) -> np.ndarray:
    """在 [-1, 1] x [-1, 1] 上均匀采样 n 个二维点。

    参数：
        n：采样点个数。
        rng：numpy 随机数生成器，显式传入以保证可复现。
    返回：形状 (n, 2) 的浮点数组。
    为什么这么采样：题设的目标函数定义域就是 [-1,1]^2，均匀采样能保证每类、
    每个格子的样本密度都一致，不会因为采样偏差人为改变任务难度。
    """
    # 两个维度独立均匀采样；区间半宽为 DOMAIN，均匀分布的下界是 -DOMAIN
    return rng.uniform(-DOMAIN, DOMAIN, size=(n, 2))


def build_dataset(n: int, cell_size: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """生成 (特征, 标签) 数据集。

    参数：
        n：样本数。
        cell_size：棋盘格格距，传给标签函数。
        seed：随机种子，固定种子保证每次运行得到完全相同的数据集。
    返回：(X, y)，X 形状 (n, 2)，y 形状 (n,)。
    为什么是程序生成而不是网上下载：该函数（棋盘格）没有现成的公开数据文件，
    PPT 里的实验本身就是用程序在 [-1,1]^2 上撒点、按公式打标签得到的合成数据；
    合成数据的好处是（1）标签无噪声，泛化差距只来自模型本身；（2）可以任意
    控制训练样本量（10 万 / 2 万）；（3）测试集可以无限大且独立同分布。
    """
    # 固定种子构造随机数生成器，这是复现性的唯一来源
    rng = np.random.default_rng(seed)
    # 先撒点
    X = sample_points(n, rng)
    # 再按棋盘格公式打标签（无噪声，标签完全由坐标决定）
    y = checkerboard_label(X, cell_size)
    return X, y


# ---------------------------------------------------------------------------
# 2. 模型：纯 NumPy 手写的多层感知机
# ---------------------------------------------------------------------------

# 交叉熵里的极小量，防止 log(0) 产生 -inf 把整轮训练污染成 NaN
EPS = 1e-12


def sigmoid(z: np.ndarray) -> np.ndarray:
    """数值稳定的 logistic 函数，把实数映射到 (0, 1) 区间。

    参数：z，任意形状的实数数组。
    返回：与 z 同形状的概率值。
    为什么分两段写：exp(-z) 在 z 很负时会上溢，exp(z) 在 z 很正时会上溢，
    按符号分段可以保证指数部分的参数永远 <= 0，从而不会溢出。
    """
    # 预分配输出数组，保持与输入同形状
    out = np.empty_like(z, dtype=np.float64)
    # 正半轴用 1/(1+exp(-z)) 形式，此时 exp 的参数为负、安全
    pos = z >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-z[pos]))
    # 负半轴改用等价形式 exp(z)/(1+exp(z))，同样让 exp 的参数为负
    neg = ~pos
    ez = np.exp(z[neg])
    out[neg] = ez / (1.0 + ez)
    return out


class MLP:
    """全连接前馈网络，隐藏层激活可选用 tanh 或 sigmoid，输出层为单个 sigmoid。

    结构：2 -> [hidden] * n_hidden_layers -> 1，二分类，输出是 P(y=1|x)。
    """

    def __init__(self, hidden_sizes, seed: int, activation: str = "tanh",
                 first_layer_scale: float = 1.0):
        """初始化权重和偏置。

        参数：
            hidden_sizes：可迭代对象，每个元素是对应隐藏层的神经元个数，如 [64] 或 [64,64,64]。
            seed：初始化随机种子。
            activation：隐藏层激活函数名，"tanh" 或 "sigmoid"。
            first_layer_scale：把第一层权重整体放大的倍数（等价于放大输入尺度）。
        为什么用 tanh 作默认：tanh 以 0 为中心、梯度在饱和前更稳，
        且"折叠空间"的直觉（把输入压成一段段 S 形再拼接）用 tanh 最容易观察和收敛；
        sigmoid 恒正、容易饱和，同等轮数下往往收敛更慢。
        为什么需要 first_layer_scale：输入只在 [-1,1]、而目标函数的最小特征尺度是 0.25，
        用标准 Glorot 初始化时第一层净输入的标准差只有 0.15 左右，tanh 几乎工作在
        线性区，整个网络退化成线性模型。放大第一层权重能让 tanh 一开始就处在
        非线性区，从而产生"折线/折叠"能力。这是本实验能否复现 PPT 现象的关键超参。
        """
        # 保存激活函数名，前向和反向都要用
        self.activation = activation
        # 逐层的权重矩阵列表，第 i 个矩阵形状是 (第 i 层输入维度, 第 i 层输出维度)
        self.W: list[np.ndarray] = []
        # 逐层的偏置向量列表，形状 (输出维度,)
        self.b: list[np.ndarray] = []
        # 拼接出每一层的宽度：输入 2 维，中间是各隐藏层，最后输出 1 维
        sizes = [2] + list(hidden_sizes) + [1]
        # 用独立的生成器保证初始化可复现
        rng = np.random.default_rng(seed)
        # 逐层初始化：fan_in 是本层输入个数，fan_out 是本层输出个数
        for fan_in, fan_out in zip(sizes[:-1], sizes[1:]):
            # Glorot 均匀初始化：上下界由输入输出维度之和决定，使前向方差大致保持稳定
            limit = np.sqrt(6.0 / (fan_in + fan_out))
            # 权重在 [-limit, limit] 上均匀取值
            self.W.append(rng.uniform(-limit, limit, size=(fan_in, fan_out)))
            # 偏置初始化为 0：对称性已由随机权重打破，偏置再随机只会让初始输出偏移
            self.b.append(np.zeros(fan_out))
        # 按需放大第一层权重，让输入层一开始就落入激活函数的非线性区
        self.W[0] = self.W[0] * first_layer_scale
        # 记录"权重矩阵的个数"，也就是层数（含输出层）
        self.n_layers = len(self.W)
        # 记录第一层缩放倍数，写入日志时要用
        self.first_layer_scale = first_layer_scale

    def _activate(self, z: np.ndarray) -> np.ndarray:
        """按设定对隐藏层净输入做非线性变换。"""
        # tanh 输出 (-1,1)，零均值
        if self.activation == "tanh":
            return np.tanh(z)
        # sigmoid 输出 (0,1)，作为备选配置
        return sigmoid(z)

    def _activate_grad(self, a: np.ndarray) -> np.ndarray:
        """给出激活函数在"激活输出 a"处的导数，用于反向传播。"""
        # tanh 的导数写成 a 的函数：1 - tanh^2
        if self.activation == "tanh":
            return 1.0 - a * a
        # sigmoid 的导数写成 a 的函数：a * (1 - a)
        return a * (1.0 - a)

    def forward(self, X: np.ndarray, keep_cache: bool = False):
        """前向传播。

        参数：
            X：形状 (n, 2) 的输入。
            keep_cache：为 True 时额外返回中间激活值列表（反向传播要用）。
        返回：keep_cache=False 时返回输出概率 p（形状 (n,1)）；
              keep_cache=True 时返回 (p, acts)。
        """
        # acts[0] 是网络输入，acts[i] 是第 i 层的输出；最后一层输出未过 sigmoid 的 logit
        acts = [X]
        # 当前层的输入，从原始特征开始
        a = X
        # 遍历所有隐藏层（权重矩阵数为 n_layers，最后一层是输出层）
        for i in range(self.n_layers - 1):
            # 线性变换：z = a W + b
            z = a @ self.W[i] + self.b[i]
            # 非线性激活，得到本层输出
            a = self._activate(z)
            # 缓存本层输出，供反向传播使用
            acts.append(a)
        # 输出层：只有线性变换，得到 logit
        logit = a @ self.W[-1] + self.b[-1]
        # logit 也放进 acts，反向传播时它是输出层的"净输入"
        acts.append(logit)
        # 用 sigmoid 把 logit 变成概率 P(y=1|x)
        p = sigmoid(logit)
        # 训练时需要缓存，预测时不需要（省内存）
        if keep_cache:
            return p, acts
        return p

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """返回正类概率，形状 (n, 1)。"""
        # 直接复用前向传播，不保留中间量
        return self.forward(X, keep_cache=False)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """返回 0/1 预测标签，形状 (n,)，阈值取 0.5（等价于 logit 的符号）。"""
        # 概率 >= 0.5 判为 1，否则判为 0；ravel 保证结果是一维 (n,)，
        # 否则后面和形状 (n,) 的标签比较会发生广播，得到 (n,n) 的矩阵而算错准确率
        return (self.predict_proba(X).ravel() >= 0.5).astype(np.float64)

    def accuracy(self, X: np.ndarray, y: np.ndarray) -> float:
        """计算在给定数据集上的分类准确率。

        参数：X 形状 (n,2)，y 形状 (n,) 的 0/1 标签。
        返回：准确率，0 到 1 之间的浮点数。
        """
        # 预测和标签都拉平成一维，逐元素比较后取相等比例
        return float(np.mean(self.predict(X) == np.asarray(y).ravel()))

    def loss_and_grads(self, X: np.ndarray, y: np.ndarray):
        """计算一个批次的二分类交叉熵损失及其对所有参数的梯度。

        参数：
            X：形状 (batch, 2)。
            y：形状 (batch, 1) 的 0/1 标签（已 reshape 成列向量）。
        返回：(loss, grads_W, grads_b)，其中 loss 是标量，后两者是与 self.W/self.b 同结构的列表。
        """
        # 批大小，用来做平均（损失和梯度都按样本数取平均）
        n = X.shape[0]
        # 前向传播并保留每层激活，反向传播要用
        p, acts = self.forward(X, keep_cache=True)
        # 交叉熵损失：把 y 转成列向量以便和 p 逐元素运算，加 EPS 防止 log(0) 变成 -inf
        loss = -np.mean(y.reshape(-1, 1) * np.log(p + EPS)
                        + (1.0 - y.reshape(-1, 1)) * np.log(1.0 - p + EPS))
        # 输出层 logit 的梯度：sigmoid + 交叉熵组合后化简为 (p - y) / n，非常简洁
        delta = (p - y.reshape(-1, 1)) / n
        # 预分配梯度列表，长度等于层数
        grads_W: list[np.ndarray] = [None] * self.n_layers
        grads_b: list[np.ndarray] = [None] * self.n_layers
        # 从最后一层往前逐层回传
        for i in range(self.n_layers - 1, -1, -1):
            # 权重梯度 = 本层输入（列）与误差信号（行）的外积
            grads_W[i] = acts[i].T @ delta
            # 偏置梯度 = 误差信号在批次维度上求和
            grads_b[i] = delta.sum(axis=0)
            # 还没有传到第 0 层时，把误差信号继续往前传
            if i > 0:
                # 先经过权重矩阵的转置把误差映射回上一层，再乘上一层激活函数的导数
                delta = (delta @ self.W[i].T) * self._activate_grad(acts[i])
        return float(loss), grads_W, grads_b


# ---------------------------------------------------------------------------
# 3. 训练：小批量梯度下降 + Adam
# ---------------------------------------------------------------------------


def train(model: MLP,
          X_train: np.ndarray,
          y_train: np.ndarray,
          X_test: np.ndarray,
          y_test: np.ndarray,
          epochs: int,
          batch_size: int,
          lr: float,
          seed: int,
          eval_every: int = 10,
          min_epochs: int = 800,
          patience: int = 250,
          rel_tol: float = 1e-3,
          lr_decay_epochs: float = 0.0,
          logger=None) -> dict:
    """训练 MLP，并记录损失/准确率历史。

    参数：
        model：待训练网络。
        X_train, y_train：训练集。
        X_test, y_test：独立测试集，用来观察泛化随训练的变化。
        epochs：训练轮数上限，每轮把整个训练集完整扫一遍。
        batch_size：小批量大小；等于训练集大小就是全批量梯度下降。
        lr：Adam 的学习率。
        seed：打乱训练数据用的种子。
        eval_every：每隔多少轮记录一次训练集/测试集准确率（最后一轮必定记录）。
        min_epochs：早停生效前至少要训练多少轮；设置它是为了不让浅层网络初始阶段那段
                    "损失几乎不动"的平台期被误判成已收敛。
        patience：连续多少轮没有出现超过 rel_tol 的相对改善就认为收敛并停止。
        rel_tol：相对改善阈值，当前损失低于历史最优的 (1 - rel_tol) 倍才算有改善。
        lr_decay_epochs：学习率按反比时间衰减的时间常数；0 表示不衰减。
        logger：可选的日志对象，需实现 log(str) 方法。
    返回：history 字典，含每轮损失与（按 eval_every 采样的）准确率曲线。
    为什么用 Adam 而不是朴素 SGD：本实验要在合理时间内让两种网络都接近各自的收敛点，
    Adam 对学习率不敏感、收敛快；四种配置统一使用同一套优化器与学习率，
    保证"网络深度"和"样本量"是唯一变量。
    为什么用"训练到收敛"而不是固定轮数：本实验要比较的是"用同样的方法、同样的数据量，
    深网和浅网各自能学到多好"。若固定轮数，结果取决于轮数选得够不够，反而掩盖了
    泛化能力本身的差别；训练到各自的收敛点后，测试准确率的差别才是纯泛化差别。
    """
    # 训练集大小，决定每轮有多少个批次
    n = X_train.shape[0]
    # 批次数量（向上取整，最后一批可能不满）
    n_batches = int(np.ceil(n / batch_size))
    # Adam 一阶/二阶矩的指数衰减率
    beta1, beta2 = 0.9, 0.999
    # Adam 分母的平滑项
    eps = 1e-8
    # 为每一层参数各准备一组一阶矩和二阶矩，初值全 0
    mW = [np.zeros_like(w) for w in model.W]
    vW = [np.zeros_like(w) for w in model.W]
    mb = [np.zeros_like(b) for b in model.b]
    vb = [np.zeros_like(b) for b in model.b]
    # 全局更新步数，用于 Adam 的偏差校正
    step = 0
    # 打乱顺序用独立生成器，避免影响数据采样本身的随机性
    rng = np.random.default_rng(seed)
    # 历史记录容器
    history = {"epoch": [], "loss": [], "train_acc": [], "test_acc": []}
    # 记录历史最优损失，作为判断"还在不在改善"的基准
    best_loss = float("inf")
    # 记录最后一次出现显著改善的轮数
    best_epoch = 0
    # 记录实际训练了多少轮
    epochs_run = 0
    # 记录是否触发了早停
    stopped_early = False
    # 开始计时，用于日志里给出每组实验的训练耗时
    t0 = time.perf_counter()
    for epoch in range(1, epochs + 1):
        # 本轮确实跑了，计数器加一
        epochs_run = epoch
        # 反比时间学习率衰减（lr_decay_epochs <= 0 时保持恒定学习率）：
        # 后期步长变小可以避免参数在大梯度噪声下持续乱走、把 tanh 单元推进饱和区
        if lr_decay_epochs > 0:
            # 当前轮使用的学习率
            current_lr = lr / (1.0 + (epoch - 1) / lr_decay_epochs)
        else:
            # 不衰减，直接用传入的固定学习率
            current_lr = lr
        # 每轮重新打乱训练样本顺序，让每个批次的组成都不同，减少梯度估计的偏差
        perm = rng.permutation(n)
        X_shuffled = X_train[perm]
        y_shuffled = y_train[perm]
        # 累加本轮各批次的损失，最后按样本数加权平均
        epoch_loss = 0.0
        # 统计本轮参与训练的样本总数，作为加权平均的分母
        seen = 0
        for start in range(0, n, batch_size):
            # 取出当前批次的输入
            X_batch = X_shuffled[start:start + batch_size]
            # 取出当前批次的标签
            y_batch = y_shuffled[start:start + batch_size]
            # 前向 + 反向，得到该批次的平均损失与梯度
            loss, grads_W, grads_b = model.loss_and_grads(X_batch, y_batch)
            # 步数加一，Adam 的偏差校正依赖它
            step += 1
            # 逐层做 Adam 参数更新
            for i in range(model.n_layers):
                # 更新一阶矩（梯度的指数滑动平均）
                mW[i] = beta1 * mW[i] + (1.0 - beta1) * grads_W[i]
                # 更新二阶矩（梯度平方的指数滑动平均）
                vW[i] = beta2 * vW[i] + (1.0 - beta2) * (grads_W[i] ** 2)
                # 偏差校正是为了修正前若干步矩估计偏向 0 的问题
                mW_hat = mW[i] / (1.0 - beta1 ** step)
                # 二阶矩同样做偏差校正
                vW_hat = vW[i] / (1.0 - beta2 ** step)
                # 参数沿"梯度均值 / 梯度标准差"方向下降，自适应缩放每个维度的步长
                model.W[i] -= current_lr * mW_hat / (np.sqrt(vW_hat) + eps)
                # 偏置的一阶矩更新
                mb[i] = beta1 * mb[i] + (1.0 - beta1) * grads_b[i]
                # 偏置的二阶矩更新
                vb[i] = beta2 * vb[i] + (1.0 - beta2) * (grads_b[i] ** 2)
                # 偏置一阶矩的偏差校正
                mb_hat = mb[i] / (1.0 - beta1 ** step)
                # 偏置二阶矩的偏差校正
                vb_hat = vb[i] / (1.0 - beta2 ** step)
                # 偏置同样沿自适应方向更新
                model.b[i] -= current_lr * mb_hat / (np.sqrt(vb_hat) + eps)
            # 把本批次的平均损失乘上该批次样本数，转成"总损失"再累加
            epoch_loss += loss * X_batch.shape[0]
            # 累加参与训练的样本数
            seen += X_batch.shape[0]
        # 本轮平均损失
        mean_loss = epoch_loss / seen
        # 若本轮损失相对历史最优有超过 rel_tol 的改善，则刷新最优记录和"最近改善轮数"
        if mean_loss < best_loss * (1.0 - rel_tol):
            # 更新历史最优损失
            best_loss = mean_loss
            # 记住这次显著改善发生在第几轮
            best_epoch = epoch
        # 早停判定：训练轮数已过保护期，且连续 patience 轮都没有显著改善，就认为已收敛
        if epoch >= min_epochs and (epoch - best_epoch) >= patience:
            # 标记触发了早停
            stopped_early = True
            # 打印早停信息，便于在日志里区分"收敛停止"和"跑满轮数停止"
            if logger is not None:
                logger.log(f"    [早停] 第 {epoch} 轮停止：最近一次显著改善在第 {best_epoch} 轮，"
                           f"已连续 {patience} 轮没有超过 {rel_tol:.0%} 的改善，"
                           f"当前损失 {mean_loss:.5f}")
            # 跳出训练循环
            break
        # 每 eval_every 轮（以及第一轮）做一次完整评估，记录曲线
        if epoch % eval_every == 0 or epoch == 1 or epoch == epochs:
            # 在整个训练集上算准确率，观察是否已经把训练集拟合好（欠拟合时会明显偏低）
            train_acc = model.accuracy(X_train, y_train)
            # 在独立测试集上算准确率，观察泛化能力
            test_acc = model.accuracy(X_test, y_test)
            # 记录到历史里，便于后面画曲线
            history["epoch"].append(epoch)
            # 损失也按记录点存，避免历史数组太长
            history["loss"].append(mean_loss)
            # 训练准确率
            history["train_acc"].append(train_acc)
            # 测试准确率
            history["test_acc"].append(test_acc)
            # 输出一行日志，包含轮数、损失、两个准确率和已耗时
            if logger is not None:
                logger.log(f"    epoch {epoch:4d} | loss {mean_loss:.5f} "
                           f"| train_acc {train_acc:.4f} | test_acc {test_acc:.4f} "
                           f"| {time.perf_counter() - t0:6.1f}s")
    # 循环结束后再做一次最终评估，保证历史里一定有"训练真正结束时"的那一点
    final_loss = model.loss_and_grads(X_train, y_train)[0]
    # 记录最终点的轮数（早停时可能不等于 eval_every 的整数倍）
    history["epoch"].append(epochs_run)
    # 记录最终损失
    history["loss"].append(final_loss)
    # 记录最终训练准确率
    history["train_acc"].append(model.accuracy(X_train, y_train))
    # 记录最终测试准确率
    history["test_acc"].append(model.accuracy(X_test, y_test))
    # 记录总训练耗时，写入返回结果
    history["train_seconds"] = time.perf_counter() - t0
    # 记录每轮的平均耗时，方便估算规模
    history["n_batches_per_epoch"] = n_batches
    # 记录实际训练轮数与是否早停，写进日志和 metrics.json
    history["epochs_run"] = epochs_run
    # 是否因为收敛而提前停止
    history["stopped_early"] = stopped_early
    # 返回历史
    return history


# ---------------------------------------------------------------------------
# 4. 日志：同时写标准输出与文件
# ---------------------------------------------------------------------------


class Logger:
    """把日志同时写到终端和文本文件，方便训练过程中随时查看、结束后作为交付日志。"""

    def __init__(self, path: str | None):
        # 记录日志文件路径，None 表示只打印到终端
        self.path = path
        # 以写模式打开（若已有内容则覆盖），编码固定 UTF-8 以免中文乱码
        self.fh = open(path, "w", encoding="utf-8") if path else None

    def log(self, message: str) -> None:
        """打印一行并同步写入文件。"""
        # 先输出到终端，保证后台运行时也能看到进度
        print(message, flush=True)
        # 再写入文件，每次写完立即 flush，进程被中断时日志也不会丢
        if self.fh is not None:
            self.fh.write(message + "\n")
            self.fh.flush()

    def close(self) -> None:
        """关闭文件句柄，释放资源。"""
        # 只有确实打开了文件才需要关闭
        if self.fh is not None:
            self.fh.close()


# ---------------------------------------------------------------------------
# 5. 绘图：决策区域 vs 标准棋盘格
# ---------------------------------------------------------------------------


def predict_grid(model: MLP, cell_size: float, resolution: int):
    """在 [-1,1]^2 的规则网格上逐点预测，得到"预测概率图"和"真实标签图"。

    参数：
        model：训练好的网络。
        cell_size：棋盘格格距，用于计算网格点上的真实标签。
        resolution：每个方向的网格点数，总点数为 resolution^2。
    返回：(xs, truth, probs)，xs 是一维坐标轴，truth/probs 都是 (resolution, resolution) 的二维图，
          注意 truth/probs 的第 0 维对应 x2，第 1 维对应 x1（与 meshgrid 的输出顺序一致）。
    """
    # 每个方向上的坐标刻度，从 -1 到 1 等间距
    xs = np.linspace(-DOMAIN, DOMAIN, resolution)
    # meshgrid 生成网格坐标矩阵，indexing="xy" 使 gx 随列变化、gy 随行变化
    gx, gy = np.meshgrid(xs, xs, indexing="xy")
    # 把网格拉平成一列点，形状 (resolution^2, 2)
    points = np.stack([gx.ravel(), gy.ravel()], axis=1)
    # 逐点预测正类概率，再还原成二维图
    probs = model.predict_proba(points).reshape(resolution, resolution)
    # 逐点计算真实标签并还原成二维图，作为"标准答案"对照
    truth = checkerboard_label(points, cell_size).reshape(resolution, resolution)
    return xs, truth, probs


def plot_single_result(xs: np.ndarray,
                       truth: np.ndarray,
                       probs: np.ndarray,
                       n_layers: int,
                       n_train: int,
                       train_acc: float,
                       test_acc: float,
                       hidden: int,
                       cell_size: float,
                       path: str) -> None:
    """画出"标准棋盘格 | 模型决策区域"双栏对比图并保存。

    参数：
        xs：一维坐标轴（两幅图共用）。
        truth：真实标签二维图。
        probs：模型预测概率二维图。
        n_layers：隐藏层层数（仅用于标题）。
        n_train：训练样本量（仅用于标题）。
        train_acc：训练集准确率（仅用于标题）。
        test_acc：测试集准确率（仅用于标题）。
        hidden：每层隐藏单元数（仅用于标题）。
        cell_size：棋盘格格距（仅用于标题）。
        path：图片保存路径。
    """
    # 创建 1 行 2 列的画布，左图是标准答案，右图是模型预测
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 5.4), dpi=130)
    # 两幅图使用同一个 extent，保证坐标轴刻度对应 [-1,1]
    extent = [-DOMAIN, DOMAIN, -DOMAIN, DOMAIN]
    # 左图：真实棋盘格，用离散红蓝上色，interpolation="nearest" 保证边界锐利
    axes[0].imshow(truth, origin="lower", extent=extent, cmap=CMAP_TRUTH,
                   vmin=0.0, vmax=1.0, interpolation="nearest")
    # 左图标题：说明这是真值函数，并标注格距与网格分辨率
    axes[0].set_title(f"标准棋盘格（真值函数，格距 {cell_size}，"
                      f"{truth.shape[0]}x{truth.shape[1]} 网格）", fontsize=11)
    # 右图：模型预测概率，用红蓝连续色标，越模糊说明模型越不确定、结构越崩坏
    im = axes[1].imshow(probs, origin="lower", extent=extent, cmap="bwr",
                        vmin=0.0, vmax=1.0, interpolation="bilinear")
    # 叠加 0.5 等概率线，即模型的实际决策边界
    axes[1].contour(xs, xs, probs, levels=[0.5], colors="black", linewidths=1.2)
    # 右图标题里给出该配置的关键信息和准确率
    axes[1].set_title(f"{n_layers} 个隐藏层, {hidden} 单元/层, "
                      f"训练样本 {n_train}\n训练准确率 {train_acc:.4f} | 测试准确率 {test_acc:.4f}",
                      fontsize=11)
    # 给两幅图统一加坐标轴标签，说明输入是二维坐标
    for ax in axes:
        ax.set_xlabel("x1")
        ax.set_ylabel("x2")
    # 右侧放一条颜色条，刻度即正类概率
    fig.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)
    # 整体大标题点明这是哪一组实验
    fig.suptitle(f"为什么深更好？棋盘格实验：{n_layers} 隐层 × {n_train} 训练样本", fontsize=13)
    # 自动调整布局，避免标题和标签重叠
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    # 保存图片到磁盘
    fig.savefig(path)
    # 关闭画布释放内存，否则连续画多张图会累积占用
    plt.close(fig)


def plot_comparison_block(records: list[dict], cell_size: float,
                          resolution: int, path: str) -> None:
    """把 4 组实验汇总到一张 2x2 大图里，方便一眼看出规律。

    参数：
        records：每组实验的结果字典（含模型、样本量、层数、准确率等）。
        cell_size：棋盘格格距。
        resolution：网格分辨率。
        path：汇总图的保存路径。
    """
    # 2 行 2 列的子图，行对应训练样本量、列对应隐藏层层数（标题里会写清楚）
    fig, axes = plt.subplots(2, 2, figsize=(12, 11), dpi=125)
    # 两幅图共用的坐标范围
    extent = [-DOMAIN, DOMAIN, -DOMAIN, DOMAIN]
    # 逐个子图填充
    for ax, rec in zip(axes.ravel(), records):
        # 取该组实验的模型和预测概率图（已有缓存则直接用，避免重复计算）
        probs = rec["probs"]
        # 一维坐标轴与网格分辨率对应
        xs = np.linspace(-DOMAIN, DOMAIN, resolution)
        # 用连续红蓝色标展示预测概率，模糊区域一目了然
        im = ax.imshow(probs, origin="lower", extent=extent, cmap="bwr",
                       vmin=0.0, vmax=1.0, interpolation="bilinear")
        # 叠加决策边界
        ax.contour(xs, xs, probs, levels=[0.5], colors="black", linewidths=1.0)
        # 子图标题给出配置与准确率
        ax.set_title(f"{rec['n_layers']} 隐层 × {rec['n_train']} 样本 | "
                     f"train {rec['train_acc']:.4f} / test {rec['test_acc']:.4f}", fontsize=11)
        # 标注坐标轴含义
        ax.set_xlabel("x1")
        # 纵轴是第二个输入维度
        ax.set_ylabel("x2")
        # 每幅图配颜色条
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    # 大标题说明这张图展示的是"深层网络在少数据下仍保住棋盘结构"
    fig.suptitle(f"4 组配置的决策区域（红=类1，蓝=类0，黑线为 0.5 决策边界，格距 {cell_size}）",
                 fontsize=13)
    # 调整布局
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    # 保存
    fig.savefig(path)
    # 释放画布
    plt.close(fig)


def plot_training_curves(records: list[dict], path: str) -> None:
    """画出 4 组实验的损失与测试准确率随训练轮数的变化曲线。

    参数：
        records：每组实验的结果字典，其中 history 里存有曲线数据。
        path：曲线图的保存路径。
    """
    # 1 行 2 列：左图是损失，右图是测试准确率
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.0), dpi=125)
    # 遍历 4 组结果逐条画线
    for rec in records:
        # 构造图例标签，标明层数与样本量
        label = f"{rec['n_layers']} 隐层 / {rec['n_train']} 样本"
        # 左图画训练损失（对训练集的平均交叉熵）
        axes[0].plot(rec["history"]["epoch"], rec["history"]["loss"], label=label, linewidth=1.6)
        # 右图画独立测试集上的准确率，这是本实验真正关心的指标
        axes[1].plot(rec["history"]["epoch"], rec["history"]["test_acc"], label=label, linewidth=1.6)
    # 左图纵轴：对数刻度更利于观察损失下降的后期细节
    axes[0].set_yscale("log")
    # 左图标题
    axes[0].set_title("训练损失（交叉熵）")
    # 右图标题
    axes[1].set_title("独立测试集准确率")
    # 两幅图统一设置横纵轴标签与网格
    for ax, ylabel in zip(axes, ["loss", "test accuracy"]):
        ax.set_xlabel("epoch")
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=9)
    # 大标题
    fig.suptitle("训练过程：损失与测试准确率", fontsize=13)
    # 调整布局
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    # 保存
    fig.savefig(path)
    # 释放画布
    plt.close(fig)


def cell_structure_report(probs: np.ndarray, cell_size: float, resolution: int) -> dict:
    """统计"每个棋盘格单元内部的平均正确率"，用来量化模型是否保住了棋盘结构。

    参数：
        probs：模型在 [-1,1]^2 网格上的预测概率，形状 (resolution, resolution)。
        cell_size：棋盘格格距。
        resolution：网格分辨率。
    返回：字典，含单元个数、单元平均正确率、最差单元的正确率，以及"判反的单元"个数。
    为什么需要这个指标：整体准确率会被"大色块"主导，看不出边界是否崩坏；
    按单元统计能直接回答"每个菱形格子内部是不是被一致地预测成同一个颜色"，
    这正是 PPT 里"深网在少数据下仍保留行列结构"的量化说法。
    """
    # 生成与 probs 对应的网格坐标（与 predict_grid 完全一致的取法）
    xs = np.linspace(-DOMAIN, DOMAIN, resolution)
    # 网格坐标矩阵
    gx, gy = np.meshgrid(xs, xs, indexing="xy")
    # 每个网格点的旋转坐标 u = (x1+x2)/cell_size
    u = (gx + gy) / cell_size
    # 每个网格点的旋转坐标 v = (x1-x2)/cell_size
    v = (gx - gy) / cell_size
    # 网格点所属的格子下标
    iu = np.floor(u).astype(np.int64)
    # 另一个方向的格子下标
    iv = np.floor(v).astype(np.int64)
    # 该点所属格子的真实标签（i+j 的奇偶）
    cell_label = ((iu + iv) % 2).astype(np.float64)
    # 模型对该点的预测标签
    pred = (probs >= 0.5).astype(np.float64)
    # 把 (iu, iv) 压成一个整数编号，方便用 np.bincount 做分组统计
    key = (iu - iu.min()) * (iv.max() - iv.min() + 1) + (iv - iv.min())
    # 展平后按单元分组
    key_flat = key.ravel()
    # 每个网格点是否预测正确
    correct = (pred == cell_label).ravel().astype(np.float64)
    # 每个单元内的网格点个数
    counts = np.bincount(key_flat)
    # 每个单元内预测正确的点数
    correct_sum = np.bincount(key_flat, weights=correct)
    # 只保留在网格里有足够采样点的单元，避免边界上几个点的单元干扰统计
    enough = counts >= 25
    # 每个单元的正确率
    cell_acc = correct_sum[enough] / counts[enough]
    # 每个单元的真实标签：用单元内标签的多数投票（格子内部标签本来是一致的）
    label_sum = np.bincount(key_flat, weights=cell_label.ravel())
    # 多数投票结果即该单元的真值标签
    cell_true_label = (label_sum[enough] / counts[enough] >= 0.5).astype(np.float64)
    # 判断每个单元是否被整体预测反了（正确率 < 0.5 视为判反）
    flipped = cell_acc < 0.5
    # 返回统计结果
    return {
        "n_cells": int(enough.sum()),
        "cell_mean_acc": float(cell_acc.mean()),
        "cell_min_acc": float(cell_acc.min()),
        "flipped_cells": int(flipped.sum()),
        "true_positive_cell_ratio": float(cell_true_label.mean()),
    }


def save_ascii_decision(probs: np.ndarray, path: str, n_layers: int, n_train: int,
                        train_acc: float, test_acc: float, n: int = 41) -> None:
    """把模型预测的决策区域渲染成 ASCII 文本保存，作为"结构有没有崩"的文字证据。

    参数：
        probs：形状 (resolution, resolution) 的预测概率图。
        path：文本文件保存路径。
        n_layers：隐藏层层数，写进表头。
        n_train：训练样本量，写进表头。
        train_acc：训练准确率，写进表头。
        test_acc：测试准确率，写进表头。
        n：ASCII 图每个方向取多少个点（从概率图上等距抽样）。
    为什么需要它：图片不便逐像素核对，ASCII 图可以直接和 checkerboard_ascii.txt 对照，
    一眼看出哪些区域被填成了错误的颜色、边界是否还保持行列结构。
    """
    # 从概率图上等距抽样 n 个下标，得到 n x n 的子采样
    idx = np.linspace(0, probs.shape[0] - 1, n).round().astype(np.int64)
    # 按行、列同时抽样，得到抽稀后的概率图
    small = probs[np.ix_(idx, idx)]
    # 打开文件写入
    with open(path, "w", encoding="utf-8") as fh:
        # 写表头，交代这一组实验的配置和准确率
        fh.write(f"模型预测的决策区域 ASCII 图：{n_layers} 隐层 x {n_train} 样本\n")
        # 写准确率
        fh.write(f"训练准确率 {train_acc:.4f} | 测试准确率 {test_acc:.4f}\n")
        # 说明符号含义：用 - 表示预测概率居中的"模糊"区域
        fh.write("符号: '##'=预测红(类1)  '..'=预测蓝(类0)  '??'=|p-0.5|<0.2 的模糊区\n")
        # 行方向说明
        fh.write("行方向为 x2 从上到下由 1 递减到 -1，列方向为 x1 从左到右递增\n")
        # 从 x2 最大的一行开始逐行画
        for row in small[::-1]:
            # 按概率取值决定该点用哪个符号，模糊区单独标出
            fh.write("".join("##" if v >= 0.7 else (".." if v <= 0.3 else "??") for v in row) + "\n")


def save_ascii_preview(cell_size: float, path: str, n: int = 41) -> None:
    """把棋盘格真值渲染成 ASCII 文本并保存，作为"数据生成是否正确"的文字证据。

    参数：
        cell_size：棋盘格格距。
        path：文本文件保存路径。
        n：每个方向的采样点数。
    """
    # 规则网格的坐标
    xs = np.linspace(-DOMAIN, DOMAIN, n)
    # 生成网格点
    gx, gy = np.meshgrid(xs, xs, indexing="xy")
    # 拉平成点列
    points = np.stack([gx.ravel(), gy.ravel()], axis=1)
    # 打标签并还原成方阵，注意转置后行号才与 x2 递增方向一致
    grid = checkerboard_label(points, cell_size).reshape(n, n)
    # 打开文本文件准备写入
    with open(path, "w", encoding="utf-8") as fh:
        # 写一行说明，交代这幅 ASCII 图怎么读
        fh.write(f"棋盘格真值 ASCII 预览（#=类1 红, .=类0 蓝），格距 cell_size={cell_size}\n")
        # 说明横纵坐标方向
        fh.write(f"行方向为 x2 从上到下由 {DOMAIN} 递减到 {-DOMAIN}，列方向为 x1 从左到右递增\n")
        # 从 x2 最大的一行开始往下画，符合图像直觉
        for row in grid[::-1]:
            # 每个点用两个字符宽度，视觉上接近正方形
            fh.write("".join("##" if v > 0.5 else ".." for v in row) + "\n")


# ---------------------------------------------------------------------------
# 6. 主流程
# ---------------------------------------------------------------------------


def parse_args(argv=None) -> argparse.Namespace:
    """解析命令行参数。"""
    # 创建解析器，描述本脚本的用途
    parser = argparse.ArgumentParser(
        description="李宏毅 Why Deep 棋盘格实验：1 隐层 vs 3 隐层，在不同训练样本量下的对比")
    # 训练样本量列表，默认 10 万与 2 万两组
    parser.add_argument("--samples", type=int, nargs="+", default=[100000, 20000],
                        help="训练样本量列表，默认 100000 20000")
    # 隐藏层层数列表，默认 1 与 3
    parser.add_argument("--layers", type=int, nargs="+", default=[1, 3],
                        help="隐藏层层数列表，默认 1 3")
    # 每个隐藏层的神经元个数，默认 64
    parser.add_argument("--hidden", type=int, default=DEFAULT_HIDDEN,
                        help="每个隐藏层的神经元个数")
    # 棋盘格格距，越小格子越密
    parser.add_argument("--cell-size", type=float, default=DEFAULT_CELL_SIZE,
                        help="棋盘格旋转坐标下的格距，默认 0.5（水平方向约 4 个菱形）")
    # 训练轮数上限（达到收敛判据会提前停止，见 train 函数的 min_epochs / patience）
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS,
                        help="训练轮数上限，默认 4000；收敛后会自动提前停止")
    # 批次大小；0 表示全批量梯度下降（每次更新用上全部训练样本）
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH,
                        help="小批量大小；传 0 表示全批量梯度下降（默认 0）")
    # 学习率
    parser.add_argument("--lr", type=float, default=DEFAULT_LR, help="Adam 学习率")
    # 学习率反比时间衰减的时间常数，0 表示不衰减
    parser.add_argument("--lr-decay-epochs", type=float, default=0.0,
                        help="学习率反比时间衰减常数，0 表示恒定学习率")
    # 第一层权重的初始化放大倍数（等价于放大输入尺度）
    parser.add_argument("--init-scale", type=float, default=DEFAULT_INIT_SCALE,
                        help="第一层权重初始化放大倍数，等价于放大输入尺度，默认 4.0")
    # 独立测试集大小
    parser.add_argument("--test-size", type=int, default=DEFAULT_TEST_SIZE,
                        help="独立测试集样本数")
    # 全局随机种子，控制数据生成和权重初始化
    parser.add_argument("--seed", type=int, default=0, help="随机种子")
    # 隐藏层激活函数
    parser.add_argument("--activation", choices=["tanh", "sigmoid"], default="tanh",
                        help="隐藏层激活函数")
    # 决策区域图的分辨率
    parser.add_argument("--resolution", type=int, default=DEFAULT_RESOLUTION,
                        help="决策区域网格分辨率")
    # 输出目录
    parser.add_argument("--outdir", type=str, default="results", help="结果输出目录")
    # 返回解析结果
    return parser.parse_args(argv)


def main(argv=None) -> int:
    """主流程：生成数据 -> 依次训练 4 组模型 -> 评估 -> 画图 -> 保存日志与汇总。"""
    # Windows 控制台默认用 GBK 编码，中文日志会乱码；显式改成 UTF-8 并允许替换无法编码的字符
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    # 解析参数
    args = parse_args(argv)
    # 输出目录若不存在则创建
    os.makedirs(args.outdir, exist_ok=True)
    # 构造日志器，日志文件放在输出目录下
    logger = Logger(os.path.join(args.outdir, "train_log.txt"))
    # 记录整段运行的开始时刻
    t_start = time.perf_counter()
    # 设置中文字体，返回值仅用于日志提示
    font_name = setup_chinese_font()
    # 打印实验头信息
    logger.log("=" * 78)
    logger.log("李宏毅《Why Deep》棋盘格实验复现（纯 NumPy 手写 MLP）")
    logger.log("=" * 78)
    # 打印关键超参，便于事后核对
    logger.log(f"Python {sys.version.split()[0]} | numpy {np.__version__} | matplotlib {matplotlib.__version__}")
    # 说明中文字体是否可用
    logger.log(f"绘图字体: {font_name}")
    # 棋盘格参数
    logger.log(f"棋盘格格距 cell_size = {args.cell_size}（水平方向约 {2.0 / args.cell_size:.0f} 个菱形）")
    # 网络与训练参数
    logger.log(f"网络: 2 -> {args.hidden} x {{1,3}} -> 1, 激活 {args.activation}, 输出 sigmoid")
    # 优化器参数；批次为 0 时按全批量处理，日志里要写清楚
    logger.log(f"优化: Adam, lr={args.lr}, "
               f"batch={'全批量' if args.batch_size <= 0 else args.batch_size}, epochs={args.epochs}")
    # 学习率衰减常数也要写进日志，否则只看日志无法判断用的是哪种学习率协议
    logger.log(f"学习率衰减: "
               f"{'恒定（不衰减）' if args.lr_decay_epochs <= 0 else f'反比时间衰减，时间常数 {args.lr_decay_epochs}'}")
    # 第一层初始化缩放倍数
    logger.log(f"第一层权重初始化放大倍数 = {args.init_scale}（等价于把输入尺度放大同样倍数）")
    # 测试集规模
    logger.log(f"独立测试集: {args.test_size} 个新样本（与训练集同分布、不同随机种子）")
    # 打印四组配置
    logger.log(f"实验配置: layers={args.layers} x samples={args.samples}")

    # 生成 ASCII 预览，作为棋盘格生成的文字证据，同时也方便在没有看图工具时检查数据
    ascii_path = os.path.join(args.outdir, "checkerboard_ascii.txt")
    save_ascii_preview(args.cell_size, ascii_path)
    # 记录路径到日志
    logger.log(f"棋盘格 ASCII 预览已保存: {ascii_path}")

    # 生成一个 10 万点的"训练样本池"，各配置都从这个池子的前 n 个样本里取，
    # 这样 2 万样本的实验是 10 万样本实验的真子集，两组数据的分布完全一致，对比更干净
    max_n = max(args.samples)
    logger.log(f"\n[数据] 正在生成 {max_n} 个训练点池 + {args.test_size} 个独立测试点 ...")
    # 固定种子生成训练池，注意池子的种子与测试集不同，保证测试集独立
    pool_X, pool_y = build_dataset(max_n, args.cell_size, seed=args.seed)
    # 用另一个种子生成独立测试集
    test_X, test_y = build_dataset(args.test_size, args.cell_size, seed=args.seed + 99991)
    # 输出训练池的类别比例，检查两类是否大致均衡
    logger.log(f"       训练池正类比例 = {pool_y.mean():.4f}（理论上接近 0.5）")
    # 输出测试集的类别比例
    logger.log(f"       测试集正类比例 = {test_y.mean():.4f}")

    # 用来收集每组实验的结果，供汇总结论和画汇总图使用
    records: list[dict] = []
    # 记录每组实验的耗时
    timings: dict[str, float] = {}
    # 逐组配置训练；外层遍历层数、内层遍历样本量，保证输出顺序稳定
    for n_layers in args.layers:
        for n_train in args.samples:
            # 该配置的唯一标识，用于图片命名
            tag = f"{n_layers}layer_{n_train}"
            # 训练集取池子的前 n_train 个样本（复制一份，避免后续操作污染池子）
            X_train = pool_X[:n_train].copy()
            # 标签同样切片
            y_train = pool_y[:n_train].copy()
            # 打印本组实验的配置
            logger.log("\n" + "-" * 78)
            logger.log(f"[实验] {n_layers} 个隐藏层 x {n_train} 个训练样本  (tag={tag})")
            # 隐藏层宽度列表：n_layers 层，每层 args.hidden 个单元
            hidden_sizes = [args.hidden] * n_layers
            # 为该配置构造网络；初始化种子随配置变化，避免偶然性但保持可复现
            model = MLP(hidden_sizes, seed=args.seed + 1000 * n_layers + n_train % 1000,
                        activation=args.activation, first_layer_scale=args.init_scale)
            # 打印参数量，便于对照"深网参数更多但泛化更好"这个反直觉现象
            n_params = sum(w.size for w in model.W) + sum(b.size for b in model.b)
            # 输出参数量
            logger.log(f"       可训练参数个数 = {n_params}")
            # 批次大小：传 0 或比样本数还大时，都退化成"每次更新用全部训练样本"
            effective_batch = n_train if args.batch_size <= 0 else args.batch_size
            # 每次更新实际使用的样本数写进日志，避免误解
            logger.log(f"       每次参数更新使用样本数 = {effective_batch} "
                       f"（共 {args.epochs} 次更新）")
            # 本组开始计时
            t0 = time.perf_counter()
            # 训练模型
            history = train(model, X_train, y_train, test_X, test_y,
                            epochs=args.epochs, batch_size=effective_batch, lr=args.lr,
                            seed=args.seed + 7, eval_every=max(1, args.epochs // 20),
                            lr_decay_epochs=args.lr_decay_epochs,
                            logger=logger)
            # 训练结束后在整个训练集上评估，用于判断是否欠拟合
            train_acc = model.accuracy(X_train, y_train)
            # 在独立测试集上评估，这是本实验的关键指标
            test_acc = model.accuracy(test_X, test_y)
            # 记录本组总耗时
            timings[tag] = time.perf_counter() - t0
            # 打印本组结果
            logger.log(f"    => 训练准确率 {train_acc:.4f} | 测试准确率 {test_acc:.4f} "
                       f"| 耗时 {timings[tag]:.1f}s")
            # 在网格上逐点预测，得到概率图与真值图
            xs, truth, probs = predict_grid(model, args.cell_size, args.resolution)
            # 统计每个棋盘格单元内部的正确率，量化"结构有没有被学出来"
            structure = cell_structure_report(probs, args.cell_size, args.resolution)
            # 把结构指标写进日志
            logger.log(f"       结构指标: 单元平均正确率 {structure['cell_mean_acc']:.4f}, "
                       f"最差单元 {structure['cell_min_acc']:.4f}, "
                       f"判反的单元 {structure['flipped_cells']}/{structure['n_cells']}")
            # 单组对比图的保存路径，命名规则见文档
            fig_path = os.path.join(args.outdir, f"{tag}.png")
            # 画图并保存
            plot_single_result(xs, truth, probs, n_layers, n_train, train_acc, test_acc,
                               args.hidden, args.cell_size, fig_path)
            # 记录路径
            logger.log(f"       决策区域对比图已保存: {fig_path}")
            # 同时保存该配置预测结果的 ASCII 图，便于在文本里核对结构是否崩坏
            ascii_dec = os.path.join(args.outdir, f"decision_ascii_{tag}.txt")
            save_ascii_decision(probs, ascii_dec, n_layers, n_train, train_acc, test_acc)
            # 记录 ASCII 图路径
            logger.log(f"       决策区域 ASCII 图已保存: {ascii_dec}")
            # 收集本组结果
            records.append({
                "tag": tag,
                "n_layers": n_layers,
                "n_train": n_train,
                "hidden": args.hidden,
                "n_params": int(n_params),
                "train_acc": float(train_acc),
                "test_acc": float(test_acc),
                "train_seconds": float(timings[tag]),
                "structure": structure,
                "history": history,
                "probs": probs,
                "figure": fig_path,
            })

    # 保存 4 组汇总的 2x2 大图
    summary_path = os.path.join(args.outdir, "all_configs_comparison.png")
    plot_comparison_block(records, args.cell_size, args.resolution, summary_path)
    # 保存训练曲线图
    curves_path = os.path.join(args.outdir, "training_curves.png")
    plot_training_curves(records, curves_path)

    # 打印最终准确率表格
    logger.log("\n" + "=" * 78)
    logger.log("汇总：4 组配置的准确率")
    logger.log("=" * 78)
    # 表头：层数、样本量、参数量、训练准确率、测试准确率、单元结构指标、耗时
    logger.log(f"{'隐层数':>6} | {'训练样本':>8} | {'参数量':>7} | {'训练准确率':>10} | "
               f"{'测试准确率':>10} | {'单元平均正确率':>13} | {'判反单元':>8} | {'耗时(s)':>8}")
    # 逐行输出
    for rec in records:
        logger.log(f"{rec['n_layers']:>6} | {rec['n_train']:>8} | {rec['n_params']:>7} | "
                   f"{rec['train_acc']:>10.4f} | {rec['test_acc']:>10.4f} | "
                   f"{rec['structure']['cell_mean_acc']:>13.4f} | "
                   f"{rec['structure']['flipped_cells']:>3}/{rec['structure']['n_cells']:<4} | "
                   f"{rec['train_seconds']:>8.1f}")

    # 计算"同一样本量下深网比浅网高多少"这个关键差值，直接印证 PPT 的结论
    logger.log("\n关键对比（同一训练样本量下，3 隐层减 1 隐层的测试准确率差）：")
    # 按样本量分组比较
    for n_train in args.samples:
        # 找出该样本量下 1 隐层与 3 隐层的结果
        one = next((r for r in records if r["n_train"] == n_train and r["n_layers"] == 1), None)
        # 深网结果
        three = next((r for r in records if r["n_train"] == n_train and r["n_layers"] == 3), None)
        # 两者都存在才做比较
        if one is not None and three is not None:
            # 打印差值（正数表示深网更好）
            logger.log(f"    {n_train:>7} 样本: 1 隐层 {one['test_acc']:.4f} | "
                       f"3 隐层 {three['test_acc']:.4f} | 差值 {three['test_acc'] - one['test_acc']:+.4f}")

    # 把结构性结果（不含大数组 probs）写入 JSON，作为机器可读的结果记录
    metrics = {
        "config": {
            "cell_size": args.cell_size,
            "hidden": args.hidden,
            "activation": args.activation,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "lr_decay_epochs": args.lr_decay_epochs,
            "init_scale": args.init_scale,
            "test_size": args.test_size,
            "seed": args.seed,
            "optimizer": "Adam",
            "implementation": "numpy",
        },
        "results": [{k: v for k, v in rec.items() if k != "probs"} for rec in records],
        "total_seconds": time.perf_counter() - t_start,
    }
    # 写 JSON 文件
    metrics_path = os.path.join(args.outdir, "metrics.json")
    with open(metrics_path, "w", encoding="utf-8") as fh:
        json.dump(metrics, fh, ensure_ascii=False, indent=2)
    # 打印总耗时与产物清单
    logger.log(f"\n总耗时 {metrics['total_seconds']:.1f}s（含数据生成、训练、评估、画图）")
    logger.log(f"结果已保存到: {os.path.abspath(args.outdir)}")
    logger.log(f"  - {metrics_path}")
    logger.log(f"  - {os.path.join(args.outdir, 'train_log.txt')}")
    logger.log(f"  - {summary_path}")
    logger.log(f"  - {curves_path}")
    # 关闭日志文件
    logger.close()
    return 0


if __name__ == "__main__":
    # 用 sys.exit 传递返回码，方便在脚本里串起多个命令
    sys.exit(main())
