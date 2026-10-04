#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
模型与数据模块：李宏毅《Bias and Variance》那一章里唯一的"数据集 + 真函数 + 模型族"。

本文件只负责四件最基础的事，别的什么都不做：
    1. 真函数 f(x)：一条"先缓后陡"的 S 形曲线，课上叫它 f^（hat），现实中永远拿不到；
    2. 抽样：从 x∈[0,700] 上抽训练点 / 验证点 / 测试点，并给它们加上高斯噪声 ε ~ N(0, σ²)；
    3. 模型族：1~5 次多项式回归，用最小二乘的闭式解拟合（见 fit_polynomial 的说明）；
    4. 评估：均方误差 MSE。

为什么要有这个文件：把"真函数长什么样""噪声怎么加""模型怎么拟合"三件事集中在一个地方，
后面 analysis.py 里那些"平行宇宙""误差分解""模型选择"才不用重复解释数据是怎么来的，
也保证四个模块用的是同一套函数、同一套噪声、同一套模型族——这是能互相比较的前提。

用法：本文件不直接运行，由 main.py 导入。
"""

from __future__ import annotations
# 打开延迟注解求值，类型标注里可以直接写 np.ndarray 而不用担心求值顺序

import numpy as np
# 全套数值计算都用 NumPy 完成，不依赖任何深度学习框架

X_MIN = 0.0
# 真函数的定义域下界：x 从 0 开始
X_MAX = 700.0
# 真函数的定义域上界：x 到 700 结束
SIGMOID_MID = 350.0
# S 形曲线的拐点位置：真函数在这里上升最快，取定义域正中间
SIGMOID_SCALE = 80.0
# 拐点的"软硬程度"：这个数越小曲线越像阶跃，越大越接近一条斜线
SIGMOID_HEIGHT = 500.0
# S 形部分的高度：曲线从 0 附近升到 500 附近，也就是课程里"Pokemon CP 值"那一档量级
LINEAR_SLOPE = 0.3
# 叠加的线性项斜率，让曲线整体带一点持续上升的趋势，而不是两端完全压平
NOISE_SIGMA = 50.0
# 观测噪声的标准差 σ：y = f(x) + ε，ε ~ N(0, σ²)，σ² = 2500 就是误差分解里那条"压不下去的地板"
DESIGN_CENTER = 350.0
# 多项式特征的居中常数；把它减掉，是为了让 (x-350)/350 落在 [-1,1]，避免 x 的高次幂数值过大
DESIGN_SCALE = 350.0
# 多项式特征的缩放系数；居中和缩放合起来就是"按固定常数标准化"
DEGREES = (1, 2, 3, 4, 5)
# 模型族：1、2、3、4、5 次多项式。1/2/3/5 是任务指定的四个代表，2 和 4 是补齐曲线用的
TARGET_DEGREES = (1, 2, 3, 5)
# 打靶图专用的四个次数（2×2 正好四格），与任务给的模型族一致
ANCHOR_A = 200.0
# 打靶图的第一个固定输入位置 x_a：处在 S 曲线"刚抬头"的位置
ANCHOR_B = 500.0
# 打靶图的第二个固定输入位置 x_b：处在 S 曲线"快压平"的位置


def true_function(x: np.ndarray | float) -> np.ndarray:
    """真函数 f(x) = 500 · sigmoid((x − 350) / 80) + 0.3 · x。

    参数：
        x：标量或任意形状的数组，单位与定义域一致（0~700）。
    返回：与 x 同形状的数组，表示没有噪声时的真实取值。

    公式拆开看：
        500 · sigmoid((x − 350) / 80) 是 S 形部分——x 比 350 小很多时趋于 0，
        x 比 350 大很多时趋于 500，中间 350 附近上升最快；
        0.3 · x 是线性部分——给整条曲线叠一个匀速上升的趋势。
    代入两个锚点验算：
        x = 200：sigmoid((200−350)/80) = sigmoid(−1.875) ≈ 0.13296，
                 500 × 0.13296 ≈ 66.48，加上 0.3 × 200 = 60，得 f(200) ≈ 126.48；
        x = 500：sigmoid((500−350)/80) = sigmoid(1.875) ≈ 0.86704，
                 500 × 0.86704 ≈ 433.52，加上 0.3 × 500 = 150，得 f(500) ≈ 583.52。
    为什么这条曲线适合当教材例子：它整体单调、先缓后陡，但"缓—陡—缓"的转折
    要 3 次左右的多项式才跟得上，于是 1 次欠拟合、5 次开始吃方差，误差曲线的 U 形很明显。

    补充说明：这里用 sigmoid(z) = 1 / (1 + exp(−z))。写成 exp 形式而不是 1/(1+1/exp(z))，
    是为了 x 很负的时候不会因为 exp 溢出而算出 inf。
    """
    x_arr = np.asarray(x, dtype=float)
    # 先统一转成浮点数组：传进来的可能是列表、可能是标量，转完就都能按数组算
    sigmoid_part = 1.0 / (1.0 + np.exp(-(x_arr - SIGMOID_MID) / SIGMOID_SCALE))
    # S 形部分：先把 x 平移到拐点、再按 scale 缩放，最后过一个标准 sigmoid
    return SIGMOID_HEIGHT * sigmoid_part + LINEAR_SLOPE * x_arr
    # S 形部分拉高到 500，再叠上斜率为 0.3 的线性项，就是真函数


def sample_x(rng: np.random.Generator, count: int) -> np.ndarray:
    """在 [0, 700] 上抽 count 个输入点，用"等宽分层 + 层内均匀"的方式。

    参数：
        rng：NumPy 的随机数生成器（由调用方传入，方便复现）。
        count：要抽的点数。
    返回：形状 (count,) 的浮点数组，值域 [0, 700]，并已按从小到大排序。

    为什么不用简单的 rng.uniform(0, 700, count)：
        10 个点完全自由地撒在 [0,700] 上，偶尔会 8 个挤在中间、两边各一个，
        这时 5 次多项式在两端就是"几乎无约束的外推"，预测值能炸到 1e5 以上。
        这种重尾抽样会让"方差"的估计被一两个极端宇宙完全带偏，实验不可复现。
        改成把 [0,700] 切成 count 个等宽小区间、每层抽一个点之后，
        训练点一定铺满整个定义域，方差估计稳定，而层内的位置仍然随机，
        "每个宇宙抽到不同数据集"这件事没有被破坏。
    排序是为了让后面画图（把训练点标出来时）好看，对拟合本身没有影响。
    """
    edges = np.linspace(X_MIN, X_MAX, count + 1)
    # 把 [0,700] 切成 count 个等宽小区间，edges 是 count+1 个分界点
    samples = rng.uniform(edges[:-1], edges[1:])
    # 每个小区间里均匀抽一个点：lower = 各段左端，upper = 各段右端，向量化一次抽完
    return np.sort(samples)
    # 排好序再返回，画图和调试时更直观


def sample_dataset(rng: np.random.Generator, count: int) -> tuple[np.ndarray, np.ndarray]:
    """抽一批带噪声的观测数据：(x, y)，其中 y = f(x) + ε，ε ~ N(0, σ²)。

    参数：
        rng：随机数生成器。
        count：样本个数。
    返回：(x, y)，两个形状都是 (count,) 的浮点数组。

    这是"一次实验的数据集"——也就是课程里说的一个宇宙。同一个宇宙可以拿去
    训练 1 次、2 次、3 次……多项式，这正是要做"同一批数据、不同复杂度"对照的前提。
    """
    x = sample_x(rng, count)
    # 先抽输入位置
    noise = rng.normal(0.0, NOISE_SIGMA, count)
    # 再抽观测噪声：均值 0、标准差 50 的正态噪声
    y = true_function(x) + noise
    # 真值加噪声就是观测到的 y，这一行是整个实验"数据从哪来"的唯一出处
    return x, y
    # 返回一批数据


def design_matrix(x: np.ndarray | float, degree: int) -> np.ndarray:
    """把输入 x 展开成多项式特征矩阵 [1, z, z², …, z^degree]，其中 z = (x − 350) / 350。

    参数：
        x：标量或一维数组，原始的输入位置。
        degree：多项式次数 d。
    返回：形状 (n, d+1) 的特征矩阵，第 0 列全是 1（对应截距项）。

    为什么要先做 z = (x − 350) / 350：
        x 最大 700，直接算 x⁵ 会到 1.7×10¹⁴，和 1 放在同一个矩阵里，
        矩阵的条件数会非常大，最小二乘的数值误差会盖过真实差别。
        把 x 线性映射到 [−1, 1] 只改变基的表示、不改变能拟合的函数集合
        （线性变换不改变张成的函数空间），但条件数从 10¹⁴ 量级降到 10³ 量级以内。
    """
    x_arr = np.asarray(x, dtype=float).reshape(-1)
    # 统一转成一维浮点数组：reshape(-1) 让标量也变成长度 1 的数组，后面矩阵乘法才合法
    z = (x_arr - DESIGN_CENTER) / DESIGN_SCALE
    # 居中 + 缩放到 [-1, 1]
    return np.vander(z, degree + 1, increasing=True)
    # NumPy 自带 Vandermonde 矩阵生成：increasing=True 表示列是 1, z, z², …, z^degree


def fit_polynomial(x: np.ndarray, y: np.ndarray, degree: int) -> np.ndarray:
    """用最小二乘的闭式解拟合多项式回归，返回系数向量。

    参数：
        x：形状 (n,) 的训练输入。
        y：形状 (n,) 的训练观测值（带噪声）。
        degree：多项式次数 d。
    返回：形状 (d+1,) 的系数，第 k 个元素是 z^k 的系数。

    为什么用闭式解而不是梯度下降：
        这一章要比较的是"模型复杂度带来的 bias / variance"，不是"优化器好不好用"。
        梯度下降会额外引入"没训够"的误差（欠优化的模型误差偏大且不稳定），
        那部分误差会和我们要观察的 bias / variance 混在一起，污染结论。
        而 n 只有 10~1500、d 只有 1~10，设计矩阵很小，
        np.linalg.lstsq 在毫秒级就能给出精确的最小二乘解，没有任何超参数要调。
        所以这里坚持闭式解：误差全部来自统计（有限数据 + 噪声），不掺优化噪声。
    """
    a = design_matrix(x, degree)
    # 构造特征矩阵 A，形状 (n, d+1)
    coef, *_ = np.linalg.lstsq(a, y, rcond=None)
    # 解 min_c ||A c − y||²。用 lstsq 而不是 (AᵀA)⁻¹Aᵀy：
    # 前者用 QR / SVD 分解，数值上稳得多；rcond=None 表示用默认的奇异值截断阈值
    return coef
    # 返回最小二乘解


def predict_polynomial(x: np.ndarray | float, coef: np.ndarray, degree: int) -> np.ndarray:
    """用拟合好的系数在给定输入上做预测。

    参数：
        x：标量或一维数组。
        coef：fit_polynomial 返回的系数。
        degree：多项式次数（必须和拟合时一致，特征矩阵要同构）。
    返回：形状与 x 沿展平后一致的预测值数组。
    """
    return design_matrix(x, degree) @ coef
    # 特征矩阵乘系数就是预测值：一行一个样本，@ 完成 (n, d+1) × (d+1,) → (n,)


def mean_squared_error(prediction: np.ndarray, target: np.ndarray) -> float:
    """均方误差 MSE = (1/N) Σ (prediction_i − target_i)²。

    参数：
        prediction：模型预测，形状 (N,)。
        target：真实观测值，形状 (N,)。
    返回：标量 MSE（单位是 y 的平方，本实验里 y 的量级是几百，所以 MSE 是几万以内）。

    为什么整章都用 MSE（而不是 MAE 或交叉熵）：
        因为 MSE 有一个漂亮的分解恒等式
        E[(y − f*)²] = bias² + variance + σ²，
        常数项可以干净地拆成"自己造成的偏"与"自己造成的散"加上"谁也躲不掉的噪声"。
        MAE 没有这样的恒等式，用 MAE 讲这一章会讲不干净。
    """
    error = np.asarray(prediction, dtype=float) - np.asarray(target, dtype=float)
    # 先算逐点误差
    return float(np.mean(error ** 2))
    # 平方后取平均，返回 Python 浮点，方便往 txt 里写


def truth_on_grid(x_grid: np.ndarray) -> np.ndarray:
    """在给定网格上取真函数的值（无噪声），画黑线和算 bias² 都要用。

    参数：
        x_grid：形状 (N,) 的网格点。
    返回：形状 (N,) 的真值。
    """
    return true_function(x_grid)
    # 直接转发到真函数，留这个包装是为了让 analysis.py 读起来不用关心真函数的实现
