#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
分析模块：把"平行宇宙"跑出来，并算出课程里那几个量。

四个模块的核心计算都在这里：
    模块一/二/三（平行宇宙）：run_parallel_universes
        对每个"宇宙"（= 一批独立的训练数据）拟合同一批多项式，
        在固定测试网格上收集所有宇宙的预测曲线，然后算
            bias²(x)  = ( 平均预测曲线 f̄(x) − 真函数 f(x) )²
            variance  = 各个宇宙的预测围绕 f̄ 的散布
            总误差    = 各宇宙在一个独立带噪测试网格上的平均 MSE
        并由这些量验证恒等式 总误差 ≈ bias² + variance + σ²。

    模块四（模型选择）：repeated_selection_experiment / board_size_sweep / homogeneous_group_experiment
        三个选模型的办法放在同一批数据上比：
            留出法（大验证集）／3 折交叉验证（小验证集）／小公榜（50 点）
        以及"公榜越小时乐观偏差越大""候选越多越接近时越容易翻车"两组对照。

一个术语先统一：本文件里的"预测曲线"指的是 f*(x)——某个宇宙训出来的模型；
"平均曲线"指的是 f̄(x)——把所有宇宙的预测曲线逐点取平均。课程里就是这么写的。
"""

from __future__ import annotations
# 打开延迟注解求值

import numpy as np
# 数值计算

from model import (
    DEGREES,
    NOISE_SIGMA,
    X_MAX,
    X_MIN,
    design_matrix,
    fit_polynomial,
    mean_squared_error,
    predict_polynomial,
    sample_dataset,
    true_function,
    truth_on_grid,
)
# 数据、模型族、评估都来自 model.py，保证四个模块用同一套定义

GRID_POINTS = 200
# 误差分解用的固定测试网格点数：200 个点均匀铺在 [0,700] 上
POOL_SIZE = 2000
# 模块四的"大训练池"大小：2000 点，切出训练集后剩下的全部当验证集
PUBLIC_SIZE = 50
# public 榜（小测试集）的大小：50 点
PRIVATE_SIZE = 5000
# private 榜（独立大测试集）的大小：5000 点
DEFAULT_TRAIN_SIZE = 30
# 模块四主设定的训练集大小：30 点（和 PPT 里"10 个训练点"同一量级，属于小数据）
DEFAULT_REPEATS = 300
# 模块四重复次数：300 次完整流程，用来统计"选错"和"乐观偏差"的分布


def make_grid(grid_points: int = GRID_POINTS) -> np.ndarray:
    """构造误差分解用的固定测试网格 x 坐标。

    参数：
        grid_points：网格点数。
    返回：形状 (grid_points,) 的等距网格，从 0 到 700。

    为什么用等距网格而不是随机抽点：
        bias² 与 variance 的定义是"在某处的期望"，要做到"整条曲线上的平均"，
        最干净的做法就是在一个确定的等距网格上取平均——这样重复实验时
        "平均"的对象始终是同一批 x，数字之间的差别只来自数据随机性。
    """
    return np.linspace(X_MIN, X_MAX, grid_points)
    # 定义域的两个端点来自 model.py，避免这里的 0 和 700 与真函数脱钩


def run_parallel_universes(
    n_universes: int,
    n_train: int,
    degrees: tuple[int, ...] = DEGREES,
    grid_points: int = GRID_POINTS,
    seed: int = 0,
) -> dict:
    """跑 n_universes 个平行宇宙，返回每个次数下的 bias²、variance、总误差和预测曲线。

    参数：
        n_universes：宇宙个数 M（每个宇宙 = 一批独立的训练数据）。
        n_train：每个宇宙的训练样本数 n。
        degrees：要考察的多项式次数集合。
        grid_points：测试网格点数 G。
        seed：随机种子，决定整批实验可复现。
    返回：字典，包含
        x_grid                    形状 (G,)，测试网格
        truth                     形状 (G,)，真函数在网格上的取值 f(x)
        predictions[degree]       形状 (M, G)，每个宇宙的预测曲线 f*_i(x)
        f_bar[degree]             形状 (G,)，平均预测曲线 f̄(x)
        bias2[degree]             标量，网格平均的 bias²
        variance[degree]          标量，网格平均的 variance（无偏估计，除以 M−1）
        observed_total[degree]    标量，M 个宇宙在独立带噪测试网格上的平均 MSE
        anchor_a[degree]          形状 (M,)，各宇宙在 x_a=200 处的预测
        anchor_b[degree]          形状 (M,)，各宇宙在 x_b=500 处的预测
        anchor_bias_distance      字典：到靶心的偏差距离 ||f̄ − f^||
        anchor_variance_radius    字典：弹着点云相对云心的均方根半径
        anchor_mean_distance      字典：弹着点云相对云心的平均距离
        examples                  前 3 个宇宙的 (x, y)，画图时用来标出训练点

    关键实现细节（三个"为什么"）：
        1. 同一个宇宙只抽一批训练数据，然后拿它分别拟合 1、2、3、4、5 次多项式。
           课程里那张图就是"同一批数据、不同复杂度"，只有共享数据集，
           不同次数之间的对照才有意义（否则差异里混进了"数据集不同"这个因素）。
        2. 每个宇宙再单独抽一份测试噪声，算总误差时用。
           这份噪声与训练噪声独立，代表"新的、没见过的测试数据"。
           训练噪声让 f* 偏离 f，测试噪声让观测到的 y 再抖一次，
           两者合起来正好还原 E[(y − f*)²] = bias² + variance + σ²。
        3. variance 用除以 (M−1) 的样本方差（贝塞尔校正）。
           若除以 M，会系统性低估算真实方差约 1/M（M=200 时约 0.5%），
           恒等式的残差里就会多出一个已知的负数偏移。用 M−1 消掉它，
           剩下的小残差才是纯粹的抽样误差，才能用来验收实现是否正确。
    """
    rng = np.random.default_rng(seed)
    # 独立的随机数流：同一个 seed 跑出来永远一样
    x_grid = make_grid(grid_points)
    # 固定测试网格，200 个等距点
    truth = truth_on_grid(x_grid)
    # 真函数在网格上的取值，后面算 bias² 要用
    m = n_universes
    # 宇宙个数

    predictions = {degree: np.empty((m, grid_points)) for degree in degrees}
    # 每个次数准备一个 (M, G) 的矩阵，逐行填每个宇宙的预测曲线
    anchor_a = {degree: np.empty(m) for degree in degrees}
    # 每个次数在每个宇宙下的 f*_i(200)
    anchor_b = {degree: np.empty(m) for degree in degrees}
    # 每个次数在每个宇宙下的 f*_i(500)
    observed_total = {degree: np.empty(m) for degree in degrees}
    # 每个次数在每个宇宙下的测试 MSE
    leverage_grid = {degree: np.zeros(grid_points) for degree in degrees}
    # 每个次数的杠杆值曲线 h(x) 在 M 个宇宙上的累加，用来验证"方差 = σ²·h"
    leverage_train = {degree: 0.0 for degree in degrees}
    # 每个次数的 h 在训练点上的均值，再对宇宙累加
    design_sum = {degree: np.zeros(grid_points) for degree in degrees}
    # "只看设计、不看噪声"的期望拟合曲线 E[f*|D] 在宇宙上的累加（见下面的说明）
    design_sq = {degree: np.zeros(grid_points) for degree in degrees}
    # 同一条曲线的平方累加，用来一次算出跨宇宙的方差（省掉存 M×G 矩阵）
    examples = []
    # 顺手留几个宇宙的原始训练数据，画图时用来标出"这 10 个点长什么样"
    grid_features = {degree: design_matrix(x_grid, degree) for degree in degrees}
    # 网格上的特征矩阵与宇宙无关，先算一次，省掉 200 次重复计算

    for i in range(m):
        # 逐个宇宙推进
        x_train, y_train = sample_dataset(rng, n_train)
        # 这个宇宙的训练数据：y = f(x) + ε，ε ~ N(0, σ²)
        if i < 3:
            # 只留前三个宇宙的数据当样例，避免把 200 份数据全存在内存里
            examples.append((x_train, y_train))
            # 存进去备用
        test_noise = rng.normal(0.0, NOISE_SIGMA, grid_points)
        # 这个宇宙专属的测试噪声，和训练噪声相互独立
        y_test = truth + test_noise
        # 该宇宙看到的带噪测试目标值
        for degree in degrees:
            # 同一批训练数据依次拟合不同次数的多项式
            coef = fit_polynomial(x_train, y_train, degree)
            # 最小二乘闭式解
            curve = predict_polynomial(x_grid, coef, degree)
            # 在测试网格上的预测曲线 f*_i(x)
            predictions[degree][i] = curve
            # 存进矩阵
            anchor_a[degree][i] = predict_polynomial(200.0, coef, degree)[0]
            # 记下这个宇宙在 x_a = 200 处的预测值（打靶图的横坐标）
            anchor_b[degree][i] = predict_polynomial(500.0, coef, degree)[0]
            # 记下这个宇宙在 x_b = 500 处的预测值（打靶图的纵坐标）
            observed_total[degree][i] = mean_squared_error(curve, y_test)
            # 这个宇宙在这个次数下的"观察到"的测试误差
            train_features = design_matrix(x_train, degree)
            # 这个宇宙训练点上的特征矩阵 (n, d+1)
            inverse = np.linalg.pinv(train_features.T @ train_features)
            # (AᵀA)⁻¹，杠杆值和"无噪声期望拟合"都要用
            grid_features_d = grid_features[degree]
            # 网格特征矩阵
            leverage_grid[degree] += np.einsum("ij,jk,ik->i", grid_features_d, inverse, grid_features_d)
            # 逐点算 h(x) = a(x)ᵀ(AᵀA)⁻¹a(x) 并累加
            leverage_train[degree] += float(
                np.mean(np.einsum("ij,jk,ik->i", train_features, inverse, train_features))
            )
            # 训练点上的 h 均值（理论上恰好等于 (d+1)/n）并累加
            only = grid_features_d @ (inverse @ (train_features.T @ true_function(x_train)))
            # "如果这批数据没有噪声、只有这组设计点"时的拟合曲线：
            # E[f*|D] = a(x)ᵀ(AᵀA)⁻¹Aᵀ f(x_train)，也就是"换一批噪声也还是这条曲线"的那部分
            design_sum[degree] += only
            # 累加一阶矩
            design_sq[degree] += only ** 2
            # 累加二阶矩
    # 所有宇宙跑完

    result = {
        "x_grid": x_grid,
        "truth": truth,
        "predictions": predictions,
        "examples": examples,
        "n_universes": m,
        "n_train": n_train,
    }
    # 先把原始曲线装进结果字典，下面再逐项算统计量

    f_bar = {}
    # 每个次数的平均曲线 f̄(x) = (1/M) Σ_i f*_i(x)
    bias2 = {}
    # 每个次数的网格平均 bias² = (1/G) Σ_x (f̄(x) − f(x))²
    variance = {}
    # 每个次数的网格平均 variance = (1/M) Σ_i (1/G) Σ_x (f*_i(x) − f̄(x))²
    total = {}
    # 每个次数的观察总误差 = (1/M) Σ_i MSE_i
    for degree in degrees:
        # 逐次数统计
        curves = predictions[degree]
        # 形状 (M, G) 的预测矩阵
        mean_curve = curves.mean(axis=0)
        # 沿宇宙方向取平均，得到 f̄(x)
        f_bar[degree] = mean_curve
        # 存起来
        bias2[degree] = float(np.mean((mean_curve - truth) ** 2))
        # 先把 f̄ 与真值的差平方，再在网格上取平均
        spread = curves - mean_curve
        # 每个宇宙的曲线相对 f̄ 的偏离，形状 (M, G)
        variance[degree] = float(np.sum(spread ** 2) / ((m - 1) * grid_points))
        # 除以 (M−1) 与网格点数 G，得到无偏的网格平均方差
        total[degree] = float(np.mean(observed_total[degree]))
        # 观察到的总误差：各宇宙测试 MSE 的平均
    # 三类统计量算完

    anchor_bias_distance = {}
    # 打靶图上的"偏"：云心（f̄ 在两个锚点的值）到靶心（真值）的欧氏距离
    anchor_variance_radius = {}
    # 打靶图上的"散"：弹着点到云心的均方根距离，等于 sqrt(var_a + var_b)
    anchor_mean_distance = {}
    # 打靶图上的"散（另一种口径）"：弹着点到云心的平均距离，与任务描述里的"平均距离"一致
    for degree in degrees:
        # 逐次数算打靶统计量
        cloud = np.column_stack([anchor_a[degree], anchor_b[degree]])
        # 形状 (M, 2) 的弹着点矩阵，两列分别是 f*_i(200) 与 f*_i(500)
        center = cloud.mean(axis=0)
        # 云心 = 平均曲线在两个锚点上的取值
        target = np.array([float(true_function(200.0)), float(true_function(500.0))])
        # 靶心 = 真函数在两个锚点上的取值（注意：是真值，不是带噪观测）
        anchor_bias_distance[degree] = float(np.linalg.norm(center - target))
        # 云心到靶心的距离就是 bias 在这两个位置上的合成
        offsets = cloud - center
        # 每个弹着点相对云心的位移
        squared = np.sum(offsets ** 2, axis=1)
        # 每个弹着点到云心的距离平方
        anchor_variance_radius[degree] = float(np.sqrt(np.mean(squared)))
        # 均方根半径：它的平方正好等于 var(f*(200)) + var(f*(500))，所以是"二维方差"的直观翻版
        anchor_mean_distance[degree] = float(np.mean(np.sqrt(squared)))
        # 平均距离：任务描述里"点到中心的平均距离"指的就是这个口径
    # 打靶统计量算完

    result.update(
        {
            "f_bar": f_bar,
            "bias2": bias2,
            "variance": variance,
            "observed_total": total,
            "anchor_a": anchor_a,
            "anchor_b": anchor_b,
            "anchor_bias_distance": anchor_bias_distance,
            "anchor_variance_radius": anchor_variance_radius,
            "anchor_mean_distance": anchor_mean_distance,
            "leverage_grid": {degree: leverage_grid[degree] / m for degree in degrees},
            "leverage_train_mean": {degree: leverage_train[degree] / m for degree in degrees},
            "design_variance": {
                degree: float(
                    np.mean((design_sq[degree] - design_sum[degree] ** 2 / m) / (m - 1))
                )
                for degree in degrees
            },
            # 设计方差 = E[f*|D] 跨宇宙的方差（在每个网格点上算，再对网格取平均）：
            # 用 Σc² 与 (Σc)²/M 直接算，等价于除以 M−1 的样本方差
        }
    )
    # 把统计量并进结果字典；杠杆值与设计方差都在同一个宇宙循环里算出来，
    # 因此它们与 variance 是"同一批数据"的三个侧面，可以直接对账
    return result
    # 返回全部结果


def identity_rows(result: dict, degrees: tuple[int, ...] = DEGREES) -> list[dict]:
    """把恒等式 总误差 = bias² + variance + σ² 的三列数字和残差整理成表格行。

    参数：
        result：run_parallel_universes 的返回值。
        degrees：要输出的次数。
    返回：每行一个字典，字段为 degree / bias2 / variance / sigma2 / rhs / observed / residual。

    为什么要单独做这个验证：
        bias²、variance、总误差这三条曲线是分别用三段不同代码算出来的：
        bias² 只用了平均曲线，variance 只用了曲线之间的散布，总误差用的是
        "每个宇宙在独立带噪网格上的 MSE"。如果代码有实现错误（比如忘了给
        测试集加独立噪声、或者 variance 少乘了一个因子），这三条曲线各自看
        都正常，但恒等式会立刻对不上。所以这个残差是整套实验的"自检灯"。
    """
    rows = []
    # 收集结果
    for degree in degrees:
        # 逐次数
        bias_sq = result["bias2"][degree]
        # bias²
        var = result["variance"][degree]
        # variance
        sigma_sq = float(NOISE_SIGMA ** 2)
        # 噪声方差 σ² = 2500，这是恒等式里的常数项
        rhs = bias_sq + var + sigma_sq
        # 恒等式右边
        observed = result["observed_total"][degree]
        # 观察到的总误差（恒等式左边）
        rows.append(
            {
                "degree": degree,
                "bias2": bias_sq,
                "variance": var,
                "sigma2": sigma_sq,
                "rhs": rhs,
                "observed": observed,
                "residual": observed - rhs,
                "relative": (observed - rhs) / observed if observed != 0 else 0.0,
            }
        )
        # 残差 = 左边 − 右边；相对残差 = 残差 / 左边，用来验收（要求 <5%）
    return rows
    # 返回表格行


def leverage_variance_check(result: dict, degrees: tuple[int, ...] = DEGREES) -> list[dict]:
    """验证"方差到底从哪来"：实测 variance 应当由 σ²·h(x) 与"设计方差"两块解释。

    参数：
        result：run_parallel_universes 的返回值。杠杆值、设计方差都在那次模拟里算好了，
                所以这里的对账用的是同一批宇宙，不存在"换一批数据再比"的额外误差。
        degrees：要检查的次数。
    返回：每个次数一行，含实测方差、噪声项、设计项、两者之和与比值。

    公式：对固定的设计点 x₁…xₙ，最小二乘预测在 x 处的方差是
        var(f*(x)) = σ² · h(x)，  h(x) = a(x)ᵀ (AᵀA)⁻¹ a(x)
    其中 a(x) 是 x 的多项式特征向量，h(x) 叫"杠杆值"，衡量这个点被自己的训练数据拽得多厉害。
    两条可检验的性质：
        1. h 在 n 个训练点上的平均值恰好等于 (d+1)/n——因为帽子矩阵的迹 = 参数个数；
        2. h 在网格上取平均时，高次多项式会被定义域两端拉高（d=5 时两端是中间的 38 倍）。

    本实验的设计点也是随机的（每个宇宙抽不同的 x），所以实测方差里除了
    "噪声让 f* 抖动"这一块（= σ²·E_D[h(x)]），还多一块"设计点位置不同让拟合结果整体偏移"
    （= Var_D(E[f*|D])，代码里叫 design_variance）。两块加起来才等于实测方差，
    这也是本检查不追求"比值恰好等于 1"的原因——它本来就是两个来源的和。
    """
    rows = []
    # 每个次数一行
    n_train = result["n_train"]
    # 每个宇宙的训练点数
    for degree in degrees:
        # 逐次数
        h_grid = result["leverage_grid"][degree]
        # 网格上的杠杆值曲线 h(x)（已在每个宇宙上取过平均）
        noise_term = float(NOISE_SIGMA ** 2 * h_grid.mean())
        # 噪声项：σ² × h 的网格平均
        design_term = result["design_variance"][degree]
        # 设计项：光是"训练点抽在哪"造成的拟合曲线波动
        measured = float(result["variance"][degree])
        # 实测方差（二项之和）
        rows.append(
            {
                "degree": degree,
                "measured_variance": measured,
                "noise_term": noise_term,
                "design_term": design_term,
                "predicted_variance": noise_term + design_term,
                "ratio": measured / (noise_term + design_term),
                "noise_share": noise_term / (noise_term + design_term),
                "h_train_mean": result["leverage_train_mean"][degree],
                "h_train_theory": (degree + 1) / n_train,
                "h_grid_mean": float(h_grid.mean()),
                "h_grid_max": float(h_grid.max()),
                "h_grid_min": float(h_grid.min()),
            }
        )
        # 记录这一行的全部数字
    return rows
    # 返回所有次数


def _fit_all(x_train: np.ndarray, y_train: np.ndarray, degrees: tuple[int, ...]) -> dict:
    """在同一批数据上把每个候选次数的模型都训一遍。

    参数：
        x_train / y_train：训练数据。
        degrees：候选次数集合。
    返回：字典 {次数: 系数向量}。
    """
    return {degree: fit_polynomial(x_train, y_train, degree) for degree in degrees}
    # 逐次数调用最小二乘，返回系数


def _score_all(coefs: dict, degrees: tuple[int, ...], x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """把每个候选模型在给定数据集上的 MSE 算出来。

    参数：
        coefs：_fit_all 返回的系数字典。
        degrees：候选次数集合，顺序决定返回数组的顺序。
        x / y：要评估的数据集（可以是验证集、public 榜或 private 榜）。
    返回：形状 (len(degrees),) 的 MSE 数组。
    """
    return np.array([mean_squared_error(predict_polynomial(x, coefs[degree], degree), y) for degree in degrees])
    # 逐次数预测并算 MSE，保持与 degrees 相同的顺序，后面 argmin 得到的下标才能映射回次数


def _split_pool(rng: np.random.Generator, n_train: int, pool_size: int) -> tuple:
    """从一个"大训练池"里切出训练集，剩下的全部当验证集。

    参数：
        rng：随机数生成器。
        n_train：训练集大小。
        pool_size：池子大小。
    返回：(x_train, y_train, x_val, y_val)。

    为什么要先 rng.permutation 洗牌再切：
        sample_dataset 抽的是"分层采样"，池子里的点按 x 从小到大排好。
        直接取前 n_train 个，等于把训练集锁死在 [0, 700] 最左边的一小段上，
        5 次多项式就会变成对右侧的疯狂外推，误差能到 1e7 量级。
        洗牌之后，训练集是池子的一个随机子集，x 仍然铺满整个定义域。
    """
    x_pool, y_pool = sample_dataset(rng, pool_size)
    # 抽一个 2000 点的大池子
    order = rng.permutation(pool_size)
    # 打乱顺序，消除"分层采样已经把 x 排好序"这一点
    x_pool, y_pool = x_pool[order], y_pool[order]
    # 按打乱后的顺序重排
    return x_pool[:n_train], y_pool[:n_train], x_pool[n_train:], y_pool[n_train:]
    # 前 n_train 个当训练集，其余全部当验证集


def _cross_validation_error(
    rng: np.random.Generator,
    x_train: np.ndarray,
    y_train: np.ndarray,
    degrees: tuple[int, ...],
    folds: int = 3,
) -> np.ndarray:
    """在训练集内部做 K 折交叉验证，返回每个次数的平均验证误差。

    参数：
        rng：随机数生成器（用来决定折的划分）。
        x_train / y_train：训练集（本实验里是 30 个点）。
        degrees：候选次数集合。
        folds：折数，默认 3。
    返回：形状 (len(degrees),) 的交叉验证误差。

    做法：把 30 个训练点随机分成 3 折，每次拿 2 折（20 个点）训练、留 1 折（10 个点）验证，
    三轮下来每个点都当过一次验证点，把 3 个验证误差平均起来当作"泛化误差的估计"。

    要注意它和"大验证集"的差别：这里的每一次验证只有 10 个点，
    单次验证误差的标准误差极大（σ=50 时，10 个点的 MSE 估计误差约上千），
    所以交叉验证给出的排名比大验证集抖得多——这正是模块四要对比的东西之一。
    另外它的训练规模是 20 点而不是 30 点，估计出来的误差会略偏悲观。
    """
    count = x_train.shape[0]
    # 训练点个数
    order = rng.permutation(count)
    # 随机打乱，避免折的划分与 x 的大小顺序相关
    fold_indices = np.array_split(order, folds)
    # 均分成 folds 份，每份是一个下标数组
    scores = np.zeros(len(degrees))
    # 累加每个次数的验证误差
    for fold in range(folds):
        # 逐折
        val_index = fold_indices[fold]
        # 本折当验证集
        train_index = np.concatenate([fold_indices[k] for k in range(folds) if k != fold])
        # 其余折拼起来当训练集
        coefs = _fit_all(x_train[train_index], y_train[train_index], degrees)
        # 在这一折的训练子集上重新拟合所有候选
        scores += _score_all(coefs, degrees, x_train[val_index], y_train[val_index])
        # 累积这一折的验证误差
    return scores / folds
    # 平均到每一折，作为该次数的交叉验证误差


def repeated_selection_experiment(
    x_private: np.ndarray,
    y_private: np.ndarray,
    n_repeats: int = DEFAULT_REPEATS,
    n_train: int = DEFAULT_TRAIN_SIZE,
    pool_size: int = POOL_SIZE,
    public_size: int = PUBLIC_SIZE,
    degrees: tuple[int, ...] = DEGREES,
    seed: int = 0,
) -> dict:
    """反复跑"选模型"这件事，比较三种选法谁更诚实。

    参数：
        x_private / y_private：固定的 private 榜数据（5000 点，带噪声）。
        n_repeats：重复次数 R。每次重复重新切池子、重新抽 public 榜。
        n_train：训练集大小。
        pool_size：大训练池大小（池子 − 训练集 = 验证集）。
        public_size：public 榜大小。
        degrees：候选次数集合。
        seed：随机种子。
    返回：字典，含三种选法的选择分布、选中模型的 private 误差、报告分数与乐观偏差。

    三种选法：
        holdout  留出法：用一个很大的验证集（池子里剩下的 1970 个点）挑次数；
        cv3      3 折交叉验证：在 30 个训练点内部做 3 折，用平均验证误差挑次数；
        public   公榜法：拿一个 50 点的小测试集挑次数（这是"错"的做法）。

    为什么"公榜法"是错的：50 个点的 MSE 估计本身抖得很厉害，
    而你是从 5 个候选里挑"在自己这 50 个点上恰好最低"的那个，
    被挑中的模型必然包含了一次运气——它的真误差没有报告出来的那么低。
    这就是乐观偏差（optimistic bias），也是 Kaggle public 榜翻车的全部机制。
    """
    rng = np.random.default_rng(seed)
    # 随机数流
    rules = ("holdout", "cv3", "public")
    # 三种选法的名字
    choices = {name: np.empty(n_repeats, dtype=int) for name in rules}
    # 每次重复选中的次数下标（不是次数本身，是它在 degrees 里的位置）
    reported = {name: np.empty(n_repeats) for name in rules}
    # 每种选法"报告出去的分数"
    realized = {name: np.empty(n_repeats) for name in rules}
    # 每种选法选中的模型在 private 榜上的真实分数
    best_possible = np.empty(n_repeats)
    # 每次重复里"如果闭着眼睛挑到最好的那个"能拿到的 private 分数
    degrees_array = np.array(degrees)
    # 转成数组，方便用下标取值
    val_track = []
    # 每一轮各个次数的验证集误差，最后取平均用来画"正确流程看的曲线"
    cv_track = []
    # 每一轮各个次数的交叉验证误差
    public_track = []
    # 每一轮各个次数的公榜误差
    private_track = []
    # 每一轮各个次数的 private 榜误差（"真值"曲线）

    for repeat in range(n_repeats):
        # 逐次重复
        x_train, y_train, x_val, y_val = _split_pool(rng, n_train, pool_size)
        # 切出训练集与大验证集
        x_public, y_public = sample_dataset(rng, public_size)
        # 另抽一个 50 点的小公榜（和训练/验证数据不重叠地独立抽样）
        coefs = _fit_all(x_train, y_train, degrees)
        # 在同一批训练数据上把所有候选模型训好——三个选法共用同一批模型，才是公平对照
        val_scores = _score_all(coefs, degrees, x_val, y_val)
        # 大验证集上的误差
        cv_scores = _cross_validation_error(rng, x_train, y_train, degrees)
        # 交叉验证误差
        public_scores = _score_all(coefs, degrees, x_public, y_public)
        # 公榜上的误差
        private_scores = _score_all(coefs, degrees, x_private, y_private)
        # private 榜上的真实误差（这是"考试"，不是用来选的，只能事后看）
        for name, scores in (("holdout", val_scores), ("cv3", cv_scores), ("public", public_scores)):
            # 逐个选法
            pick = int(np.argmin(scores))
            # 挑误差最低的候选
            choices[name][repeat] = pick
            # 记下选中了谁
            reported[name][repeat] = float(scores[pick])
            # 记下"你会报出去的那个分数"
            realized[name][repeat] = float(private_scores[pick])
            # 记下这个模型在 private 榜上的真实分数
        best_possible[repeat] = float(private_scores.min())
        # 本次重复的理论最优 private 分数
        val_track.append(val_scores)
        # 累积这一轮各个次数的验证集误差
        cv_track.append(cv_scores)
        # 累积这一轮各个次数的交叉验证误差
        public_track.append(public_scores)
        # 累积这一轮各个次数的公榜误差
        private_track.append(private_scores)
        # 累积这一轮各个次数的 private 榜误差
    # 全部重复跑完

    summary = {}
    # 汇总每种选法
    for name in rules:
        # 逐选法
        pick = choices[name]
        # 每次选中的候选下标
        chosen_degrees = degrees_array[pick]
        # 换成次数
        counts = np.bincount(pick, minlength=len(degrees))
        # 选择次数的直方图
        summary[name] = {
            "choices": chosen_degrees,
            "counts": counts,
            "reported_all": reported[name],
            "realized_all": realized[name],
            "mean_private": float(realized[name].mean()),
            "mean_reported": float(reported[name].mean()),
            "optimism": float(realized[name].mean() - reported[name].mean()),
            "optimism_median": float(np.median(realized[name] - reported[name])),
            "optimism_se": float((realized[name] - reported[name]).std(ddof=1) / np.sqrt(n_repeats)),
            "private_std": float(realized[name].std()),
            "mean_regret": float(realized[name].mean() - best_possible.mean()),
        }
        # 记录：平均 private 分数、平均报告分数、乐观偏差（含中位数与标准误）、private 分数的波动、相对理论最优的后悔值
    summary["best_possible"] = float(best_possible.mean())
    # 理论最优（每次都挑到最好的候选）的平均 private 分数
    summary["n_repeats"] = n_repeats
    # 重复次数，写报告要用
    summary["private_std_of_repeat"] = float(best_possible.std())
    # "理论最优"自己也有波动，这是数据集的随机性造成的
    summary["degrees"] = tuple(degrees)
    # 候选次数，画图时要当横轴刻度
    summary["per_degree_val"] = np.mean(np.array(val_track), axis=0)
    # 各个次数的平均验证集误差（正确流程看的那条曲线）
    summary["per_degree_cv"] = np.mean(np.array(cv_track), axis=0)
    # 各个次数的平均交叉验证误差
    summary["per_degree_public"] = np.mean(np.array(public_track), axis=0)
    # 各个次数的平均公榜误差
    summary["per_degree_private"] = np.mean(np.array(private_track), axis=0)
    # 各个次数的平均 private 榜误差（"真值"曲线）
    return summary
    # 返回汇总


def board_size_sweep(
    x_private: np.ndarray,
    y_private: np.ndarray,
    sizes: tuple[int, ...] = (10, 20, 50, 100, 200, 500, 1000, 2000),
    draws: int = 400,
    n_train: int = DEFAULT_TRAIN_SIZE,
    degrees: tuple[int, ...] = DEGREES,
    seed: int = 0,
) -> dict:
    """扫"公榜大小"，看乐观偏差怎么随公榜变大而衰减。

    参数：
        x_private / y_private：private 榜。
        sizes：要试的公榜大小列表。
        draws：每个大小重复抽多少次公榜。
        n_train：训练集大小。
        degrees：候选次数集合。
        seed：随机种子。
    返回：字典 {公榜大小: 平均乐观偏差}，另附每个大小下选中模型的平均 private 分数。

    实验设计（为什么这里是"固定一批模型 + 反复抽公榜"）：
        先把一批候选模型训好并冻结（只抽一次训练数据），private 分数也就固定了；
        然后反复抽不同的小公榜，看"用公榜挑出来的那个模型，在 private 榜上比它的公榜分数差多少"。
        这样唯一的随机性就来自公榜本身，正好把"选榜"这一件事的影响单独拎出来。
    结论形状：乐观偏差大致按 1/sqrt(公榜点数) 衰减；公榜 10 点时能到几百，
        公榜 2000 点时几乎为 0（说明公榜误差本身是无偏估计，"乐观"完全来自"挑最小"这个动作）。
    """
    rng = np.random.default_rng(seed)
    # 随机数流
    x_train, y_train, _, _ = _split_pool(rng, n_train, POOL_SIZE)
    # 固定一批训练数据（公榜只负责"选"，不负责"训"）
    coefs = _fit_all(x_train, y_train, degrees)
    # 冻结这一批候选模型
    private_scores = _score_all(coefs, degrees, x_private, y_private)
    # 这批模型的 private 分数（固定不变）
    out = {"sizes": list(sizes), "optimism": {}, "selected_private": {}, "mean_reported": {}, "n_train": n_train}
    # 结果容器

    for size in sizes:
        # 逐公榜大小
        picked_private = np.empty(draws)
        # 每次抽样里被选中的模型，其 private 分数
        picked_reported = np.empty(draws)
        # 每次抽样里被选中的模型，其公榜分数（你会报出去的数字）
        for draw in range(draws):
            # 反复抽公榜
            x_board, y_board = sample_dataset(rng, size)
            # 这一轮的公榜
            board_scores = _score_all(coefs, degrees, x_board, y_board)
            # 候选在这张公榜上的分数
            pick = int(np.argmin(board_scores))
            # 挑公榜上最好的那个
            picked_private[draw] = private_scores[pick]
            # 记下它真实的 private 分数
            picked_reported[draw] = board_scores[pick]
            # 记下你在公榜上看到的分数
        out["optimism"][size] = float((picked_private - picked_reported).mean())
        # 乐观偏差 = 真榜分数 − 公榜分数，为正说明"报告虚高"
        out["selected_private"][size] = float(picked_private.mean())
        # 选中模型的平均 private 分数
        out["mean_reported"][size] = float(picked_reported.mean())
        # 平均报出去的分数
    return out
    # 返回扫描结果


def homogeneous_group_experiment(
    x_private: np.ndarray,
    y_private: np.ndarray,
    group_size: int = 30,
    board_size: int = 50,
    draws: int = 400,
    degree: int = 3,
    n_train: int = DEFAULT_TRAIN_SIZE,
    seed: int = 0,
) -> dict:
    """对照组：全是"同一个模型、不同训练样本"的候选群，看乐观偏差会变成多少。

    参数：
        x_private / y_private：private 榜。
        group_size：候选群里有多少个变体（都是 degree 次多项式，只是训练数据不同）。
        board_size：公榜大小。
        draws：抽多少次公榜。
        degree：候选群用的次数。
        n_train：每个变体的训练集大小。
        seed：随机种子。
    返回：字典，含候选群的真实水平、选中者的 private 分数、乐观偏差。

    为什么要有这一组：真实比赛里，选手交上去的往往是同一个网络、
    随机种子换一下、或者某个超参微调一下的"同质候选"——它们的真实水平几乎一样，
    但在公榜上的分数各有各的运气。候选之间的真实差距越小，
    "公榜最高的那个"就越是纯运气，乐观偏差也越大。
    这里用 30 个"同一批次、不同训练样本"的 3 次多项式把这件事量出来。
    """
    rng = np.random.default_rng(seed)
    # 随机数流
    variants = []
    # 每个变体的系数
    variant_private = np.empty(group_size)
    # 每个变体在 private 榜上的真实分数
    for index in range(group_size):
        # 每个变体用一批独立的训练数据
        x_train, y_train, _, _ = _split_pool(rng, n_train, POOL_SIZE)
        # 切出它自己的训练集
        coef = fit_polynomial(x_train, y_train, degree)
        # 训练（次数都一样，只有数据不同）
        variants.append(coef)
        # 存起来
        variant_private[index] = mean_squared_error(predict_polynomial(x_private, coef, degree), y_private)
        # 算它在 private 榜上的真实分数，作为"这个变体的真实水平"
    picked_private = np.empty(draws)
    # 每次公榜抽样的选中者 private 分数
    picked_reported = np.empty(draws)
    # 每次公榜抽样的公榜分数
    for draw in range(draws):
        # 反复抽公榜
        x_board, y_board = sample_dataset(rng, board_size)
        # 这一轮的公榜
        scores = np.array(
            [mean_squared_error(predict_polynomial(x_board, coef, degree), y_board) for coef in variants]
        )
        # 所有变体在这张公榜上的分数
        pick = int(np.argmin(scores))
        # 挑公榜上最好的
        picked_private[draw] = variant_private[pick]
        # 它的真实 private 分数
        picked_reported[draw] = float(scores[pick])
        # 公榜上看到的分数
    return {
        "group_size": group_size,
        "degree": degree,
        "board_size": board_size,
        "variants_private_mean": float(variant_private.mean()),
        "variants_private_std": float(variant_private.std()),
        "variants_private_min": float(variant_private.min()),
        "variants_private_max": float(variant_private.max()),
        "selected_private": float(picked_private.mean()),
        "mean_reported": float(picked_reported.mean()),
        "optimism": float((picked_private - picked_reported).mean()),
    }
    # 返回这一组的全部数字


def selection_seed_robustness(
    x_private: np.ndarray,
    y_private: np.ndarray,
    seeds: tuple[int, ...],
    n_repeats: int = 200,
    n_train: int = DEFAULT_TRAIN_SIZE,
    pool_size: int = POOL_SIZE,
    public_size: int = PUBLIC_SIZE,
    degrees: tuple[int, ...] = DEGREES,
) -> list[dict]:
    """换几个随机种子把实验重跑一遍，看乐观偏差这个数字稳不稳。

    参数：
        x_private / y_private：private 榜（与其他实验共用同一份）。
        seeds：要试的随机种子列表。
        n_repeats：每个种子重复多少次。
        n_train / pool_size / public_size / degrees：与主实验一致的设定。
    返回：每个种子一行，含乐观偏差、平均报告分数、平均 private 分数。

    为什么需要：乐观偏差是"重复若干次实验再取平均"的统计量，
        而每次重复的结果分布很宽（有时候挑对了还倒赚，有时候亏一大截），
        所以均值本身也带误差。只报一个种子的数字容易让人以为它是精确值，
        这里把多个种子的结果并排给出，读者才看得出"这个结论稳不稳"。
    """
    rows = []
    # 每个种子一行
    for seed in seeds:
        # 逐个种子
        summary = repeated_selection_experiment(
            x_private,
            y_private,
            n_repeats=n_repeats,
            n_train=n_train,
            pool_size=pool_size,
            public_size=public_size,
            degrees=degrees,
            seed=seed,
        )
        # 完整跑一遍三种选法
        rows.append(
            {
                "seed": seed,
                "public_optimism": summary["public"]["optimism"],
                "public_reported": summary["public"]["mean_reported"],
                "public_private": summary["public"]["mean_private"],
                "holdout_optimism": summary["holdout"]["optimism"],
                "holdout_private": summary["holdout"]["mean_private"],
                "best_possible": summary["best_possible"],
            }
        )
        # 记录这个种子的结果
    return rows
    # 返回所有种子


def training_size_sweep(
    x_private: np.ndarray,
    y_private: np.ndarray,
    sizes: tuple[int, ...] = (10, 30, 60, 1500),
    n_repeats: int = 200,
    public_size: int = PUBLIC_SIZE,
    pool_size: int = POOL_SIZE,
    degrees: tuple[int, ...] = DEGREES,
    seed: int = 0,
) -> list[dict]:
    """扫"训练集大小"，看数据量怎么改变"选模型"的难度。

    参数：
        x_private / y_private：private 榜。
        sizes：要试的训练集大小。1500 对应"2000 点大训练池切 1500 训练 / 500 验证"。
        n_repeats：每个大小重复多少次。
        public_size：公榜大小。
        pool_size：池子大小。
        degrees：候选次数集合。
        seed：随机种子。
    返回：每个训练规模一行的字典列表。

    看什么：训练集变大以后，5 次多项式的方差被压下去，1~5 次之间的真实差距缩小，
        选模型这件事整体变简单——错误流程的代价和乐观偏差都会变小。
        但"公榜报告分数虚高"这件事不会消失，因为它是"挑最小值"这个动作本身带来的，
        与训练集大小无关。
    """
    rng = np.random.default_rng(seed)
    # 随机数流
    rows = []
    # 每个规模一行
    for size in sizes:
        # 逐规模
        holdout_private = np.empty(n_repeats)
        # 正确流程选中的模型，在 private 榜上的分数
        public_private = np.empty(n_repeats)
        # 错误流程选中的模型，在 private 榜上的分数
        public_reported = np.empty(n_repeats)
        # 错误流程报出去的分数
        best_possible = np.empty(n_repeats)
        # 理论最优
        private_track = []
        # 每一轮各个候选在 private 榜上的分数，用来算"各次数的平均误差"
        for repeat in range(n_repeats):
            # 逐次重复
            x_train, y_train, x_val, y_val = _split_pool(rng, size, pool_size)
            # 同一个池子里切训练/验证
            x_board, y_board = sample_dataset(rng, public_size)
            # 另抽 50 点公榜
            coefs = _fit_all(x_train, y_train, degrees)
            # 训练候选
            val_scores = _score_all(coefs, degrees, x_val, y_val)
            # 验证误差
            board_scores = _score_all(coefs, degrees, x_board, y_board)
            # 公榜误差
            private_scores = _score_all(coefs, degrees, x_private, y_private)
            # private 真误差
            holdout_pick = int(np.argmin(val_scores))
            # 验证集选的
            public_pick = int(np.argmin(board_scores))
            # 公榜选的
            holdout_private[repeat] = private_scores[holdout_pick]
            # 记录
            public_private[repeat] = private_scores[public_pick]
            # 记录
            public_reported[repeat] = board_scores[public_pick]
            # 记录
            best_possible[repeat] = private_scores.min()
            # 记录
            private_track.append(private_scores)
            # 累积这一轮各个候选的 private 分数
        per_degree = np.mean(np.array(private_track), axis=0)
        # 各个次数在 private 榜上的平均误差（"如果只用一个固定次数"会拿到多少分）
        per_degree_median = np.median(np.array(private_track), axis=0)
        # 各个次数在 private 榜上误差的中位数（重尾时中位数比均值更能代表"典型情况"）
        per_degree_max = np.max(np.array(private_track), axis=0)
        # 各个次数最坏的那一次（用来看重尾有多重）
        rows.append(
            {
                "n_train": size,
                "n_val": pool_size - size,
                "holdout_private": float(holdout_private.mean()),
                "public_private": float(public_private.mean()),
                "public_reported": float(public_reported.mean()),
                "optimism": float((public_private - public_reported).mean()),
                "best_possible": float(best_possible.mean()),
                "regret": float(public_private.mean() - holdout_private.mean()),
                "per_degree_private": per_degree,
                "per_degree_median": per_degree_median,
                "per_degree_max": per_degree_max,
            }
        )
        # 一行汇总
    return rows
    # 返回所有规模
