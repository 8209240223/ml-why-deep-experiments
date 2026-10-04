#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
用纯 NumPy 手写的 softmax 多层感知机（MLP）：对应李宏毅《Why Deep》PPT 中"全蓝"的端到端模型。

网络结构（一行写清楚）：64 维原始像素 -> 64 单元 ReLU 隐藏层 -> 10 类 softmax 输出。

为什么不用 PyTorch：本机 /c/msys64/ucrt64/bin/python 没有安装 torch，
而这里网络极小（参数总量不到 5 千个），用 NumPy 手写前向/反向传播代码量完全可控，
还能把"隐藏层第一层的 64 个权重向量其实就是网络自己学出来的 8x8 特征探测器"
这件事看得一清二楚——这正是本实验要展示的重点。

和 features.py 的关系：
  features.py 里的绿色模块是人写死的公式，本文件的 W1 是"从数据里学出来的特征提取器"，
  两者结构上是同一层（输入都是 64 维像素、输出都是 64 维），区别只在参数从哪来。

用法：
    from model import SoftmaxMLP
    net = SoftmaxMLP(seed=0)
    net.fit(X_train, y_train, epochs=200, batch_size=64, eval_set=(X_test, y_test))
    net.accuracy(X_test, y_test)
"""

from __future__ import annotations
# 打开延迟注解求值，类型标注里可以直接写 np.ndarray

from dataclasses import dataclass, field
# dataclass 用来定义"每个 epoch 记录一条"的训练日志结构，省掉手写 __init__

import numpy as np
# 全部数值计算用 NumPy 完成：矩阵乘法、广播、随机数

DEFAULT_INPUT_DIM = 64
# 输入维度：8x8 = 64，也就是把图片直接拉平，不做任何人工作用上的特征工程

DEFAULT_HIDDEN_DIM = 64
# 隐藏层宽度：64 个 ReLU 单元；刻意取成 64，和"人工特征 16/20 维"以及"原始像素 64 维"都好对照

DEFAULT_NUM_CLASSES = 10
# 输出维度：手写数字 0~9 共 10 类

ADAM_BETA1 = 0.9
# Adam 一阶动量（梯度的指数滑动平均）的衰减系数，用于平滑随机梯度噪声

ADAM_BETA2 = 0.999
# Adam 二阶动量（梯度平方的指数滑动平均）的衰减系数，用于给每个参数自适应地缩放步长

ADAM_EPS = 1e-8
# 防止 Adam 更新时除以 0 的极小常数


@dataclass
class EpochLog:
    """一个 epoch 结束时的训练快照，用于画训练曲线和判断有没有过拟合。"""

    epoch: int
    # 第几个 epoch（从 1 开始计数）
    train_loss: float
    # 该 epoch 结束时全训练集上的平均交叉熵损失
    train_accuracy: float
    # 该 epoch 结束时全训练集上的分类准确率
    eval_loss: float | None = None
    # 该 epoch 结束时在外部验证/测试集上的平均损失；没提供评估集时为 None
    eval_accuracy: float | None = None
    # 该 epoch 结束时在外部验证/测试集上的准确率；没提供评估集时为 None


def softmax(logits: np.ndarray) -> np.ndarray:
    """按行做数值稳定的 softmax，把 logits 变成概率。

    参数：
        logits：形状 (n, c) 的浮点数组，每一行是某个样本对 10 个类别的原始打分。
    返回：形状 (n, c) 的概率矩阵，每一行非负且和为 1。

    为什么要减去每行最大值：exp 在大数上会溢出成 inf，减去行最大值后，
    指数的自变量不超过 0，结果一定落在 (0, 1]，且数学上与不减时完全等价
    （分子分母同乘一个常数 e^{-max} 不改变比值）。
    """
    shifted = logits - logits.max(axis=1, keepdims=True)
    # 每行减去该行最大值，keepdims=True 让形状保持 (n, 1) 以便广播到 (n, c)
    exp_values = np.exp(shifted)
    # 逐元素取指数，此时所有元素都不超过 1，不会溢出
    return exp_values / exp_values.sum(axis=1, keepdims=True)
    # 逐行归一化，除以每行之和，使每行概率之和为 1


def one_hot(labels: np.ndarray, num_classes: int = DEFAULT_NUM_CLASSES) -> np.ndarray:
    """把整数标签转成 one-hot 矩阵。

    参数：
        labels：形状 (n,) 的整数数组，取值 0~9。
        num_classes：类别总数。
    返回：形状 (n, num_classes) 的 0/1 浮点矩阵。

    为什么需要：交叉熵损失写成矩阵形式时要和 softmax 输出逐元素相乘，
    把标签铺成 one-hot 后就能直接用 numpy 的逐元素运算，不必写循环。
    """
    out = np.zeros((labels.shape[0], num_classes), dtype=np.float64)
    # 先建一个全 0 的矩阵，行数等于样本数
    out[np.arange(labels.shape[0]), labels] = 1.0
    # 花式索引：第 i 行第 labels[i] 列置 1，其余保持 0
    return out
    # 返回 (n, num_classes) 的 one-hot 矩阵


def cross_entropy(probabilities: np.ndarray, targets_one_hot: np.ndarray) -> float:
    """计算一批样本的平均交叉熵损失。

    参数：
        probabilities：形状 (n, c) 的 softmax 输出概率。
        targets_one_hot：形状 (n, c) 的 one-hot 真标签。
    返回：一个标量，表示这批样本的平均损失。

    公式：L = -(1/n) * sum_i sum_c  y_ic * log(p_ic)
    因为 y 是 one-hot，内层求和实际只留下"真实类别那一项"的 -log(p)，
    所以损失可以直白地理解为"真实类别的预测概率越小，罚得越狠"。
    """
    clipped = np.clip(probabilities, 1e-12, 1.0)
    # 把概率截断到不小于 1e-12，避免真实类别概率恰好为 0 时 log(0) = -inf
    return float(-(targets_one_hot * np.log(clipped)).sum(axis=1).mean())
    # 逐元素相乘后按行求和，得到每个样本的损失，再对样本取平均并转成 Python float


class SoftmaxMLP:
    """单隐藏层 softmax 分类器：64 -> 64(ReLU) -> 10(softmax)，用 Adam 做 mini-batch 梯度下降。

    参数：
        input_dim：输入维度，默认 64（8x8 像素拉平）。
        hidden_dim：隐藏层单元数，默认 64。
        num_classes：类别数，默认 10。
        seed：随机种子，固定后初始化权重完全可复现。
        learning_rate：Adam 的学习率。
        l2：权重衰减系数（只作用在两个权重矩阵上，不罚偏置），0 表示只优化交叉熵。
    """

    def __init__(
        self,
        input_dim: int = DEFAULT_INPUT_DIM,
        hidden_dim: int = DEFAULT_HIDDEN_DIM,
        num_classes: int = DEFAULT_NUM_CLASSES,
        seed: int = 0,
        learning_rate: float = 1e-3,
        l2: float = 0.0,
    ) -> None:
        self.input_dim = input_dim
        # 记下输入维度，forward 里做形状断言、外部想画权重图时也用得上
        self.hidden_dim = hidden_dim
        # 记下隐藏层宽度，即"网络自己学出来的特征个数"
        self.num_classes = num_classes
        # 记下类别数
        self.learning_rate = learning_rate
        # 保存学习率，fit 里的 Adam 更新要用
        self.l2 = l2
        # 保存权重衰减系数，算梯度时要用
        rng = np.random.default_rng(seed)
        # 显式构造随机数生成器对象（而不是用全局 np.random），保证不同实验之间互不干扰、结果可复现
        self.W1 = rng.normal(0.0, np.sqrt(2.0 / input_dim), size=(input_dim, hidden_dim))
        # 第一层权重 (64, 64)：He 初始化，标准差 sqrt(2/fan_in)，让 ReLU 前的激活值方差保持稳定
        self.b1 = np.zeros(hidden_dim, dtype=np.float64)
        # 第一层偏置初始化为 0，这是常规做法（对称性已由随机权重打破）
        self.W2 = rng.normal(0.0, np.sqrt(1.0 / hidden_dim), size=(hidden_dim, num_classes))
        # 第二层权重 (64, 10)：Xavier 初始化，标准差 sqrt(1/fan_in)，配合 softmax 输出更稳
        self.b2 = np.zeros(num_classes, dtype=np.float64)
        # 第二层偏置同样初始化为 0
        self.history: list[EpochLog] = []
        # 训练日志列表，每过一个 epoch 追加一条 EpochLog
        self._adam_m = [np.zeros_like(self.W1), np.zeros_like(self.b1), np.zeros_like(self.W2), np.zeros_like(self.b2)]
        # Adam 的一阶动量，按 (W1, b1, W2, b2) 的顺序各存一份，形状与参数一致
        self._adam_v = [np.zeros_like(self.W1), np.zeros_like(self.b1), np.zeros_like(self.W2), np.zeros_like(self.b2)]
        # Adam 的二阶动量，形状与参数一致
        self._adam_step = 0
        # Adam 的全局步数计数器，用来做偏差校正（第 1 步时动量还接近 0，需要放大）

    def forward(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """前向传播，返回中间激活值和最终概率，一次算完供反向传播复用。

        参数：
            X：形状 (n, 64) 的输入，每行是一张拉平并归一化到 0~1 的数字图。
        返回：(Z1, A1, P)
            Z1：形状 (n, 64)，第一层的线性输出 W1*x + b1；
            A1：形状 (n, 64)，ReLU 之后的隐藏层激活，也就是"网络自己学出的 64 维特征"；
            P ：形状 (n, 10)，softmax 之后的类别概率。
        """
        Z1 = X @ self.W1 + self.b1
        # 第一层线性变换：矩阵乘法把 64 维像素映射到 64 维隐藏空间，再加偏置（广播到每一行）
        A1 = np.maximum(Z1, 0.0)
        # ReLU 激活：逐元素取 max(0, z)，负数被压成 0，这一操作让网络能拟合非线性边界
        Z2 = A1 @ self.W2 + self.b2
        # 第二层线性变换：把 64 维隐藏特征映射到 10 个类别的打分
        P = softmax(Z2)
        # 归一化成概率，所有后续损失和预测都用它
        return Z1, A1, P
        # 返回三个中间量：Z1 的符号在反向传播里判断 ReLU 是否导通，A1 用来算第二层梯度

    def _backward(
        self,
        X: np.ndarray,
        targets_one_hot: np.ndarray,
        Z1: np.ndarray,
        A1: np.ndarray,
        P: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """反向传播，计算四个参数各自的平均梯度。

        参数：
            X：形状 (n, 64) 的输入批次。
            targets_one_hot：形状 (n, 10) 的 one-hot 标签。
            Z1, A1, P：前向传播留下的中间结果。
        返回：(dW1, db1, dW2, db2)，形状分别与对应参数一致。

        推导要点：softmax 加交叉熵组合求导后，对第二层打分的梯度异常简洁地等于 (P - Y) / n，
        这就是"softmax + 交叉熵"被称为好用搭档的原因之一：省掉了 softmax 的雅可比矩阵。
        """
        n = X.shape[0]
        # 批次大小，所有梯度最后都要除以它取平均
        dZ2 = (P - targets_one_hot) / n
        # 第二层打分的梯度：(预测概率 - 真标签) 再对批次取平均，形状 (n, 10)
        dW2 = A1.T @ dZ2
        # 链式法则：a1 的转置乘 dZ2，得到权重梯度，形状 (64, 10)
        db2 = dZ2.sum(axis=0)
        # 偏置梯度等于该批次内所有样本梯度的和，形状 (10,)
        dA1 = dZ2 @ self.W2.T
        # 把第二层的梯度往回传，得到隐藏层激活的梯度，形状 (n, 64)
        dZ1 = dA1 * (Z1 > 0.0)
        # ReLU 的导数：z>0 处为 1、z<=0 处为 0，所以用布尔矩阵逐元素相乘把"关掉的神经元"的梯度清零
        dW1 = X.T @ dZ1
        # 再往前传一层，得到第一层权重梯度，形状 (64, 64)
        db1 = dZ1.sum(axis=0)
        # 第一层偏置梯度，形状 (64,)
        if self.l2 > 0.0:
            # 只有显式开启权重衰减时才加正则项（默认关闭，保持"纯交叉熵"的教学口径）
            dW1 = dW1 + self.l2 * self.W1
            # L2 惩罚对权重的梯度就是权重本身乘以系数，注意不作用于偏置
            dW2 = dW2 + self.l2 * self.W2
            # 第二层权重同样处理
        return dW1, db1, dW2, db2
        # 四个梯度一次性返回，交给 _adam_update 更新参数

    def _adam_update(self, gradients: list[np.ndarray]) -> None:
        """用 Adam 规则更新四个参数：动量 + 自适应步长 + 偏差校正。

        参数：
            gradients：长度 4 的列表，顺序为 [dW1, db1, dW2, db2]。
        返回：无（原地修改 self.W1 / self.b1 / self.W2 / self.b2）。

        为什么用 Adam 而不是最朴素的 SGD：digits 只有一千多个训练样本，
        朴素 SGD 对学习率很敏感、需要仔细调参才能收敛；Adam 对每个参数自适应地缩放步长，
        默认学习率 1e-3 就能稳定收敛，课堂上不必把时间花在调参上。
        Adam 仍然是 mini-batch 梯度下降的一种变体，优化目标完全没变。
        """
        self._adam_step += 1
        # 全局步数加一，偏差校正公式里要用
        parameters = [self.W1, self.b1, self.W2, self.b2]
        # 把四个参数放进列表，按下标统一处理
        for index in range(4):
            # 依次更新 W1、b1、W2、b2
            grad = gradients[index]
            # 取出当前参数对应的梯度
            self._adam_m[index] = ADAM_BETA1 * self._adam_m[index] + (1.0 - ADAM_BETA1) * grad
            # 一阶动量：梯度的指数滑动平均，起到"平滑随机梯度噪声"的作用
            self._adam_v[index] = ADAM_BETA2 * self._adam_v[index] + (1.0 - ADAM_BETA2) * grad * grad
            # 二阶动量：梯度平方的指数滑动平均，反映每个参数梯度的历史波动幅度
            m_hat = self._adam_m[index] / (1.0 - ADAM_BETA1**self._adam_step)
            # 偏差校正：训练初期动量被初始化为 0 拉低，除以 (1 - beta1^t) 放大回正常尺度
            v_hat = self._adam_v[index] / (1.0 - ADAM_BETA2**self._adam_step)
            # 二阶动量的偏差校正，同理
            parameters[index] -= self.learning_rate * m_hat / (np.sqrt(v_hat) + ADAM_EPS)
            # 参数更新：减去"学习率 * 校正后的一阶动量 / 校正后的梯度均方根"，梯度大的方向步长自动变小
        self.W1, self.b1, self.W2, self.b2 = parameters
        # 把更新后的参数写回实例属性（列表里存的是同一批数组对象，这行主要是为了语义清晰）

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        epochs: int = 200,
        batch_size: int = 64,
        eval_set: tuple[np.ndarray, np.ndarray] | None = None,
        seed: int = 0,
        verbose: bool = False,
    ) -> list[EpochLog]:
        """用 mini-batch 梯度下降训练网络，每个 epoch 记录一次训练/评估指标。

        参数：
            X：形状 (n, 64) 的训练输入。
            y：形状 (n,) 的训练标签。
            epochs：把训练集完整扫一遍算一个 epoch，这里是总轮数。
            batch_size：每个 mini-batch 的样本数，越小梯度噪声越大、更新越频繁。
            eval_set：(X_eval, y_eval) 元组，给了就在每个 epoch 末尾额外评估一次；
                      本实验传入的是测试集，只用于画曲线和观察泛化，不参与任何训练决策。
            seed：打乱训练集顺序用的随机种子，固定后每个 epoch 的批次划分完全一致。
            verbose：True 时每隔若干轮打印一行进度。
        返回：EpochLog 列表，长度等于 epochs。
        """
        rng = np.random.default_rng(seed)
        # 只为"每个 epoch 打乱样本顺序"服务的随机数生成器
        targets_one_hot = one_hot(y, self.num_classes)
        # 训练标签一次性转成 one-hot，避免在循环里反复转换
        self.history = []
        # 清空旧日志，保证重复调用 fit 时 history 只反映本次训练
        self._adam_m = [np.zeros_like(self.W1), np.zeros_like(self.b1), np.zeros_like(self.W2), np.zeros_like(self.b2)]
        # 重置一阶动量；重新用 fit 训练相当于从头开始，动量不该带着上一轮的惯性
        self._adam_v = [np.zeros_like(self.W1), np.zeros_like(self.b1), np.zeros_like(self.W2), np.zeros_like(self.b2)]
        # 重置二阶动量，理由同上
        self._adam_step = 0
        # 重置 Adam 步数计数器，偏差校正从第 1 步重新开始

        for epoch in range(1, epochs + 1):
            # 外层循环：一个 epoch 就是把训练集按 mini-batch 扫一遍
            order = rng.permutation(X.shape[0])
            # 随机打乱样本下标，防止固定顺序让梯度下降反复沿同一方向走、影响收敛
            for start in range(0, X.shape[0], batch_size):
                # 内层循环：每次取连续 batch_size 个（下标已经被打乱过）样本组成一个 mini-batch
                index = order[start : start + batch_size]
                # 取出当前批次的样本下标，最后一批可能不足 batch_size
                X_batch = X[index]
                # 该批次的输入
                y_batch = targets_one_hot[index]
                # 该批次的 one-hot 标签
                Z1, A1, P = self.forward(X_batch)
                # 前向传播拿到中间结果和预测概率
                gradients = self._backward(X_batch, y_batch, Z1, A1, P)
                # 反向传播算出四个参数的梯度
                self._adam_update(list(gradients))
                # 用 Adam 规则更新参数
            _, _, train_probabilities = self.forward(X)
            # 一个 epoch 结束后，在全训练集上算一次损失与准确率，作为这条日志的"训练侧"指标
            train_loss = cross_entropy(train_probabilities, targets_one_hot)
            # 训练集平均交叉熵
            train_accuracy = float((train_probabilities.argmax(axis=1) == y).mean())
            # 训练集准确率：取概率最大的类别与真标签比较，再对样本求平均
            eval_loss = None
            # 默认没有评估指标
            eval_accuracy = None
            # 默认没有评估指标
            if eval_set is not None:
                # 只有传了评估集才额外算一次
                X_eval, y_eval = eval_set
                # 拆出评估集输入和标签
                _, _, eval_probabilities = self.forward(X_eval)
                # 前向传播（注意这里不更新参数，纯评估）
                eval_loss = cross_entropy(eval_probabilities, one_hot(y_eval, self.num_classes))
                # 评估集平均交叉熵
                eval_accuracy = float((eval_probabilities.argmax(axis=1) == y_eval).mean())
                # 评估集准确率
            self.history.append(EpochLog(epoch, train_loss, train_accuracy, eval_loss, eval_accuracy))
            # 把这一轮的快照追加进日志
            if verbose and (epoch == 1 or epoch % 20 == 0 or epoch == epochs):
                # 只在第 1、每 20 轮、以及最后一轮打印，避免刷屏
                message = f"    epoch {epoch:>3}/{epochs}  训练损失 {train_loss:.4f}  训练准确率 {train_accuracy:.4f}"
                # 组装基础信息
                if eval_accuracy is not None:
                    # 有评估集时把评估指标也接上去
                    message += f"  测试准确率 {eval_accuracy:.4f}"
                    # 追加测试准确率
                print(message)
                # 输出这一行
        return self.history
        # 返回完整训练日志，供画训练曲线

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """返回样本属于各类别的概率矩阵。

        参数：
            X：形状 (n, 64) 的输入。
        返回：形状 (n, 10) 的概率矩阵。
        """
        _, _, probabilities = self.forward(X)
        # 前向传播只关心最后的概率输出，中间量用下划线丢弃
        return probabilities
        # 返回概率矩阵

    def predict(self, X: np.ndarray) -> np.ndarray:
        """返回每个样本的预测类别。

        参数：
            X：形状 (n, 64) 的输入。
        返回：形状 (n,) 的整数标签数组。
        """
        return self.predict_proba(X).argmax(axis=1)
        # 取每行概率最大的那一列作为预测类别

    def accuracy(self, X: np.ndarray, y: np.ndarray) -> float:
        """计算在给定数据上的分类准确率。

        参数：
            X：形状 (n, 64) 的输入。
            y：形状 (n,) 的真标签。
        返回：0~1 之间的浮点准确率。
        """
        return float((self.predict(X) == y).mean())
        # 预测标签与真标签逐元素比较，布尔数组求均值就是准确率

    def first_layer_filters(self) -> np.ndarray:
        """把第一层权重整理成"一张图一个神经元"的形式，用于可视化。

        参数：无。
        返回：形状 (hidden_dim, 8, 8) 的数组，第 k 张图是第 k 个隐藏单元对 64 个像素的权重。

        为什么可以这样看：隐藏单元 k 的输入是 W1[:, k] 与图像像素的内积，
        所以 W1[:, k] 本身就是一个 8x8 的模板——这就是"网络自己学出来的特征探测器"，
        和 features.py 里人工写死的行/列投影形成直接对照。
        """
        return self.W1.T.reshape(self.hidden_dim, 8, 8)
        # W1 的形状是 (64 个像素, 64 个隐藏单元)，转置后每行是一个单元的权重，再拉成 8x8 小图
