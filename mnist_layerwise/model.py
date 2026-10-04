#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
用纯 NumPy 手写的「3 个隐藏层」softmax 多层感知机（MLP）。

对应的课程内容：李宏毅《Why Deep》里那一页 “把同一批 MNIST 样本分别画在
input / 1-st hidden / 2-nd hidden / 3-rd hidden 四个表示空间里” 的图。
本文件只负责造出那个深层网络并把它训练好，同时暴露一个
“取出任意一层输出表示” 的接口，交给 analysis.py 算指标、figures.py 画图。

网络结构（一行写清楚）：
    64 维原始像素 -> 128(ReLU) -> 64(ReLU) -> 32(ReLU) -> 10(softmax)

为什么选 64 -> 128 -> 64 -> 32 -> 10 这个「先变宽再逐层收窄」的金字塔：
  1) 与 PPT 原页那个深层网络同形：原页是 784 -> 2000 -> 1000 -> 500 -> 10，
     第一层明显比输入宽、之后逐层减半收窄；本实验按同一节奏缩小到 8x8 图像能承受的规模。
  2) 第一层给 128 个单元（比输入的 64 维还宽）是必要的：8x8 图像里可辨认的笔画模板
     （横、竖、撇、捺、弧）比 64 个还多，第一层太窄会让后面几层无米下炊。
  3) 最深一层收到 32 维，既保留了足够的自由度让十类分开，又保证降维到 2 维时
     “十个团”这件事不是靠维度浪费堆出来的。
  4) 三层隐藏层正是 PPT 那张图的层数，四宫格刚好对应 input + 3 个隐藏层。

为什么不用 PyTorch / sklearn 的 MLPClassifier：
  本机 /c/msys64/ucrt64/bin/python 没有装 torch（见 README 的环境说明），
  而 sklearn 的 MLPClassifier 只在 coefs_ 里按层暴露权重，要把“第 k 层输出表示”取出来
  还得自己重放一遍前向传播、且拿不到中间激活的缓存，可控性反而更差。
  这里网络极小（参数总量约 1.5 万个），NumPy 手写前向/反向传播只有两百行左右，
  每一层的形状、每一处梯度都能指名道姓地讲清楚——这正是本教学实验想要的透明度。

用法：
    from model import DeepMLP
    net = DeepMLP(seed=42)
    net.fit(X_train, y_train, epochs=300, batch_size=64, eval_set=(X_test, y_test))
    net.accuracy(X_test, y_test)
    nets_repr = net.representations(X_all)   # [input, h1, h2, h3] 四种表示
"""

from __future__ import annotations
# 打开延迟注解求值：类型标注里可以直接写 np.ndarray 而不用加引号，也不要求运行时真的有这个对象

from dataclasses import dataclass
# dataclass 用来定义“每个 epoch 记一条”的训练日志结构，省掉手写 __init__ 的样板代码

import numpy as np
# 全部数值计算都用 NumPy 完成：矩阵乘法、广播、随机数生成

DEFAULT_LAYER_SIZES = (64, 128, 64, 32, 10)
# 各层宽度：第 0 项是输入层（8x8=64 像素），中间三项是三个隐藏层，最后一项是 10 类输出
# 相邻两项之间有一个权重矩阵，所以本网络一共 4 个权重矩阵（3 个隐藏层 + 1 个输出层）

REPRESENTATION_NAMES = ("input", "1-st hidden", "2-nd hidden", "3-rd hidden")
# 四宫格四个子图的标题；顺序与 DeepMLP.representations() 返回的四个数组一一对应
# 这几个字符串刻意保留 PPT 上的英文原文（input / 1-st hidden / 2-nd hidden / 3-rd hidden）

ADAM_BETA1 = 0.9
# Adam 一阶动量（梯度的指数滑动平均）的衰减系数，作用是平滑随机梯度的噪声

ADAM_BETA2 = 0.999
# Adam 二阶动量（梯度平方的指数滑动平均）的衰减系数，作用是按每个参数自身的历史波动缩放步长

ADAM_EPS = 1e-8
# 防止 Adam 更新时除以 0 的极小常数


@dataclass
class EpochLog:
    """一个 epoch 结束时的训练快照，用来画训练曲线、判断有没有过拟合。"""

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
    """按行做数值稳定的 softmax，把 10 个类别打分变成概率。

    参数：
        logits：形状 (n, 10) 的浮点数组，每一行是某个样本对 10 个类别的原始打分。
    返回：形状 (n, 10) 的概率矩阵，每一行非负且和为 1。

    为什么要减去每行最大值：exp 在大数上会溢出成 inf，减去行最大值之后
    指数的自变量不超过 0，结果一定落在 (0, 1] 区间内；而数学上与不减时完全等价
    （分子分母同时乘上 e^{-max}，比值不变）。
    """
    shifted = logits - logits.max(axis=1, keepdims=True)
    # 每行减去该行最大值；keepdims=True 让形状保持 (n, 1)，以便广播回 (n, 10)
    exp_values = np.exp(shifted)
    # 逐元素取指数，此时所有元素都不超过 1，绝不会溢出
    return exp_values / exp_values.sum(axis=1, keepdims=True)
    # 逐行归一化：除以每行之和，使每行概率之和恰好为 1


def one_hot(labels: np.ndarray, num_classes: int) -> np.ndarray:
    """把整数标签转成 one-hot 矩阵。

    参数：
        labels：形状 (n,) 的整数数组，取值 0~9。
        num_classes：类别总数，本实验是 10。
    返回：形状 (n, num_classes) 的 0/1 浮点矩阵。

    为什么需要：交叉熵写成矩阵形式后要和 softmax 输出逐元素相乘，
    把标签铺成 one-hot 就能直接用 NumPy 的向量化运算，不必写 Python 循环。
    """
    out = np.zeros((labels.shape[0], num_classes), dtype=np.float64)
    # 先建一个全 0 矩阵，行数等于样本数
    out[np.arange(labels.shape[0]), labels] = 1.0
    # 花式索引：把第 i 行第 labels[i] 列置成 1，其余列保持 0
    return out
    # 返回 (n, num_classes) 的 one-hot 矩阵


def cross_entropy(probabilities: np.ndarray, targets_one_hot: np.ndarray) -> float:
    """计算一批样本的平均交叉熵损失。

    参数：
        probabilities：形状 (n, 10) 的 softmax 输出概率。
        targets_one_hot：形状 (n, 10) 的 one-hot 真标签。
    返回：一个标量，表示这批样本的平均损失。

    公式：L = -(1/n) * sum_i sum_c y_ic * log(p_ic)
    因为 y 是 one-hot，内层求和实际只留下“真实类别那一项”的 -log(p)，
    所以这个损失可以直白地理解为“真实类别的预测概率越小，罚得越狠”。
    """
    clipped = np.clip(probabilities, 1e-12, 1.0)
    # 把概率截断到不小于 1e-12，避免真实类别概率恰好为 0 时出现 log(0) = -inf
    return float(-(targets_one_hot * np.log(clipped)).sum(axis=1).mean())
    # 逐元素相乘后按行求和得到每个样本的损失，再对样本取平均并转成 Python float


class DeepMLP:
    """3 隐藏层 softmax 分类器：64 -> 128(ReLU) -> 64(ReLU) -> 32(ReLU) -> 10(softmax)，用 Adam 训练。

    参数：
        layer_sizes：各层宽度元组，默认 (64, 128, 64, 32, 10)。
        seed：随机种子，固定后权重初始化与每个 epoch 的样本打乱顺序都完全可复现。
        learning_rate：Adam 的学习率。
        l2：权重衰减系数，只作用在权重矩阵上、不罚偏置；0 表示只优化交叉熵。
    """

    def __init__(
        self,
        layer_sizes: tuple[int, ...] = DEFAULT_LAYER_SIZES,
        seed: int = 42,
        learning_rate: float = 1e-3,
        l2: float = 1e-4,
    ) -> None:
        if len(layer_sizes) < 3:
            # 至少要“输入 + 1 个隐藏层 + 输出”，否则连一层可学的非线性都没有
            raise ValueError("layer_sizes 至少要有 3 项：输入层、一个隐藏层、输出层")
        # 参数校验：与其在报错时看到形状诡异的结果，不如在构造阶段就明确拦下
        self.layer_sizes = tuple(int(size) for size in layer_sizes)
        # 记下各层宽度，前向传播、可视化、报告里都要用
        self.num_hidden_layers = len(self.layer_sizes) - 2
        # 隐藏层个数 = 总层数 - 输入层 - 输出层；本实验为 3
        self.learning_rate = learning_rate
        # 保存学习率，Adam 更新时使用
        self.l2 = l2
        # 保存权重衰减系数，算梯度时使用
        rng = np.random.default_rng(seed)
        # 显式构造随机数生成器对象（而不是用全局 np.random），保证多次实验互不干扰、结果可复现
        self.weights: list[np.ndarray] = []
        # 权重矩阵列表：第 k 项形状为 (layer_sizes[k], layer_sizes[k+1])
        self.biases: list[np.ndarray] = []
        # 偏置向量列表：第 k 项形状为 (layer_sizes[k+1],)
        for index in range(len(self.layer_sizes) - 1):
            # 逐个构造 4 个变换层（3 个隐藏层 + 1 个输出层）
            fan_in = self.layer_sizes[index]
            # 这一层的输入维度（扇入）：用来定初始化的标准差
            fan_out = self.layer_sizes[index + 1]
            # 这一层的输出维度（扇出）
            is_output_layer = index == len(self.layer_sizes) - 2
            # 最后一层是输出层；它后面接的是 softmax 而不是 ReLU，初始化策略要区别对待
            if is_output_layer:
                # 输出层用 Xavier 初始化：标准差 sqrt(1/fan_in)
                std = np.sqrt(1.0 / fan_in)
                # 让 logits 的方差稳定在 O(1)，softmax 一开始不会退化成几乎 one-hot 的极端分布
            else:
                # 隐藏层用 He 初始化：标准差 sqrt(2/fan_in)
                std = np.sqrt(2.0 / fan_in)
                # 系数里的 2 是专门补偿 ReLU 把一半神经元压成 0 导致的方差减半
            self.weights.append(rng.normal(0.0, std, size=(fan_in, fan_out)))
            # 按正态分布采样权重矩阵，形状 (fan_in, fan_out)
            self.biases.append(np.zeros(fan_out, dtype=np.float64))
            # 偏置初始化为 0：这是常规做法，权重里的随机性已经足够打破神经元之间的对称性
        self.history: list[EpochLog] = []
        # 训练日志列表，每过一个 epoch 追加一条 EpochLog
        self._adam_m = [np.zeros_like(weight) for weight in self.weights] + [
            np.zeros_like(bias) for bias in self.biases
        ]
        # Adam 一阶动量：按 [W1..W4, b1..b4] 的顺序各存一份，形状与对应参数一致
        self._adam_v = [np.zeros_like(weight) for weight in self.weights] + [
            np.zeros_like(bias) for bias in self.biases
        ]
        # Adam 二阶动量，形状与对应参数一致
        self._adam_step = 0
        # Adam 全局步数计数器，用于偏差校正（第 1 步时动量还接近 0，需要放大回来）

    def parameter_count(self) -> int:
        """统计网络中可训练参数的总个数，写进报告用。

        参数：无。
        返回：权重与偏置的元素总数（整数）。
        """
        total = sum(int(weight.size) for weight in self.weights)
        # 所有权重矩阵的元素个数之和
        total += sum(int(bias.size) for bias in self.biases)
        # 再加上所有偏置向量的元素个数
        return total
        # 返回总量

    def forward(self, X: np.ndarray) -> tuple[list[np.ndarray], list[np.ndarray], np.ndarray]:
        """前向传播，一次算完所有中间结果供反向传播和可视化复用。

        参数：
            X：形状 (n, 64) 的输入，每行是一张拉平并归一化到 0~1 的数字图。
        返回：(activations, pre_activations, probabilities)
            activations：长度 4 的列表 [X, h1, h2, h3]，
                         第 0 项是输入表示，第 1~3 项分别是三个隐藏层的 ReLU 输出表示
                         ——这正是本实验要在四宫格里画出来的四种表示。
            pre_activations：长度 3 的列表 [z1, z2, z3]，各隐藏层 ReLU 之前的线性输出；
                             ReLU 的导数要用 z 的符号来判断，所以必须留下来。
            probabilities：形状 (n, 10) 的 softmax 概率。
        """
        activations = [X]
        # 表示列表的第一项就是输入本身，不做任何变换，对应四宫格的 “input” 面板
        pre_activations = []
        # 各隐藏层 ReLU 之前的线性输出
        current = X
        # current 始终指向“当前这一层算完之后的表示”，初始为输入
        for index in range(len(self.weights) - 1):
            # 只遍历隐藏层（不含最后的输出层），本实验循环 3 次
            linear = current @ self.weights[index] + self.biases[index]
            # 第 index 个隐藏层的线性变换；偏置 (fan_out,) 会自动广播到每个样本
            pre_activations.append(linear)
            # 把 ReLU 之前的线性输出存下来，反向传播里判断“哪些神经元导通”要用
            current = np.maximum(linear, 0.0)
            # ReLU 激活：逐元素取 max(0, z)，负数被压成 0，正是这个非线性让深层网络能拟合复杂边界
            activations.append(current)
            # 把这一层 ReLU 之后的表示追加进列表，就是四宫格里“1-st/2-nd/3-rd hidden”三个面板的数据
        logits = current @ self.weights[-1] + self.biases[-1]
        # 输出层线性变换：把最后一层隐藏表示映射到 10 个类别的打分
        probabilities = softmax(logits)
        # 归一化成概率，后续的损失、预测、准确率都用它
        return activations, pre_activations, probabilities
        # 三类中间量一次性返回：activations 用来做可视化，pre_activations 用来算 ReLU 导数，probabilities 用来算损失

    def _backward(
        self,
        activations: list[np.ndarray],
        pre_activations: list[np.ndarray],
        probabilities: np.ndarray,
        targets_one_hot: np.ndarray,
    ) -> list[np.ndarray]:
        """反向传播，算出自下而上每一层参数的梯度。

        参数：
            activations：forward 返回的 [X, h1, h2, h3]。
            pre_activations：forward 返回的 [z1, z2, z3]。
            probabilities：形状 (n, 10) 的 softmax 输出。
            targets_one_hot：形状 (n, 10) 的 one-hot 标签。
        返回：长度 8 的梯度列表，顺序为 [dW1, dW2, dW3, dW4, db1, db2, db3, db4]，
              与 self.weights + self.biases 的顺序一一对应。

        推导要点：softmax 与交叉熵组合求导之后，对输出层打分的梯度异常简洁地等于 (P - Y) / n。
        这就是“softmax + 交叉熵”被称为好用搭档的原因之一：省掉了 softmax 雅可比矩阵那一大坨。
        之后每往前一层，就套一次链式法则，并把 ReLU 的导数（z>0 处为 1，否则为 0）逐元素乘上去。
        """
        n = activations[0].shape[0]
        # 批次大小，所有梯度最后都要除以它取平均
        gradients_w = [np.zeros_like(weight) for weight in self.weights]
        # 先把 4 个权重梯度都初始化成同形状的零矩阵，再按层填
        gradients_b = [np.zeros_like(bias) for bias in self.biases]
        # 4 个偏置梯度同样先置零
        delta = (probabilities - targets_one_hot) / n
        # 输出层打分的梯度 delta_L = (预测概率 - 真标签) / n，形状 (n, 10)
        for index in range(len(self.weights) - 1, -1, -1):
            # 从输出层往输入方向逐层回传，本实验循环 4 次（index 取 3, 2, 1, 0）
            previous_activation = activations[index]
            # 该层的输入就是这个层的上一层表示（index=0 时即原始像素 X）
            gradients_w[index] = previous_activation.T @ delta
            # 权重梯度 = 该层输入（转置）乘该层打分梯度，形状 (fan_in, fan_out)
            gradients_b[index] = delta.sum(axis=0)
            # 偏置梯度 = 该批次内所有样本的梯度之和，形状 (fan_out,)
            if index == 0:
                # 已经回传到最底层的输入层了，前面没有可更新的权重，结束循环
                break
            # 退出循环，避免再往前访问不存在的层
            delta_previous = delta @ self.weights[index].T
            # 把梯度往上一层传：乘上这一层的权重矩阵的转置，得到上一层表示的梯度
            delta = delta_previous * (pre_activations[index - 1] > 0.0)
            # ReLU 的导数：z>0 处为 1、z<=0 处为 0，
            # 所以用布尔矩阵逐元素相乘，把“被 ReLU 关掉的神经元”的梯度直接清零
        if self.l2 > 0.0:
            # 只有显式开启权重衰减时才追加正则项的梯度（默认开启，系数很小）
            for index in range(len(gradients_w)):
                # 对所有 4 个权重矩阵生效
                gradients_w[index] = gradients_w[index] + self.l2 * self.weights[index]
                # L2 惩罚对权重的梯度就是权重本身乘以系数；注意不作用于偏置
        return gradients_w + gradients_b
        # 权重梯度在前、偏置梯度在后拼成一个列表返回，顺序与 _adam_update 的约定一致

    def _adam_update(self, gradients: list[np.ndarray]) -> None:
        """用 Adam 规则更新全部参数：一阶动量 + 二阶动量 + 偏差校正。

        参数：
            gradients：长度 8 的梯度列表 [dW1..dW4, db1..db4]。
        返回：无（原地修改 self.weights 与 self.biases）。

        为什么用 Adam 而不是最朴素的 SGD：digits 只有一千多个训练样本，
        朴素 SGD 对学习率极其敏感、需要反复调参才能收敛；Adam 对每个参数自适应地缩放步长，
        默认 1e-3 就能稳定收敛，课堂上不必把时间花在调参上。
        需要强调：Adam 依然是 mini-batch 梯度下降的一种变体，优化目标（交叉熵）完全没变，
        它只是“怎么走”的策略不同——本实验想讲的是网络深不深，不该被优化器的脾气干扰。
        """
        self._adam_step += 1
        # 全局步数加一，偏差校正公式里要用
        parameters = self.weights + self.biases
        # 按 [W1..W4, b1..b4] 的顺序拼出参数列表，与梯度列表一一对应
        for index in range(len(parameters)):
            # 逐个参数做同样的更新
            grad = gradients[index]
            # 取出当前参数对应的梯度
            self._adam_m[index] = ADAM_BETA1 * self._adam_m[index] + (1.0 - ADAM_BETA1) * grad
            # 一阶动量：梯度的指数滑动平均，起到“平滑随机梯度噪声”的作用
            self._adam_v[index] = ADAM_BETA2 * self._adam_v[index] + (1.0 - ADAM_BETA2) * grad * grad
            # 二阶动量：梯度平方的指数滑动平均，反映该参数梯度的历史波动幅度
            m_hat = self._adam_m[index] / (1.0 - ADAM_BETA1**self._adam_step)
            # 偏差校正：训练初期动量被初始化的 0 拉低，除以 (1 - beta1^t) 放大回正常尺度
            v_hat = self._adam_v[index] / (1.0 - ADAM_BETA2**self._adam_step)
            # 二阶动量的偏差校正，同理
            parameters[index] -= self.learning_rate * m_hat / (np.sqrt(v_hat) + ADAM_EPS)
            # 参数更新：减去“学习率 × 校正后的一阶动量 / 校正后的梯度均方根”，
            # 梯度一直很大的方向步长会自动变小，梯度一直很小的方向步长会相对放大
        weight_count = len(self.weights)
        # 先记下权重矩阵的个数（本实验是 4），下面切分列表时要用；若不先存下来，后面会依赖列表已被替换的副作用
        self.weights = parameters[:weight_count]
        # 前 4 项写回权重列表
        self.biases = parameters[weight_count:]
        # 后 4 项写回偏置列表

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        epochs: int = 300,
        batch_size: int = 64,
        eval_set: tuple[np.ndarray, np.ndarray] | None = None,
        seed: int = 42,
        verbose: bool = False,
    ) -> list[EpochLog]:
        """用 mini-batch 梯度下降训练网络，每个 epoch 记录一次训练/评估指标。

        参数：
            X：形状 (n, 64) 的训练输入。
            y：形状 (n,) 的训练标签。
            epochs：把训练集完整扫一遍算一个 epoch，这里是总轮数。
            batch_size：每个 mini-batch 的样本数，越小梯度噪声越大、参数更新越频繁。
            eval_set：(X_eval, y_eval) 元组，给了就在每个 epoch 末尾额外评估一次。
                      本实验传入测试集，只用于画曲线和观察泛化，绝不参与任何训练决策。
            seed：打乱训练集顺序用的随机种子，固定后每个 epoch 的批次划分完全一致。
            verbose：True 时每隔若干轮打印一行进度。
        返回：EpochLog 列表，长度等于 epochs。
        """
        rng = np.random.default_rng(seed)
        # 只为“每个 epoch 打乱样本顺序”服务的随机数生成器，与权重初始化的随机源分开
        targets_one_hot = one_hot(y, self.layer_sizes[-1])
        # 训练标签一次性转成 one-hot，避免在训练循环里反复转换
        self.history = []
        # 清空旧日志，保证重复调用 fit 时 history 只反映本次训练
        self._adam_m = [np.zeros_like(weight) for weight in self.weights] + [
            np.zeros_like(bias) for bias in self.biases
        ]
        # 重置一阶动量：重新 fit 相当于从头训练，动量不该带着上一轮的惯性
        self._adam_v = [np.zeros_like(weight) for weight in self.weights] + [
            np.zeros_like(bias) for bias in self.biases
        ]
        # 重置二阶动量，理由同上
        self._adam_step = 0
        # 重置 Adam 步数计数器，偏差校正从第 1 步重新开始

        for epoch in range(1, epochs + 1):
            # 外层循环：一个 epoch 就是把训练集按 mini-batch 完整扫一遍
            order = rng.permutation(X.shape[0])
            # 随机打乱样本下标：固定顺序会让梯度下降反复沿同一方向走，收敛更慢也更容易卡住
            for start in range(0, X.shape[0], batch_size):
                # 内层循环：每次取连续 batch_size 个（下标已打乱）样本组成一个 mini-batch
                index = order[start : start + batch_size]
                # 当前批次的样本下标，最后一批可能不足 batch_size
                X_batch = X[index]
                # 该批次的输入，形状 (batch, 64)
                y_batch = targets_one_hot[index]
                # 该批次的 one-hot 标签，形状 (batch, 10)
                activations, pre_activations, probabilities = self.forward(X_batch)
                # 前向传播，拿到所有中间表示和预测概率
                gradients = self._backward(activations, pre_activations, probabilities, y_batch)
                # 反向传播，算出 8 个参数张量的梯度
                self._adam_update(gradients)
                # 用 Adam 规则更新参数，完成一步梯度下降
            _, _, train_probabilities = self.forward(X)
            # 一个 epoch 结束后，在全训练集上算一次指标，作为这条日志的“训练侧”读数
            train_loss = cross_entropy(train_probabilities, targets_one_hot)
            # 训练集平均交叉熵
            train_accuracy = float((train_probabilities.argmax(axis=1) == y).mean())
            # 训练集准确率：取概率最大的类别与真标签比较，布尔数组求均值
            eval_loss = None
            # 默认没有评估指标
            eval_accuracy = None
            # 默认没有评估指标
            if eval_set is not None:
                # 只有传了评估集才额外算一次
                X_eval, y_eval = eval_set
                # 拆出评估集的输入与标签
                _, _, eval_probabilities = self.forward(X_eval)
                # 前向传播（纯推理，不更新任何参数）
                eval_loss = cross_entropy(eval_probabilities, one_hot(y_eval, self.layer_sizes[-1]))
                # 评估集平均交叉熵
                eval_accuracy = float((eval_probabilities.argmax(axis=1) == y_eval).mean())
                # 评估集准确率
            self.history.append(EpochLog(epoch, train_loss, train_accuracy, eval_loss, eval_accuracy))
            # 把这一轮的快照追加进日志
            if verbose and (epoch == 1 or epoch % 25 == 0 or epoch == epochs):
                # 只在第 1 轮、每 25 轮、以及最后一轮打印，避免刷屏
                message = f"    epoch {epoch:>3}/{epochs}  训练损失 {train_loss:.4f}  训练准确率 {train_accuracy:.4f}"
                # 组装基础信息
                if eval_accuracy is not None:
                    # 有评估集时把测试指标也接上
                    message += f"  测试准确率 {eval_accuracy:.4f}"
                    # 追加测试准确率
                print(message, flush=True)
                # 立即输出这一行（flush 保证后台运行时日志能实时看到）
        return self.history
        # 返回完整训练日志，供画训练曲线

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """返回样本属于各类别的概率矩阵。

        参数：
            X：形状 (n, 64) 的输入。
        返回：形状 (n, 10) 的概率矩阵。
        """
        _, _, probabilities = self.forward(X)
        # 前向传播只关心最后的概率输出，前面两项中间结果用下划线丢弃
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

    def representations(self, X: np.ndarray) -> list[np.ndarray]:
        """取出“任意一层输出表示”的接口：一次返回四种表示。

        参数：
            X：形状 (n, 64) 的输入。
        返回：长度 4 的列表 [input, h1, h2, h3]，形状依次为
              (n, 64)、(n, 128)、(n, 64)、(n, 32)。
              第 0 项是原始像素，第 1~3 项是三个隐藏层 ReLU 之后的激活。

        这就是本实验的核心接口：四宫格的四个面板、四个 silhouette 分数、
        四个 1-NN 准确率，全部来自这一个函数返回的四个数组。
        注意 h1/h2/h3 是“ReLU 之后”的值，也就是真正传给下一层的那种表示，
        与 PPT 上“把第 k 层神经元的值画出来”的口径一致。
        """
        activations, _, _ = self.forward(X)
        # 前向传播一次就能同时拿到输入和三层隐藏层的表示，不必分层重算
        return activations
        # 返回 [input, h1, h2, h3]

    def layer_representation(self, X: np.ndarray, layer: int) -> np.ndarray:
        """取出单独一层的表示，方便只想看某一层时调用。

        参数：
            X：形状 (n, 64) 的输入。
            layer：层序号，0 = input，1 = 第 1 隐藏层，2 = 第 2 隐藏层，3 = 第 3 隐藏层。
        返回：形状 (n, layer_sizes[layer]) 的表示矩阵。
        """
        if not 0 <= layer <= self.num_hidden_layers:
            # 层序号超出范围就直接报错，而不是悄悄返回别的层
            raise IndexError(f"layer 应在 0~{self.num_hidden_layers} 之间，收到 {layer}")
        # 显式校验，避免把“取错层”这种 bug 藏起来
        return self.representations(X)[layer]
        # 复用 representations，取第 layer 项返回

    def representation_dimensions(self) -> list[int]:
        """返回四种表示的维度，写进报告用。

        参数：无。
        返回：长度 4 的整数列表，依次是 input/h1/h2/h3 的维度。
        """
        return [self.layer_sizes[0]] + list(self.layer_sizes[1:-1])
        # 输入维度 + 三个隐藏层宽度；输出层是 10 类打分，不算“表示”
