# 会话交接文档

> 本文档汇总本会话（2026-09 ~ 2026-10）的全部工作成果、文件资产、环境注意事项与待办事项，
> 供后续会话（新对话 / 其他工具）无缝接手。

---

## 一、会话主线

用户在按课件逐页学习三门内容，本会话完成了"讲解 + 实验复现 + 交付物制作"三类工作：

1. **李宏毅机器学习《01 Linear Regression》**——全部讲完（前序会话，本会话沿用其结论）；
2. **李宏毅机器学习《02 Why Deep》**——全部讲完（22 页讲义 + 逐页讲解完成）；
3. **李宏毅机器学习《03 Bias and Variance》**——**22 页全部讲完**，并额外完成了全章配套数值实验、PDF 报告；
4. **《SOA 技术概述》（软件体系结构课程，中南大学，邝砾老师）**——进度到 1.3 节，详见"待办"。

---

## 二、GitHub 仓库（实验代码与数据）

仓库：**https://github.com/8209240223/ml-why-deep-experiments** （public，main 分支）

| 子项目 | 对应课程论点 | 核心结果 |
| --- | --- | --- |
| `checkerboard_experiment/` | 折叠空间 / 表达效率 | 2 万样本时 1 隐层崩溃（49%）、3 隐层保结构（98.5%）；追加对照证明浅网失败是优化效率问题 |
| `handcrafted_vs_learned/` | End-to-end Learning 绿蓝流水线 | 人工特征+逻辑回归 88.7% vs 原始像素+MLP 97.6%；参照组隔离出"人工特征丢信息"是主因 |
| `mnist_layerwise/` | 复杂任务：逐层聚类证据 | 3 隐层 MLP 98.4%；实测轮廓系数 0.163→0.441 逐层上升，四宫格图复刻 PPT |
| `bias_variance_experiment/` | Bias-Variance 全流程 | 打靶图/三色叠图/误差分解三曲线/模型选择对照；恒等式残差 ≤0.59%；公榜法乐观偏差 +77 |

本地路径均在 `F:\kimi\` 下（与仓库同结构）。所有项目：纯 NumPy/Python，逐行中文注释，`python3 main.py` 可直接复跑。

---

## 三、本会话新产生的交付物（未全部入仓）

| 交付物 | 位置 | 说明 |
| --- | --- | --- |
| 《偏差与方差实验报告》PDF | `C:\Users\王仪杰\Desktop\偏差与方差实验报告.pdf` | 7 页，含 4 张结果图 + 22 张公式图（matplotlib mathtext 渲染），专业讲解+通俗理解双轨 |
| PDF 生成脚本 | `F:\kimi\bias_variance_experiment\make_report.py` | reportlab + SimHei，改内容重跑即可（约 2 秒） |
| SOA Java 示例 | `F:\kimi\soa_service_java\` | 对象→构件→服务→接口→客户端五层完整实现，Corretto JDK8 编译运行通过 |
| SOA 结构图（HTML 纯 SVG） | `F:\kimi\soa_service_java\soa_diagram.html` | 复刻 PPT"服务/构件/对象/服务接口"结构图，像素级验证连线精确相接 |
| 结构图截图 | `C:\Users\王仪杰\Desktop\deepseek\soa_diagram.png` | 浏览器截图统一存放位置 |

课件全文提取文本：`F:\kimi\whydeep_slides.txt`、`F:\kimi\biasvariance_slides.txt`、`F:\kimi\soa_slides.txt`。

---

## 四、环境与工具注意事项（踩过的坑）

1. **Python**：`/c/msys64/ucrt64/bin/python`（3.14.5），已装 numpy / matplotlib / scikit-learn / reportlab / pypdf / pillow。MSYS2 是 PEP668 环境，pip 装不了时用 `pacman -S mingw-w64-ucrt-x86_64-python-xxx`。
2. **Java**：机器级 `JAVA_HOME=D:\JDK21`（真实可用）；用户级 JDK8 安装在 `C:\Users\王仪杰\Java\jdk8`，会话级切换脚本 `C:\Users\王仪杰\Java\use-jdk8.bat`（全局未改，避免降级 JDK21）。`C:\Program Files\Common Files\Oracle\Java\javapath` 的 javac 跳板是坏的（PATH 首位，会截胡 `javac` 命令）。
3. **截图**：一律存 `C:\Users\王仪杰\Desktop\deepseek\`；浏览器截图可用 Chrome headless（`--headless --screenshot`）或桌面浏览器工具（被用户接管时不可用）。
4. **PDF 渲染检查**：Poppler 已装（`pdftoppm`），可把 PDF 渲染成 PNG 核查版式。
5. **中文字体坑**：SimHei（黑体）缺 `̂`（组合尖角）、`̄`（组合横线）、`²`（上标2）三个字符，PDF 里会变方框——解决方案：公式一律用 matplotlib mathtext 渲成图片嵌入；正文西文用 Times New Roman；题目要求。
6. **报告版式规范**（用户偏好）：正文宋体、西文 Times、标题黑体、公式 STIX 数学字体；图片居中、图注在图下。

---

## 五、讲解工作流约定（AGENTS.md 摘要）

用户对课件讲解有严格规则（完整版在 `F:\kimi\AGENTS.md`）：

- 固定结构：**原版式汉化 → 手写标注说明 → 本页重点/核心理解 → 图/电路怎么读 → 和前后内容的关系 → 考试重点/易混点**；
- 篇幅：**默认较高详细度**，每节写透；只有说"简要解释"才用短版；
- 代码：**逐行中文注释（注释在下一行的下方）、每段代码后跟"干什么的/和前后的关系"**；
- 公式：禁行内 `$`、禁表格内 LaTeX，独立 `$$` 块；短符号用 Unicode；
- 禁止：Mermaid/ASCII 图（除非明确要求）、"人话解释/生活类比"段、结尾复读式总结表、讲解类任务列表；
- SOA 讲义逐页讲解时：用户发截图（或让我从 PDF 渲染），按上述结构输出。

---

## 六、课件讲解进度明细

### 《03 Bias and Variance》——已完成 22/22 页

标题页、Review、估计量、E[m]=μ、Var[m]=σ²/N、有偏估计量、四象限打靶图、平行宇宙（两页）、100 宇宙 f*、Variance 总结、Bias 定义、5000 宇宙实验、Bias 总结、bias-variance 总图、大 bias 对策、大 variance 对策、模型选择禁忌、public/private 作业页、交叉验证、N 折交叉验证、思考题（神经网络泛化）。

### 《SOA 技术概述》——进度约 60%

- **1.1 需求拉动**：已讲（集成/应变两关键词、企业交互、价值链、供应商案例、三层异构、频繁变化、流程插拔两页、归纳三问题）；
- **1.1 技术推动**：已讲（三齿轮、计算环境四阶段、体系结构三阶段、软件工程四代 SD/OOD/CBSD/SOD、复用·松耦合·互操作三机制）；
- **1.2 什么是 SOA**：已讲（字面拆解、三层定义、印刷术类比、核心三要素）；
- **1.3 构件与连接件**：已讲（构件=服务的内外特性、第 25 页图配套的 Java 代码实现）；
- **未讲**：第 27 页（连接件/XML 协议）、**1.4 典型特征与优势（六组对比，考试重灾区）**、**1.5 体系结构模式（P2P/适配器/远程服务策略/服务集成器/ESB，含 Web Service 三协议 WSDL/UDDI/SOAP）**。

### 《02 Why Deep》——已完成

实验对照、模块化（长发男/CNN 分层/语音 GMM→tied-state→DNN/元音舌位图）、表达效率（普适性定理/奇偶校验/剪窗花/折叠空间/棋盘格）、端到端（语音与图像流水线）、复杂任务（t-SNE 说话人归一化/MNIST 逐层聚类）、结论。

---

## 七、待办事项（下一步建议）

1. **继续 SOA 讲解**：下一站第 27 页（连接件）→ 1.4 节六组对比 → 1.5 节五种模式；
2. **SOA 资产处理**：`soa_service_java/` 和 `soa_diagram.html` 目前**未上传 GitHub**，如需可加入仓库或新建仓库；
3. **SOA 作业**：课程如有编程作业（服务实现/ESB 相关），可基于现有 Java 示例扩展；
4. 其余课程中学员随时可能发新课件截图，按第五节工作流继续。
