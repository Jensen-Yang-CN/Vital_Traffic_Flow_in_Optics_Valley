# 光谷路网微循环韧性评估与优化

> **Vital Traffic Flow in Optics Valley** · 基于时空图神经网络（ST-GNN / T-GCN）的武汉光谷广场核心区路网韧性评估与交通调控决策研究

[![Python](https://img.shields.io/badge/Python-3.9-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.x-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![PyTorch Lightning](https://img.shields.io/badge/PyTorch%20Lightning-1.x-792EE5?logo=lightning&logoColor=white)](https://lightning.ai/)
[![Model](https://img.shields.io/badge/Model-T--GCN-2E7D32)](#核心方法)
[![Data](https://img.shields.io/badge/Data-高德交通态势%20API-1478FF)](#数据说明)

---

## 📖 项目简介

本项目以**武汉市光谷广场核心区**（民族大道、珞喻路、鲁磨路、虎泉街）为研究对象，将**时空图神经网络**引入城市路网的**韧性（Resilience）**评估问题：

> 当核心路段遭遇突发冲击（交通事故、信号故障、瞬时过饱和）时，拥堵会沿着路网拓扑"传染"到哪些路段？整个微循环系统需要多久才能恢复？政府应当在哪里、以多大强度介入才能把损失压到最低？

围绕这三个问题，项目完成了从**数据采集 → 拓扑建模 → 时空预测 → 级联失效推演 → 韧性量化 → 调控策略寻优**的全链路实验，并产出可用于竞赛论文与决策建议的可视化结论。

本仓库是参赛作品 **TJJM20260520005696（研究生组）** 的代码、数据与文档归档，作品标题为《**数智赋能激发光谷新活力：基于 ST-GNN 与多源数据的路网微循环韧性评估与优化路径**》。

### 与传统交通预测的区别

| | 传统时序方法（ARIMA / LSTM） | 本项目（ST-GNN） |
| --- | --- | --- |
| 空间建模 | ❌ 各路段**孤立预测**，无法感知相邻路段影响 | ✅ 通过**邻接矩阵**显式建模拥堵的**空间溢出/传染** |
| 输出用途 | 单点车速预测 | ✅ 预测 + **级联失效推演** + 韧性量化 + 策略寻优 |
| 数据形态 | 规则欧氏网格 | ✅ **非欧几里得**的路网拓扑图 |

---

## 🗺️ 技术路线

```
 ① 数据采集            ② 数据清洗              ③ 拓扑构建
┌──────────────┐      ┌──────────────┐       ┌────────────────────┐
│ 高德交通态势  │      │ traffic_     │       │ build_graph_v3.py  │
│ API 定时抓取  │─────▶│ filter.py    │──────▶│ 有向图 → 12×12     │
│ (每 120 秒)   │      │ hebing.py    │       │ 邻接矩阵 A         │
└──────────────┘      └──────────────┘       └────────────────────┘
                                                       │
 ④ 特征工程            ⑤ 模型训练               │
┌──────────────────────┐  ┌──────────────────┐        │
│ reTrain.py           │  │ main.py          │        │
│ 长表 → 宽表透视       │─▶│ GCN / GRU / TGCN │◀───────┘
│ 节点顺序强制对齐      │  │ PyTorch Lightning│
│ 缺失值 ffill/bfill   │  │ 300 轮迭代       │
│ → X ∈ R^(T×N)        │  └──────────────────┘
└──────────────────────┘           │
                                   ▼
 ⑥ 韧性推演与决策赋能
┌────────────────────────┬────────────────────────┬────────────────────────┐
│ resilience_index_calc  │ simulate_cascade_v9    │ policy_optimize(1)     │
│ 韧性三角 + 指数 R       │ 级联失效空间蔓延推演    │ 干预强度 α 边际效益寻优 │
├────────────────────────┼────────────────────────┼────────────────────────┤
│ rank_resilience        │ policy_simulation      │ 三张论文引用图 + 曲线   │
│ 各节点抗冲击排名        │ 数治 vs 常规模式对比    │                        │
└────────────────────────┴────────────────────────┴────────────────────────┘
```

---

## 📂 目录结构

```
Vital_Traffic_Flow_in_Optics_Valley/
│
├── 🐍 数据采集与清洗
│   ├── traffic_collector1.py        # 高德「交通态势矩形区域」API 定时采集（带限流与重试）
│   ├── traffic_filter.py            # 两轮筛选：目标道路 + 合法行驶方向 → *_cleaned.csv
│   ├── hebing.py                    # 多日清洗结果纵向合并（智能编码嗅探）
│   └── washData.py                  # 【legacy】透视成车速矩阵（16 节点口径，已被 reTrain.py 取代）
│
├── 🕸️ 路网拓扑构建
│   ├── build_graph_v2.py            # 邻接矩阵构建 v2（13 节点版本）
│   ├── build_graph_v3.py            # 邻接矩阵构建 v3（12 节点，✅ 当前采用版本）
│   ├── view_npy.py                  # .npy 邻接矩阵 → 可读 CSV（方便 Excel 核对）
│   ├── guanggu_adj.npy              # ⭐ 12×12 int32 邻接矩阵（模型的空间骨架）
│   └── guanggu_node_order.txt       # ⭐ 12 个节点的「法定顺序」，保证 X 与 A 严格对齐
│
├── 🧠 模型训练与特征工程
│   ├── reTrain.py                   # 长表 → 时空特征矩阵 data/guanggu_speed.csv（供 T-GCN 读取）
│   └── main.py                      # 训练入口（PyTorch Lightning，支持 GCN / GRU / TGCN 切换）
│
├── 📉 韧性推演与策略优化
│   ├── resilience_index_calc.py     # 韧性三角图 + 早/晚高峰韧性指数 R
│   ├── simulate_cascade_v9.py       # 级联失效推演（Node 8 事故 → 邻居受损）
│   ├── rank_resilience.py           # 12 个节点的抗冲击韧性排名
│   ├── policy_optimize(1).py        # 主动管控强度 α 的边际效益寻优（拐点 α=0.6）
│   └── policy_simulation.py         # 「数治」主动干预 vs 常规模式效果对比
│
├── 📊 data/                         # 清洗后的光谷路况数据（7 个 CSV）
├── 🖼️ node_*.png                    # 论文引用的预测效果对比图（3 张）
├── 📚 项目文档                      # 技术报告、方法论、公式推导、参赛全文（含 PDF / DOCX）
├── requirements.txt
└── .gitignore
```

---

## 📊 数据说明

### 数据来源

| 来源 | 内容 | 状态 |
| --- | --- | --- |
| **高德开放平台**「交通态势 - 矩形区域」API | 光谷广场核心区实时路况，每 **120 秒**采集一次，含车速与拥堵状态 | ✅ 已采集（见 `data/`） |
| **滴滴盖亚开放数据** 成都轨迹数据集 | 28.1 GB 脱敏轨迹，作为模型预训练"题库" | ⚠️ 计划中，未入库（体积过大） |
| 高德原始日文件 `guanggu_traffic.csv` 等 | 未清洗的原始采集结果 | ⚠️ 未入库 |

> **说明**：`traffic_collector1.py` 的采集范围参数 `RECTANGLE = "114.397,30.504;114.401,30.508"` 即光谷广场核心区矩形框，`ROAD_LEVEL = "6"` 表示只取主干道及以上等级道路。

### 数据字段（`data/*_cleaned.csv`）

| 字段 | 含义 |
| --- | --- |
| `采集时间` | 采集时刻（时间步长 5 分钟级） |
| `整体路况代码` / `整体路况描述` | 区域整体拥堵等级（畅通 / 缓行 / 拥堵） |
| `平均车速(km/h)` | 区域平均车速 |
| `畅通率(%)` / `拥堵率(%)` / `阻塞率(%)` | 区域路况构成比 |
| `道路名称` | 民族大道 / 珞喻路 / 鲁磨路 / 虎泉街 |
| `道路状态` | 该路段拥堵状态 |
| **`速度(km/h)`** | ⭐ **模型的核心目标变量** |
| **`行驶方向`** | ⭐ 与道路名称拼接构成**有向路段节点 ID** |
| `方向角度` | 道路走向角度 |

### 已入库的数据文件

| 文件 | 记录数 | 编码 | 说明 |
| --- | --- | --- | --- |
| `data/guanggu_traffic_4_24_cleaned.csv` | 544 | UTF-8 | 4 月 24 日清洗结果 |
| `data/guanggu_traffic_4_25_cleaned.csv` | 1326 | **GBK** | 4 月 25 日 |
| `data/guanggu_traffic_4_26_cleaned.csv` | 717 | UTF-8 | 4 月 26 日 |
| `data/guanggu_traffic_4_27_cleaned.csv` | 1488 | UTF-8 | 4 月 27 日 |
| `data/guanggu_traffic_0427_cleaned.csv` | 705 | UTF-8 | 4 月 27 日另一批次 |
| `data/guanggu_traffic_ALL_combined.csv` | 2587 | **GBK** | 合并集 v1 |
| `data/guanggu_traffic_ALL_combined_V2.csv` | 4780 | **GBK** | 合并集 v2 |

> ⚠️ **编码不统一是这份数据的真实特征**（部分 CSV 为 GBK，部分为 UTF-8）。这也是 `reTrain.py` / `hebing.py` 中 `smart_read()` 遍历多种编码做"编码嗅探"的原因 —— 直接 `pd.read_csv` 会抛 `UnicodeDecodeError`。

### 拓扑与特征文件

| 文件 | 规格 | 说明 |
| --- | --- | --- |
| `guanggu_adj.npy` | **12 × 12**，`int32` | 有向图邻接矩阵 $A$，元素为 0/1 |
| `guanggu_node_order.txt` | 12 行 | 节点顺序"户口本"，与 $A$ 的行列索引一一对应 |

**12 个有向路段节点：**

```
 0  虎泉街_从民族大道到卓刀泉南路        6  珞喻路_从喻家湖路到珞狮路
 1  虎泉街_从雄楚大道到民族大道          7  民族大道_从汤逊湖北路到珞喻路
 2  虎泉街_从卓刀泉南路到民族大道        8  民族大道_从雄楚大道到珞喻路   ← 推演的"事故冲击源"
 3  鲁磨路_从团山路到珞喻路              9  民族大道_从珞喻路到南湖大道
 4  鲁磨路_从珞喻路到喻家山北路         10  民族大道_从珞喻路到大学园路
 5  珞喻路_从珞狮路到喻家湖路           11  民族大道_从珞喻路到雄楚大道
```

> 📌 **节点数演进**：项目文档早期规划为 **16 个节点**，`build_graph_v2.py` 为 **13 个**，`build_graph_v3.py` 定为 **12 个**并沿用至今。**当前入库的 `guanggu_adj.npy`（12×12）与 `guanggu_node_order.txt`（12 行）均为 v3 口径**，实验脚本中的 `N_NODES = 12` 也与此一致。

---

## 🚀 快速开始

### 环境准备

```bash
pip install -r requirements.txt
```

> ⚠️ **PyTorch Lightning 必须用 1.x**。`main.py` 使用了 `pl.Trainer.from_argparse_args()` 和 `pl.callbacks.ModelCheckpoint(monitor=...)`，装 2.x 会直接抛 `AttributeError`。

图表中文标题依赖 **SimHei** 字体（Windows 自带；Linux 需自行安装并调整 `plt.rcParams['font.sans-serif']`）。

### 配置高德 API Key（采集脚本必需）

出于安全考虑，**源码中不包含任何密钥**。`traffic_collector1.py` 会从环境变量 `AMAP_API_KEY` 读取 Key，未配置时会直接退出并给出提示。请先到[高德开放平台](https://lbs.amap.com/)申请 **Web 服务**类型的 Key，然后设置环境变量：

```powershell
# PowerShell（仅当前终端窗口有效）
$env:AMAP_API_KEY="你的Key"

# PowerShell（永久写入用户环境变量，需重开终端）
[Environment]::SetEnvironmentVariable("AMAP_API_KEY","你的Key","User")
```

```bash
# CMD
set AMAP_API_KEY=你的Key

# Bash / zsh
export AMAP_API_KEY="你的Key"
```

也可以写入项目根目录的 `.env` 文件（`.env` 已在 `.gitignore` 中，不会被提交）。

### 复现实验的完整流程

```bash
# ── 步骤 1：采集（需要有效的高德 API Key，且需长时间挂机累积数据）──
python traffic_collector1.py

# ── 步骤 2：清洗与合并 ──
python traffic_filter.py data/guanggu_traffic_0430.csv     # 逐日清洗 → *_cleaned.csv
python hebing.py                                          # 多日纵向合并

# ── 步骤 3：构建路网拓扑（生成 12×12 邻接矩阵与节点顺序）──
python build_graph_v3.py
python view_npy.py                                        # 可选：导出 CSV 人工核对

# ── 步骤 4：生成时空特征矩阵 data/guanggu_speed.csv ──
python reTrain.py

# ── 步骤 5：训练 T-GCN ──
python main.py --data guanggu --model_name TGCN --max_epochs 300

# ── 步骤 6：韧性推演与策略优化（生成论文图表）──
python resilience_index_calc.py    # 韧性三角图 + 早晚高峰韧性指数
python simulate_cascade_v9.py      # 级联失效空间蔓延对比
python rank_resilience.py          # 12 节点韧性排名
python "policy_optimize(1).py"     # 干预强度 α 边际效益寻优
python policy_simulation.py        # 数治模式 vs 常规模式
```

### ⚠️ 运行前必读：本仓库无法"开箱即跑"

这不是一份 self-contained 的可复现仓库，以下依赖项**未随代码入库**，直接执行会报错：

| 缺失项 | 影响 | 解决方式 |
| --- | --- | --- |
| `models.py`、`tasks.py`、`utils/` 包 | `main.py` 与全部推演脚本 `import models / tasks` **必然失败** | 需从 T-GCN 参考实现补入（见[团队与致谢](#团队与致谢)） |
| `lightning_logs/TGCN/version_10/checkpoints/epoch=247-step=21080.ckpt` | 5 个推演脚本硬编码加载该权重 | 需先自行训练生成，或修改脚本中的 `CHECKPOINT_PATH` |
| `data/guanggu_speed.csv`、`data/guanggu_adj.csv` | T-GCN 读取的特征矩阵与邻接 CSV | ✅ 执行 `python reTrain.py` 自动生成 |
| `guanggu_traffic.csv`（原始未清洗文件） | `washData.py` 直接读取该文件名 | 该脚本为 legacy，建议直接用 `traffic_filter.py` + `reTrain.py` |
| 4/28、4/29、4/30 的 `*_cleaned.csv` | `hebing.py` 列出的 7 个文件中有 3 个不存在 | 需补齐采集数据，或修改 `hebing.py` 中的 `files` 列表 |

---

## 🧩 模块说明

### 数据采集与清洗

| 脚本 | 作用 | 输入 | 输出 |
| --- | --- | --- | --- |
| `traffic_collector1.py` | 调用高德「交通态势/矩形区域」接口，按 120 秒间隔轮询；内置 `RateLimiter`（2 QPS）与指数退避重试（最多 3 次） | — | `guanggu_traffic_0430.csv` |
| `traffic_filter.py` | 两轮筛选：① 只保留 4 条目标道路；② 按 `direction_rules` 白名单过滤合法行驶方向 | 原始日 CSV | `*_cleaned.csv`（`utf-8-sig`） |
| `hebing.py` | 通过 `smart_read_csv()` 遍历 `utf-8-sig / utf-8 / gbk / ansi` 嗅探编码后纵向合并 | 多日 cleaned CSV | `guanggu_traffic_ALL_combined_V3_latest.csv` |

### 拓扑构建

| 脚本 | 作用 |
| --- | --- |
| `build_graph_v2.py` | 13 节点版本，采用"流入转盘"与"珞喻路横穿"两条规则建边 |
| **`build_graph_v3.py`** | **当前采用版本**。定义 12 个有向路段，用正则 `从(.*)到(.*)` 解析起终点，**当节点 $i$ 的终点 == 节点 $j$ 的起点**时置 $A_{ij}=1$，并额外补上珞喻路内部的横穿连通 |
| `view_npy.py` | 把 `.npy` 二进制矩阵导出为无表头 CSV，方便用 Excel 人工核对连通关系 |

### 模型与特征工程

| 脚本 | 作用 |
| --- | --- |
| `reTrain.py` | 读合并数据 → 拼接 `路段ID` → `pivot_table` 透视成 `行=采集时间, 列=路段ID, 值=速度` → **按 `guanggu_node_order.txt` 强制 `reindex` 列顺序** → `ffill().bfill()` 填补缺失 → 导出无表头纯数值 CSV |
| `main.py` | 训练入口。基于 **PyTorch Lightning** 解耦为 DataModule / Task / Model 三层；`--model_name` 支持 `GCN` / `GRU` / `TGCN`，通过 `getattr` 反射动态注册各模型专属超参；内置 `ModelCheckpoint` 与训练失败邮件告警 |

### 韧性推演与策略优化

| 脚本 | 作用 | 关键设置 |
| --- | --- | --- |
| `resilience_index_calc.py` | 生成**韧性三角图**并计算早/晚高峰韧性指数 R | `start_idx=147`（早高峰）、`1003`（晚高峰），对比节点 `node=5` |
| `simulate_cascade_v9.py` | **级联失效推演**：自回归滚动预测未来 6 步，对比"直接关联节点(5)"与"拓扑隔离节点(0)"的效率损失 | 冲击源 `node=8`，归一化速度置 `0.05` |
| `rank_resilience.py` | 计算 12 个节点各自的韧性指数 $R=\sum V_{actual}/\sum V_{ideal}$ 并横向排名 | 基准/受冲击两次仿真，`SIM_START_IDX=752` |
| `policy_optimize(1).py` | 扫描主动管控强度 $\alpha \in [0,1]$（步长 0.1），寻找路网整体效率提升的**边际效益拐点**（结论：$\alpha=0.6$） | 冲击源 `node=8`，控制节点 `node=7` |
| `policy_simulation.py` | 「常规模式（持续拥堵）」vs「数治模式（主动干预）」效果对比 | 对节点 `[2,3,5,7]` 施加 1.5 倍速度增益，观测 `node=9` |

---

## 🔬 核心方法

### 1. 有向图与邻接矩阵的构建逻辑

交通网络是**有向图**：车辆不能逆行，因此"民族大道（南向北）"与"民族大道（北向南）"必须被定义为**两个完全不同的节点**。节点 ID 由 `道路名称 + 行驶方向` 拼接而成，从而把物理道路降维为图论意义上的独立节点。

有向边的判定规则很简单：**如果车流能合法地从节点 $A$ 的出口直接驶入节点 $B$ 的入口，则 $A_{ij}=1$，否则为 $0$。**

$$
A_{ij} =
\begin{cases}
1, & \text{节点 } i \text{ 的终点} = \text{节点 } j \text{ 的起点（或珞喻路内部横穿连通）} \\
0, & \text{其他}
\end{cases}
$$

这张 0/1 矩阵是"空间传染"的物理介质：当珞喻路车速暴降，拥堵权重会沿 $A_{ij}=1$ 的边**倒灌**进民族大道，从而让模型具备"牵一发而动全身"的推演能力。

### 2. 韧性指数（Resilience Index）

采用**离散时间序列版**的韧性三角定义：

$$
R = \frac{\sum_{t=t_0}^{t_e} V_{actual}(t)}{\sum_{t=t_0}^{t_e} V_{ideal}(t)}
$$

- $V_{ideal}(t)$：**无事故**情况下模型预测的正常车速（图中绿色虚线，理想性能基准）
- $V_{actual}(t)$：**发生事故后**模型预测的受损与恢复曲线（图中红色实线）
- $R \to 1.0$ 表示路网拓扑冗余度高、抗冲击与自愈能力强

引入**性能损失面积 PLA（Performance Loss Area）** 后可等价表述为"1 减去损失占比"：

$$
PLA = \sum_{t=t_0}^{t_e}\left[V_{ideal}(t) - V_{actual}(t)\right]\cdot \Delta t
\qquad\Longrightarrow\qquad
R = 1 - \frac{PLA}{TA_{ideal}}
$$

> 完整推导见 [`韧性公式.md`](韧性公式.md)。

### 3. 级联失效的模拟方式

无需额外的仿真软件，项目巧妙地**用预测模型本身做"数字孪生"**：

1. 取一段真实历史车速（如 `start_idx=752` 起 18 个时间步）做归一化，作为初始输入窗口；
2. 人为把冲击源节点（Node 8）的速度**钳制到极低值**（`0.05`），模拟交通事故导致的车速崩塌；
3. 让 T-GCN **自回归滚动预测**未来 12 步（每步 5 分钟，共 1 小时），每一步都把上一个预测结果拼回输入窗口；
4. 同时跑一条"不注入事故"的基准曲线，两条曲线的差异即**冲击的空间蔓延路径与恢复过程**。

### 4. 一个关键工程细节：节点顺序强制对齐

特征矩阵 $X$（各路段的实际车速，来自《高德 API》）与邻接矩阵 $A$（路的连通关系，来自 `build_graph_v3.py`）是**分开存储**的，在 GCN 内部才做矩阵乘法。

> **风险**：如果 $X$ 的第 1 列是"民族大道"而 $A$ 的第 1 行代表"珞喻路"，GCN 就会把珞喻路的拥堵特征错误地传递给民族大道的邻居 —— 结果看似正常，实则全错。

因此项目引入 `guanggu_node_order.txt` 作为**"法定基准"**，用 `speed_matrix.reindex(columns=node_order)` 强制让 $X$ 的列顺序与 $A$ 严格一致。这是整条流水线中最容易被忽略、但一旦出错就全盘失效的一环。

---

## 📈 实验结果

### 模型预测精度

项目文档中记录了两组指标口径，**请注意它们来自不同的实验批次**，撰写论文时需先统一口径：

**口径 A —— 光谷核心区实验（4 月 24–26 日数据，300 轮迭代）**，出自 [`基于 T-GCN 的武汉光谷核心区路网时空预测效能分析与实验验证.md`](基于%20T-GCN%20的武汉光谷核心区路网时空预测效能分析与实验验证.md)：

| 指标 | 数值 |
| --- | --- |
| 预测准确率 Accuracy | **89.6%** |
| 决定系数 $R^2$ | **0.841** |
| 平均绝对误差 MAE | **3.19 km/h** |

**口径 B —— 指标说明文档**，出自 [`对比.md`](对比.md)（其中提到"156 条路"，对应 T-GCN 公开的 Shenzhen 数据集规模）：

| 指标 | 数值 |
| --- | --- |
| RMSE | 5.74 → **4.09** |
| MAE | 4.25 → **2.77** |
| Accuracy | **0.714** |
| $R^2$ / ExplainedVar | **0.846** |

### 三个典型路段的定性发现

仓库中的三张预测对比图对应三种截然不同的模型行为，是论文分析的核心素材：

| 图 | 路段 | 现象 | 结论 |
| --- | --- | --- | --- |
| `node_9_民族大道_从珞喻路到南湖大道.png` | Node 9（转盘南向流出"瓶颈"） | 预测曲线与实测曲线在状态切换点（样本 31 缓行→拥堵、样本 66 拥堵消散）**几乎零延迟同步** | 证明 GCN 成功捕获了转盘上游流量的**空间冲击** |
| `node_3_鲁磨路_从珞喻路到喻家山北路.png` | Node 3（噪声最强路段） | 实测车速在 25/30 km/h 间高频抖动，预测曲线**连续平滑** | 空间邻里校验起到了**低通滤波**作用，模型具备抗噪鲁棒性，还原的是交通流的物理中轴线 |
| `node_10_民族大道_从珞喻路到大学园路.png` | Node 10（大学园方向） | 波形复刻近乎完美，但数值**整体上偏约 5 km/h** | 典型的**系统性偏差**，源于训练集与验证集的分布差异，可通过残差修正模块/局部偏置层校准 |

> 值得强调的方法论观点：**预测"趋势"的意义往往大于预测"绝对数值"** —— 能提前预判车速下降趋势，就足以支撑前馈式信号配时决策。

### 决策推演结论

- **级联影响不对称**：事故冲击并非均匀扩散。与冲击源**直接拓扑关联**的路段（Node 5 珞喻路）出现显著效率损失，而**拓扑隔离**的路段（Node 0 虎泉街）受影响轻微 —— 这直接验证了邻接矩阵建模的有效性。
- **调控存在拐点**：主动管控强度 $\alpha$ 与路网整体效率提升之间是**非线性**关系，$\alpha=0.6$ 处为边际效益拐点 —— 过度干预反而不再带来收益，为"精准适度"的信号配时补偿提供了量化依据。

---

## ⚠️ 已知问题与限制

**工程完整性**

- [ ] `models.py` / `tasks.py` / `utils/` 缺失，**训练与推演脚本目前均无法直接运行**
- [ ] 训练权重 `.ckpt` 未入库，5 个推演脚本的 `CHECKPOINT_PATH` 硬编码为 `lightning_logs/TGCN/version_10/...`
- [ ] `policy_simulation.py` 的 `savefig` 指向绝对路径 `G:\WorkSpace-Yangjunjie\...`，在他人机器上会直接报错，且缺少本地回退路径
- [ ] `washData.py` 依赖不存在的 `guanggu_traffic.csv`；`hebing.py` 依赖缺失的 4/28–4/30 数据
- [ ] 缺少 `LICENSE` 文件

**科学严谨性（论文投稿前建议核对）**

- [ ] **节点数口径不统一**：文档写 16 个、`build_graph_v2.py` 为 13 个、`build_graph_v3.py` 为 12 个，需在论文中统一说明演进过程与最终口径
- [ ] **精度指标口径不统一**：Accuracy 出现 89.6% 与 0.714 两个值，MAE 出现 3.19 与 2.77，需确认各自对应的数据集与实验批次
- [ ] `韧性公式.md` 中"早高峰 R=0.8888"是**论文撰写建议里的示例表述**，并非本仓库脚本的实测输出，引用前请以实际运行结果为准
- [ ] 级联推演中把受影响节点速度**人为钳制为 0.05** 属于强假设，需在论文的局限性部分明确讨论
- [ ] 滴滴轨迹数据（28.1 GB）尚未接入，当前模型实际仅使用了高德 API 采集的光谷数据

**数据与安全**

- [x] `traffic_collector1.py` 中的高德 API Key 已改为**从环境变量 `AMAP_API_KEY` 读取**，源码不再包含任何密钥（配置方法见[上文](#配置高德-api-key采集脚本必需)）
- [ ] 原始采集文件与 28.1 GB 滴滴数据未入库，无法完整复现数据采集环节
- [ ] `data/` 中 CSV 编码不统一（UTF-8 与 GBK 混用），下游脚本必须依赖编码嗅探

---

## 👥 团队与致谢

- **参赛作品**：TJJM20260520005696（研究生组）
- **作品标题**：《数智赋能激发光谷新活力：基于 ST-GNN 与多源数据的路网微循环韧性评估与优化路径》
- **核心贡献者**：Jensen-Yang-CN、Guiyue、Hongyu 等

**参考与致谢**

- **T-GCN（Temporal Graph Convolutional Network）**：`main.py` 的三层解耦架构（DataModule / Task / Model）、`GCN`/`GRU`/`TGCN` 模型选择机制与 PyTorch Lightning 工程范式，源自 T-GCN 的开源参考实现。理论出处：
  > Zhao, L., Song, Y., Zhang, C., Liu, Y., Wang, P., Lin, T., Deng, M., & Li, H. (2020). *T-GCN: A Temporal Graph Convolutional Network for Traffic Prediction.* IEEE Transactions on Intelligent Transportation Systems, 21(9), 3848–3858.
- **高德开放平台**：提供「交通态势 / 矩形区域」路况数据接口
- **滴滴盖亚开放数据计划**：提供脱敏轨迹数据集（本项目计划用于模型预训练）

---

## 📚 文档索引

仓库根目录的 Markdown / PDF / DOCX 为项目研究过程文档：

| 文档 | 内容 |
| --- | --- |
| [`数智赋能激发光谷新活力...md`](数智赋能激发光谷新活力：基于ST-GNN与多源数据的路网微循环韧性评估与优化路径.md) | 项目总纲：三阶段目标、任务分解与里程碑 |
| [`基于 T-GCN 的武汉光谷核心区路网时空预测效能分析与实验验证.md`](基于%20T-GCN%20的武汉光谷核心区路网时空预测效能分析与实验验证.md) | 技术报告：实验综述、三段路案例分析、源码架构解析、算法原理 |
| [`核心文档一：核心概念文档：路网拓扑与邻接矩阵的构建逻辑.md`](核心文档一：核心概念文档：路网拓扑与邻接矩阵的构建逻辑.md) | 邻接矩阵的 What / Why / How 完整推导 |
| [`韧性公式.md`](韧性公式.md) / [`韧性公式.pdf`](韧性公式.pdf) | 韧性指数 R 与性能损失面积 PLA 的公式推导 |
| [`多路段对比&绿灯占比优化策略.md`](多路段对比&绿灯占比优化策略.md) | 多路段横向对比与信号配时优化策略 |
| [`4.28矩阵相关公式以及特征提取.md`](4.28矩阵相关公式以及特征提取.md) | 矩阵运算公式与特征提取方法 |
| [`模型搭建.md`](模型搭建.md) | 模型结构搭建记录 |
| [`对比.md`](对比.md) | RMSE / MAE / Accuracy / R² / Loss 等指标含义与数值解读 |
| [`正则表达式.md`](正则表达式.md) | 行驶方向文本解析所用的正则说明 |

---

## 📄 许可协议

本仓库当前**未包含 `LICENSE` 文件**，默认保留全部权利。
如涉及竞赛作品的知识产权归属，请以赛事主办方规定与团队内部约定为准。

---

<p align="center">
  <b>让每一条路的拥堵，都能被提前看见。</b><br/>
  <sub>基于 ST-GNN 的光谷路网微循环韧性研究</sub>
</p>
