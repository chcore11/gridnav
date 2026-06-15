# GridNav 最终汇总

## 1. 项目背景

GridNav 是一个用二维网格导航任务串联基础算法与机器学习概念的学习项目。项目从可解释的 GridWorld 和 A* 开始，再逐步进入专家数据、传统机器学习、PyTorch 行为克隆、closed-loop rollout、tabular Q-learning 和 C++ 工程实现。

我希望通过同一个导航任务理解不同方法的输入、学习方式、评估指标和边界，而不是把它们当成互不相关的代码示例。

## 2. 阶段路线

| 阶段 | 核心内容 | 状态 |
| --- | --- | --- |
| Stage 1 | GridWorld + Python A* | 已完成 |
| Stage 2 | A* Expert Dataset | 已完成 |
| Stage 3 | Traditional ML baseline | 已完成 |
| Stage 4 | PyTorch Behavior Cloning | 已完成 |
| Stage 4.5 | BC Rollout Evaluation | 已完成 |
| Stage 5 | Tabular Q-learning | 已完成 |
| Stage 6 | C++ A* | 已完成 |
| Stage 7 | Final Summary & Presentation | 最终交付整理中 |

Stage 4.5 是 BC policy 的 closed-loop 评估，不使用 reward，也不训练或更新策略，因此不是强化学习。

## 3. 关键结果总表

| 阶段 / 方法 | 关键结果 |
| --- | --- |
| Python A* | 19 个路径节点，18 个移动步数 |
| Expert Dataset | 尝试 100 张地图，98 张有路径，共 970 个样本 |
| KNN | single-step action accuracy：`0.8144` |
| Logistic Regression | single-step action accuracy：`0.8557` |
| Decision Tree | single-step action accuracy：`0.7732` |
| PyTorch BC | single-step action accuracy：`0.8763` |
| BC Rollout | `13/20 = 65.00%` success rate |
| Tabular Q-learning | 固定 `6 x 6` 地图上 greedy success rate：`100/100 = 100%` |
| C++ A* | 19 个路径节点，18 步，路径合法性检查通过 |

## 4. 各阶段学到的概念

### Stage 1：GridWorld + A*

我理解了地图坐标、障碍物、邻居节点、曼哈顿距离，以及 A* 中 `g(n)`、`h(n)`、`f(n)`、frontier、路径回溯之间的关系。A* 是搜索规划算法，不是机器学习。

### Stage 2：Expert Dataset

我学会了把 A* 最终路径拆成多个 `state -> action` 样本，并理解了 feature、label、expert trajectory 和 state-action pair。A* 的搜索分数本身不是专家数据，最终路径才是数据来源。

### Stage 3：Traditional ML baseline

我使用 KNN、Logistic Regression 和 Decision Tree 学习单步专家动作。这个阶段让我理解了 train/test split、accuracy、classification report、confusion matrix 和 baseline 的作用。

### Stage 4：PyTorch Behavior Cloning

我用 MLP 模仿 A* 专家动作，理解了 Tensor、Dataset、DataLoader、forward、loss、backward 和 optimizer。Behavior Cloning 本质上仍然是监督学习。

### Stage 4.5：BC Rollout Evaluation

我把训练好的 BC policy 放回环境中连续运行，理解了 closed-loop evaluation、错误累积和 distribution shift。`0.8763` 的单步 accuracy 最终只得到 `65.00%` 的 rollout success rate，说明单步预测正确不等于稳定导航。

### Stage 5：Tabular Q-learning

我理解了 state、action、reward、Q-table、epsilon-greedy、exploration 和 exploitation。与 BC 不同，Q-learning 不使用专家 action，而是通过环境反馈更新 `Q(s,a)`。

### Stage 6：C++ A*

我使用 C++17、`vector`、`priority_queue`、`unordered_map` 和 CMake 重新实现 A*，并理解了 Python 原型与 C++ 工程表达之间的对应关系。

## 5. 主要结论

- 同一个导航任务可以用规划、监督学习、模仿学习和强化学习等不同方式处理，但它们的输入、训练信号和评估方法不同。
- Logistic Regression 在传统 baseline 中表现最好，但 PyTorch BC 的单步 accuracy 更高。
- 单步 action accuracy 不能代表完整导航能力，必须通过 rollout 检查连续决策中的错误累积。
- Tabular Q-learning 可以在固定小地图上学到有效策略，但结果高度依赖状态定义和环境范围。
- Python 适合快速学习和验证算法，C++ 实现有助于理解更明确的类型、数据结构和构建流程。

## 6. 局限性

- GridWorld 是二维离散环境，和真实机器人导航差距很大。
- 专家数据的状态只包含当前位置、目标位置和局部障碍，没有完整地图视觉输入。
- BC rollout success rate 只有 `65.00%`，还不能稳定完成随机测试地图导航。
- Tabular Q-learning 只在固定 `6 x 6` 地图上验证，`100%` success rate 不代表随机地图泛化。
- 当前没有连续动作、动态障碍物、传感器噪声、真实物理或机器人控制。
- 当前项目是入门学习项目，不是工业级导航系统。

## 7. 下一步扩展方向

以下内容属于当前项目完成后的扩展方向，不在 Stage 7 文档整理阶段实现：

- CNN 视觉输入
- DQN
- 更复杂地图
- 动态障碍物
- ROS/Gazebo
- 机器人学习 / 具身智能衔接

## 8. 最终交付入口

- `README.md`：项目总览、关键结果、复现顺序和结果边界。
- `reports/engineering_walkthrough.md`：工程目录、脚本输入输出、数据流和读代码顺序。
- `reports/teacher_update.md`：发给张老师前的阶段进展、项目边界和反馈问题。
- `reports/presentation_outline.md`：10 页汇报结构、每页重点和可引用材料。
- `notes/README.md`：各阶段学习笔记索引。

当前 Stage 7 只进行交付整理和汇报准备，不新增 CNN、DQN、动态障碍物或 ROS/Gazebo 等技术功能。
