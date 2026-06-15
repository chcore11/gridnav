# GridNav Roadmap

## 0. 项目定位

GridNav 是一个学习项目，不是性能项目，也不是用来堆复杂功能的炫技项目。

项目的核心目标是通过一个可控的小型二维 GridWorld 导航任务，把路径规划、传统机器学习、深度学习、强化学习和 C++ 工程基础串联起来。每个阶段都要先理解问题，再完成实现和验证。

每个阶段必须有：

- 代码：实现当前阶段的核心功能。
- 结果：运行代码并保留可以检查的输出、图表或报告。
- 笔记：用自己的话记录概念、实现过程和遇到的问题。
- 解释：能够说明代码为什么这样写，以及结果代表什么。

代码能跑但不能解释，不算完成。

## 1. 总体阶段路线

项目按照以下顺序推进：

1. Stage 0：项目环境与仓库
2. Stage 1：GridWorld + A* 路径规划
3. Stage 2：专家轨迹数据生成
4. Stage 3：传统机器学习 baseline
5. Stage 4：PyTorch 行为克隆
6. Stage 4.5：BC Rollout Evaluation
7. Stage 5：Q-learning 强化学习
8. Stage 6：C++ A* 工程实现
9. Stage 7：最终整理与汇报

不要跳阶段。开始下一阶段前，需要先检查当前阶段的产出和验收标准是否完成，并确认自己能解释核心概念。

---

## Stage 0：项目环境与仓库

### 目标

建立稳定的项目环境，保证后续开发、学习记录和 GitHub 托管能够持续进行。

### 已有内容

- 项目目录：`~/projects/gridnav`
- GitHub 仓库
- Python 虚拟环境 `.venv`
- 基础依赖
- `README.md`
- `notes` 目录
- `figures` 目录

### 验收标准

- `git status` 干净。
- `.venv` 可以激活。
- 基础依赖可以 import。
- `README.md` 和 `AGENTS.md` 存在。
- 项目目录结构清楚，能够说明各目录的用途。

---

## Stage 1：GridWorld + A* 路径规划

### 目标

理解 GridWorld 如何表示地图，以及 A* 如何在已知地图中搜索路径。

### 核心文件

- `env/grid_world.py`
- `planning_astar/astar.py`
- `scripts/run_astar_demo.py`
- `figures/astar_demo.png`
- `notes/01_astar_notes.md`

### 必须掌握的概念

- `(x, y)` 坐标
- `grid[y, x]`
- 起点、终点、障碍物
- `neighbors()`
- 曼哈顿距离
- `f(n)=g(n)+h(n)`
- `frontier`
- `cost_so_far`
- `came_from`
- 路径回溯

### 关键理解

A* 不是机器学习。A* 是搜索规划算法，它根据 `f(n)=g(n)+h(n)` 决定优先搜索哪个节点。A* 最终生成的路径可以作为后续阶段的专家路径。

### 产出

- A* demo 能运行。
- 路径能打印。
- `figures/astar_demo.png` 能生成。
- A* 学习笔记完成。
- 可选：创建 `scripts/debug_astar_trace.py`，用于观察搜索过程。

### 验收标准

能解释：

- GridWorld 如何表示地图。
- A* 为什么能搜索路径。
- `g(n)`、`h(n)`、`f(n)` 分别对应代码里的什么变量。
- 为什么 A* 生成的路径可以作为专家路径。

---

## Stage 2：专家轨迹数据生成

### 目标

把 A* 生成的路径转换成机器学习可以使用的 `state -> action` 数据。

### 核心任务

- 随机生成多张地图。
- 对每张地图运行 A*。
- 得到专家路径。
- 将路径拆成状态-动作样本。
- 保存为 CSV 或 npz 数据集。

### 建议文件

- `data/generate_expert_data.py`
- `data/expert_dataset.csv`
- `reports/dataset_summary.md`
- `figures/expert_path_examples.png`
- `notes/02_expert_data_notes.md`

### 数据字段建议

- `agent_x`
- `agent_y`
- `goal_x`
- `goal_y`
- `obs_up`
- `obs_down`
- `obs_left`
- `obs_right`
- `action`

### 必须掌握的概念

- state
- action
- expert trajectory
- state-action pair
- feature
- label

### 关键理解

A* 的 `f(n)` 不是专家数据。A* 根据 `f(n)` 规则生成最终路径，最终路径再拆成 `state -> action`，这才是专家数据。

### 验收标准

能解释：

- 一条路径如何拆成多个训练样本。
- 为什么当前状态需要包含局部障碍信息。
- action 是如何由相邻坐标差转换出来的。

---

## Stage 3：传统机器学习 baseline

### 目标

先不用神经网络，使用传统机器学习方法学习专家动作，建立后续模型可以比较的 baseline。

### 建议模型

- KNN
- Logistic Regression
- Decision Tree，可选

### 建议文件

- `ml_baseline/train_baselines.py`
- `ml_baseline/evaluate_baselines.py`
- `reports/ml_baseline_result.md`
- `figures/ml_baseline_comparison.png`
- `notes/02_ml_baseline_notes.md`

### 必须掌握的概念

- feature
- label
- train/test split
- classification
- accuracy
- generalization
- baseline

### 评估指标

- 动作准确率
- classification report
- confusion matrix

当前 Stage 3 只验收 single-step action prediction。导航成功率、平均路径长度和与 A* 路径长度的差距属于后续 rollout 扩展，本阶段暂不实现。

### 关键理解

动作准确率高，不代表导航一定成功。单步错误可能让智能体进入训练集中少见的状态，后续错误会继续累积。

### 验收标准

能解释：

- 为什么需要 baseline。
- KNN 和 Logistic Regression 的基本区别。
- 为什么不能只看动作准确率。

---

## Stage 4：PyTorch 行为克隆

### 目标

使用 PyTorch MLP 学习专家的 `state -> action` 映射。

### 建议文件

- `bc_pytorch/dataset.py`
- `bc_pytorch/models.py`
- `bc_pytorch/train_mlp_bc.py`
- `bc_pytorch/evaluate_bc.py`
- `figures/mlp_loss_curve.png`
- `figures/mlp_eval_path.png`
- `reports/bc_result.md`
- `notes/03_pytorch_bc_notes.md`

### 必须掌握的概念

- Tensor
- Dataset
- DataLoader
- `nn.Module`
- `CrossEntropyLoss`
- forward
- loss
- backward
- `optimizer.step`
- behavior cloning
- distribution shift

### 关键理解

行为克隆本质上是监督学习。它使用专家演示数据训练模型，让模型模仿专家动作。它的问题是模型一旦走错，就可能进入专家数据中少见的状态，导致错误累积。

### 验收标准

能解释：

- PyTorch 训练循环。
- `loss.backward()` 做了什么。
- `optimizer.step()` 做了什么。
- 行为克隆为什么属于监督学习。
- 行为克隆为什么可能失败。

---

## Stage 4.5：BC Rollout Evaluation

### 目标

加载 Stage 4 已训练好的 BC policy，在测试地图中进行 closed-loop rollout，评估 single-step action prediction 能否转化为完整导航能力。

### 核心文件

- `rollout/evaluate_rollout.py`
- `reports/rollout_result.md`
- `notes/04_rollout_notes.md`
- `figures/rollout_success_summary.png`

### 评估指标

- success rate
- 成功 episode 的平均模型路径长度
- 相同地图的平均 A* 路径长度
- 成功 episode 的平均路径长度差距
- failure reasons

### 关键边界

- Stage 4.5 是评估，不是训练。
- Stage 4.5 只加载 Stage 4 BC policy，不更新策略。
- Stage 4.5 不使用 reward，也不是强化学习。
- Stage 5 Q-learning 才会通过 reward 与环境交互并学习策略。

### 当前结果

- 测试地图数量：20
- Success rate：`65.00%`
- 失败原因：`loop_detected` 5 次，`hit_obstacle` 2 次。

---

## Stage 5：Q-learning 强化学习

### 当前状态

Stage 5 tabular Q-learning 最小工程闭环已完成。当前实现使用固定 `6 x 6` 地图和坐标 state，通过 reward 学习 Q-table，并在训练结束后使用 greedy policy 评估。

已完成内容：

- `rl_qlearning/train_qlearning.py`
- Q-table、epsilon-greedy、episode 训练循环和 Q-table 更新公式
- `reports/qlearning_result.md`
- `notes/04_qlearning_notes.md`
- `figures/qlearning_training_curve.png`
- `figures/qlearning_success_summary.png`

当前是 tabular Q-learning，不是 DQN。训练不使用专家 action，也不使用 PyTorch。固定地图验证成功只说明 Q-table 学会了当前地图上的策略，不代表具备随机地图泛化能力。

### 目标

不依赖专家路径，让智能体通过 reward 自己来学习导航策略。

### 核心文件

- `rl_qlearning/train_qlearning.py`
- `reports/qlearning_result.md`
- `figures/qlearning_training_curve.png`
- `figures/qlearning_success_summary.png`
- `notes/04_qlearning_notes.md`

### 必须掌握的概念

- state
- action
- reward
- policy
- Q-value
- Q-table
- epsilon-greedy
- exploration
- exploitation

### 当前奖励设计

- 到达目标：`+100`
- 撞墙、越界或撞障碍：`-10`
- 每走一步：`-1`

### 关键理解

行为克隆依赖专家标签；Q-learning 不依赖专家标签，而是通过和环境交互，根据奖励学习 `Q(s,a)`。

### 验收标准

能解释：

- Q-learning 和行为克隆的区别。
- `Q(s,a)` 表示什么。
- 为什么需要 epsilon-greedy。
- reward 曲线说明了什么。

### 当前结果

- 固定地图大小：`6 x 6`
- 训练 episode 数：`3000`
- Alpha：`0.1`
- Gamma：`0.95`
- Epsilon：从 `1.0` 衰减到 `0.05`
- Greedy evaluation success rate：`100/100 = 100%`
- Average steps on success：`10.00`
- 当前边界：未验证随机地图泛化能力

---

## Stage 6：C++ A* 工程实现

### 当前状态

C++ A* 工程实现已提前完成。Stage 5 tabular Q-learning 最小工程闭环完成后，可以继续 Stage 6，学习和理解 C++ A* 的工程表达。

### 目标

用 C++ 重新实现 A*，理解机器人和工程系统中常见的 C++ 路径规划模块。

### 建议文件

- `cpp_astar/CMakeLists.txt`
- `cpp_astar/main.cpp`
- `cpp_astar/astar.h`
- `cpp_astar/astar.cpp`
- `reports/cpp_astar_result.md`
- `notes/05_cpp_astar_notes.md`

### 必须掌握的概念

- struct / class
- vector
- priority_queue
- pair
- CMake
- 编译与运行

### 关键理解

Python 适合快速验证算法，C++ 更接近机器人系统中的工程实现，尤其是路径规划、控制等底层模块。

### 验收标准

能解释：

- C++ 版本 A* 和 Python 版本 A* 的对应关系。
- `priority_queue` 在 A* 中的作用。
- CMake 编译流程。

### 当前结果

- CMake 配置、编译和运行成功。
- C++ demo 与 Python demo 使用相同地图。
- 输出 19 个路径节点和 18 个移动步数。
- 起点、终点、相邻移动、边界和障碍物合法性检查全部通过。

---

## Stage 7：最终整理与汇报

### 目标

把 GridNav 整理成可以展示给老师看的完整学习项目。

### 最终交付物

- 完整 `README.md`
- 清晰的 GitHub 仓库结构
- 每个阶段的 notes
- 每个阶段的 figures
- 实验结果报告
- 最终汇报 PDF 或 PPT

### 汇报结构建议

1. 项目背景：为什么做 GridNav
2. 项目目标：用一个导航任务串起基础知识
3. GridWorld 环境
4. A* 路径规划
5. 专家轨迹数据
6. 传统机器学习 baseline
7. PyTorch 行为克隆
8. Stage 4.5：BC Rollout Evaluation
9. Q-learning
10. C++ A*
11. 总结与下一步计划

### 下一步扩展方向

- CNN 视觉输入
- DQN
- 更复杂地图
- 动态障碍物
- ROS/Gazebo 仿真
- 机器人学习 / 具身智能方向衔接

---

## 日常推进规则

每次开始任务前：

1. 先确认当前阶段。
2. 查看对应阶段的目标和验收标准。
3. 不要跳阶段。

每次完成任务后：

1. 运行相关脚本。
2. 检查结果文件。
3. 更新 notes。
4. 检查 `git diff`。
5. 再决定是否 commit。

## Codex 使用规则

后续 Codex 应同时根据 `AGENTS.md` 和 `ROADMAP.md` 工作。

Codex 在执行任务时应：

- 先判断当前任务属于哪个 Stage。
- 只修改任务相关文件。
- 不自动安装依赖。
- 不自动执行 `git commit` 或 `git push`。
- 修改完成后说明改动内容和建议运行命令。
- 如果任务可能跨阶段，应先提醒，而不是直接执行。

## 当前推荐下一步

当前进度：

- Stage 1：GridWorld + A* 路径规划，已完成。
- Stage 2：随机地图专家轨迹数据生成，已完成。
- Stage 3：传统机器学习 baseline 已完成工程与文档收尾验收。
- Stage 4：PyTorch Behavior Cloning 已完成。
- Stage 4.5：BC Rollout Evaluation 已完成。
- Stage 5：Tabular Q-learning 最小工程闭环已完成。
- Stage 6：C++ A* 工程实现已提前完成，学习理解可在 Stage 5 完成后继续。

Stage 4.5 是 BC Rollout Evaluation，只评估 Stage 4 已训练的 BC policy，不使用 reward，也不训练或更新策略，因此不是强化学习。

Stage 5 已在固定 `6 x 6` 地图上完成 tabular Q-learning 训练与 greedy policy 评估，不使用专家 action、PyTorch 或 DQN。Greedy evaluation success rate 为 `100/100`，平均成功步数为 `10.00`。当前结果不代表具备随机地图泛化能力。

Stage 6 已使用 C++17 重新实现 A*，并在与 Python demo 相同的地图上完成构建、运行和路径合法性验证。Stage 5 最小工程闭环完成后，可以继续 Stage 6 的学习理解。
