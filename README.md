# GridNav

GridNav 是一个小型 GridWorld 导航学习项目。

这个项目不是为了快速做出一个复杂系统，而是用一个尽量清晰、可控的网格导航问题，逐步理解 A* 规划、机器学习 baseline、PyTorch 行为克隆、Q-learning 强化学习，以及 C++ 版本 A* 的底层逻辑。

## 项目目标

建立一个简单的 GridWorld 导航任务，并围绕同一个任务比较不同方法的思路：

- A* 如何根据代价和启发式搜索路径。
- 如何把 A* 生成的路径转成专家轨迹数据。
- 传统机器学习 baseline 如何从特征和标签中学习动作。
- PyTorch 行为克隆如何用神经网络模仿专家策略。
- BC Rollout Evaluation 如何检查单步动作预测能否转化为完整导航能力。
- Q-learning 如何通过与环境交互学习动作价值。
- C++ A* 如何把算法落实到更接近工程实现的层面。

当前已完成 GridWorld、Python A*、专家数据生成、传统机器学习 baseline、PyTorch 行为克隆和 Stage 4.5 BC Rollout Evaluation。Stage 5 Q-learning 正在理解中，工程尚未开始。Stage 6 C++ A* 工程实现已提前完成，但当前学习主线暂不进入 Stage 6。

## 推进路线

1. 实现基础 GridWorld 环境。
2. 用 Python 实现 A* 规划。
3. 用 A* 生成专家轨迹数据。
4. 构建机器学习 baseline，学习从状态特征预测动作。
5. 用 PyTorch 实现行为克隆模型和训练流程。
6. 完成 Stage 4.5：BC Rollout Evaluation。
7. 实现 Q-learning，让智能体通过奖励信号学习。
8. 用 C++ 重新实现 A*，理解算法在工程语言中的组织方式。
9. 整理实验图表、报告、笔记和问题。

## 项目结构

```text
gridnav/
├── env/                  # GridWorld 环境
├── data/                 # 生成的数据集和专家轨迹
├── planning_astar/       # Python A* 规划实验
├── ml_baseline/          # 传统机器学习 baseline
├── bc_pytorch/           # PyTorch 行为克隆实验
├── rollout/              # BC policy closed-loop rollout evaluation
├── rl_qlearning/         # Q-learning 强化学习实验
├── cpp_astar/            # C++ A* 实现
├── figures/              # 图表和可视化结果
├── reports/              # 实验报告
├── notes/                # 学习笔记和每日记录
├── scripts/              # 辅助脚本
├── requirements.txt      # Python 依赖
└── README.md             # 项目说明
```

## 学习目标

- 理解 GridWorld 中状态、动作、障碍物、起点和终点的表示方式。
- 扎实掌握 A* 的搜索逻辑、代价函数和启发式函数。
- 理解专家轨迹如何从规划算法转化为监督学习数据。
- 掌握机器学习中的特征、标签、训练集、测试集、泛化与 baseline 的作用。
- 通过行为克隆理解“模仿专家策略”为什么可以写成监督学习问题。
- 熟悉 PyTorch 中 forward、loss、backward 和 optimizer.step 的基本训练流程。
- 理解 Q-learning 中 state、action、reward、policy、Q(s,a) 和探索策略的关系。
- 通过 C++ A* 练习数据结构、优先队列和路径规划算法的工程表达。
- 在每一步记录问题、实验现象和自己的理解。

## 当前状态

- Stage 1：GridWorld + A* 已完成。
- Stage 2：专家轨迹数据生成已完成。
- Stage 3：传统机器学习 baseline 已完成收尾验收。
- Stage 4：PyTorch Behavior Cloning 已完成。
- Stage 4.5：BC Rollout Evaluation 已完成。
- Stage 5：Q-learning 强化学习理解中，工程未开始。
- Stage 6：C++ A* 工程实现已提前完成，学习理解暂缓。
- 当前学习主线：继续 Stage 5，完成后再回到 Stage 6 理解。

Stage 4.5 只加载 Stage 4 BC policy 做 BC Rollout Evaluation，不使用 reward，也不训练或更新策略，因此不是强化学习。

Stage 5 尚缺 Q-learning 训练脚本、Q-table、reward 设计、epsilon-greedy、episode 训练循环、Q-table 更新公式、训练与评估结果、报告和图表。Stage 6 已使用 C++17、`priority_queue` 和 `unordered_map` 完成与 Python demo 对应的 A* 工程实现，但待 Stage 5 完成后再回到 Stage 6 学习和理解。
