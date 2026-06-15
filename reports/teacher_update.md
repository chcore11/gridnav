# GridNav 给张老师的阶段汇报

## 1. 这段时间完成了什么

我围绕同一个 GridWorld 导航任务，按顺序完成了 Python A*、专家数据生成、传统机器学习 baseline、PyTorch Behavior Cloning、BC rollout evaluation、固定地图上的 tabular Q-learning，以及 C++ A* 工程实现。

目前 Stage 1 至 Stage 6 已完成，正在进行 Stage 7 最终交付整理。当前不再新增算法功能，重点是统一仓库说明、复现步骤、结果边界和汇报材料。

## 2. 每个阶段学到了什么

- Stage 1：理解 GridWorld、曼哈顿距离、A* 的 `g(n)`、`h(n)`、优先队列和路径回溯。
- Stage 2：理解如何把 A* 路径拆成 `state -> action` 专家样本。
- Stage 3：理解 feature、label、按地图划分训练/测试集、accuracy、classification report 和 confusion matrix。
- Stage 4：理解 PyTorch 的 Tensor、Dataset、DataLoader、forward、loss、backward 和 optimizer。
- Stage 4.5：理解 closed-loop rollout、错误累积和 distribution shift；这一阶段只评估 BC policy，不是强化学习。
- Stage 5：理解 reward、Q-table、episode、epsilon-greedy、alpha 和 gamma；Q-learning 不依赖专家 action。
- Stage 6：理解同一个 A* 算法在 C++17、`priority_queue`、`unordered_map` 和 CMake 中的工程表达。

## 3. 当前结果

| 内容 | 结果 |
| --- | --- |
| Python A* | 19 个路径节点，18 个移动步数 |
| Expert Dataset | 100 张地图尝试，98 张有路径，970 个样本 |
| 最佳传统 ML baseline | Logistic Regression accuracy：`0.8557` |
| PyTorch BC | single-step accuracy：`0.8763` |
| BC rollout | `13/20 = 65.00%` |
| Tabular Q-learning | 固定 `6 x 6` 地图 greedy success rate：`100/100` |
| C++ A* | 19 个路径节点，18 步，路径合法性检查通过 |

我认为最重要的观察是：BC 的单步 accuracy 达到 `0.8763`，但完整 rollout success rate 只有 `65.00%`。这让我更清楚地理解了单步分类指标与连续决策能力之间的差距。

## 4. 项目边界

- 这是学习型入门项目，不是工业级导航系统。
- Stage 4.5 只是 BC rollout evaluation，不使用 reward，也不更新策略，因此不是强化学习。
- Q-learning 的 `100%` 结果只适用于一张固定 `6 x 6` 地图，不代表随机地图泛化能力。
- 当前未实现 CNN、DQN、复杂动态地图、ROS/Gazebo 或真实机器人控制。
- 现有结果主要用于验证概念理解和最小工程闭环，不夸大为稳定导航能力。

## 5. 希望老师反馈什么

- 当前从规划、监督学习、模仿学习到强化学习的学习顺序是否合理？
- 现有 state 特征只包含位置、目标和局部障碍，下一步应优先改进状态表示还是评估设计？
- BC rollout 暴露出的分布偏移问题，适合作为后续深入模仿学习的入口吗？
- 固定地图 tabular Q-learning 完成后，是否应先研究随机地图泛化，再考虑 DQN？
- 从 C++ A* 继续衔接机器人方向时，适合优先补充哪些路径规划、控制或 ROS 基础？
