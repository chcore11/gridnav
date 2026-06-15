# Q-learning 学习笔记

## 当前状态

我现在只开始理解 Q-learning 的基本概念，工程还没有开始。目前没有训练脚本、Q-table 或训练和评估结果，所以 Stage 5 不能算完成。

Stage 4.5 做的是 BC Rollout Evaluation。它只把已经训练好的 BC policy 放进环境里评估，不使用 reward，也不训练或更新策略，因此不是强化学习。Q-learning 不依赖专家 action，而是通过和环境交互，根据 reward 学习策略。

## 接下来需要理解和完成

- 定义 state、action 和 reward。
- 理解 Q(s,a) 和 Q-table 表示什么。
- 使用 epsilon-greedy 平衡探索和利用。
- 编写 episode 训练循环。
- 理解并实现 Q-table 更新公式。
- 观察学习率、折扣因子和 reward 设计对结果的影响。
- 完成训练和评估，记录成功率、路径长度等结果。
- 整理结果报告和图表。

Stage 6 的 C++ A* 工程实现虽然已经提前完成，但当前学习主线先停在 Stage 5。等 Q-learning 的工程、结果和理解都完成后，再回到 Stage 6 学习。
