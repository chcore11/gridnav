# GridNav 学习笔记索引

这些笔记按项目阶段记录自己的理解、运行结果和遇到的问题。Stage 4 只保留正式笔记 `03_bc_pytorch_notes.md`，原重复占位文件 `03_pytorch_bc_notes.md` 已删除。

| 阶段 / 主题 | 笔记 | 主要内容 |
| --- | --- | --- |
| 项目过程 | `00_daily_log.md` | 项目初始化与阶段推进记录 |
| Stage 1 | `01_astar_notes.md` | GridWorld、A*、邻居、启发式和路径回溯 |
| Stage 2 | `02_expert_data_notes.md` | 专家路径、state-action 样本、动作分布 |
| Stage 3 | `02_ml_baseline_notes.md` | 分类任务、按地图划分、baseline 与评估指标 |
| Stage 4 | `03_bc_pytorch_notes.md` | PyTorch BC、训练循环、单步 accuracy 与局限 |
| Stage 4.5 | `04_rollout_notes.md` | Closed-loop rollout、错误累积、失败原因 |
| Stage 5 | `04_qlearning_notes.md` | 强化学习、reward、Q-table、episode、epsilon-greedy、alpha 和 gamma |
| Stage 6 | `05_cpp_astar_notes.md` | C++ 数据结构、CMake 和 Python/C++ A* 对应关系 |
| 跨阶段概念 | `06_core_concepts.md` | A*、baseline、行为克隆、PyTorch、Q-learning 和 C++ A* 总结 |
| 老师反馈 | `07_questions_for_teacher.md` | 后续适合向老师请教的问题 |

Stage 5 当前只完成固定地图上的 tabular Q-learning 最小闭环。它是强化学习，不依赖专家 action，但当前结果不代表随机地图泛化能力，也不是 DQN。
