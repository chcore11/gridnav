# GridNav 汇报大纲

## 汇报标题

**GridNav：用一个网格导航任务串联路径规划、模仿学习、强化学习与 C++ 工程实现**

建议控制在 10 页，重点讲清方法之间的关系、结果边界和自己的理解。

| 页码 | 页面标题 | 要讲什么 | 可引用材料 |
| ---: | --- | --- | --- |
| 1 | 项目目标与阶段路线 | 为什么选择可控的 GridWorld；Stage 1 至 Stage 7 如何逐步推进 | `README.md` 阶段总览、`reports/final_summary.md` |
| 2 | GridWorld 与 Python A* | 地图坐标、障碍、邻居、曼哈顿距离、A* 搜索与路径回溯 | `figures/astar_demo.png`、`notes/01_astar_notes.md` |
| 3 | 从 A* 路径到专家数据 | 路径如何拆成 `state -> action`；八个特征和四个动作；数据统计 | `reports/dataset_summary.md` |
| 4 | Traditional ML baseline | 为什么先做 baseline；按 `map_id` 划分；三个模型结果比较 | `figures/ml_baseline_confusion_matrix.png`、`reports/ml_baseline_result.md` |
| 5 | PyTorch Behavior Cloning | MLP、训练循环、loss 与单步 accuracy；BC 仍是监督学习 | `figures/bc_training_curve.png`、`reports/bc_pytorch_result.md` |
| 6 | BC Rollout：单步正确不等于导航成功 | `0.8763` 单步 accuracy 与 `65.00%` rollout success rate；错误累积和失败原因 | `figures/rollout_success_summary.png`、`reports/rollout_result.md` |
| 7 | Tabular Q-learning | reward、Q-table、episode、epsilon-greedy；与 BC 的学习信号区别 | `figures/qlearning_training_curve.png`、`notes/04_qlearning_notes.md` |
| 8 | Q-learning 结果与边界 | 固定 `6 x 6` 地图上 greedy `100%`；为什么不能解释为随机地图泛化 | `figures/qlearning_success_summary.png`、`reports/qlearning_result.md` |
| 9 | C++ A* 与工程组织 | Python 与 C++ A* 的对应关系；类型、优先队列、哈希和 CMake | `reports/cpp_astar_result.md`、`notes/05_cpp_astar_notes.md` |
| 10 | 总结、局限与希望老师建议 | 主要学习结论；当前没有实现的扩展；希望老师指导下一步顺序 | `reports/final_summary.md`、`notes/07_questions_for_teacher.md` |

## 建议讲述主线

1. 先用 A* 得到可解释路径。
2. 再把路径转换为专家数据，分别训练传统模型和 PyTorch BC。
3. 用 rollout 发现单步分类指标无法完全代表连续导航能力。
4. 再通过 Q-learning 对比“模仿专家”和“根据 reward 学习”的区别。
5. 最后用 C++ A* 理解同一算法的工程表达，并明确整个项目的学习边界。

## 最后一页希望老师给的建议

- 当前学习路线是否有概念缺口？
- 下一步应优先研究状态表示、评估方法，还是更复杂的强化学习？
- 如何从这个简化 GridWorld 项目稳妥衔接机器人学习或具身智能？
- C++ 与 ROS/Gazebo 的学习顺序应该如何安排？
