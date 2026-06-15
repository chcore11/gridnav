# GridNav

GridNav 是一个用网格导航任务串联路径规划、专家数据生成、传统机器学习、PyTorch 行为克隆、BC rollout、tabular Q-learning 和 C++ A* 工程实现的学习项目。

项目围绕同一个可控的 GridWorld 任务，逐步比较不同方法的输入、学习方式、评估指标和适用边界。重点是概念清楚、过程可解释、结果可复现，而不是堆叠复杂功能。

## 当前状态

- Stage 1 至 Stage 6 已完成。
- 当前处于 **Stage 7：最终交付整理与汇报**。
- 当前不再新增技术功能，重点是完善仓库交付、统一复现说明并收集老师反馈。

## 阶段总览

| 阶段 | 内容 | 状态 |
| --- | --- | --- |
| Stage 1 | GridWorld + Python A* | 已完成 |
| Stage 2 | Expert Dataset | 已完成 |
| Stage 3 | Traditional ML baseline | 已完成 |
| Stage 4 | PyTorch Behavior Cloning | 已完成 |
| Stage 4.5 | BC Rollout Evaluation | 已完成 |
| Stage 5 | Tabular Q-learning | 已完成 |
| Stage 6 | C++ A* | 已完成 |
| Stage 7 | Final Summary & Presentation | 最终交付整理中 |

## 关键结果

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

## 项目结构

```text
gridnav/
├── env/                  # GridWorld 地图表示、边界、障碍和邻居操作
├── planning_astar/       # Python A* 搜索实现
├── scripts/              # A* demo、轻量测试和辅助脚本
├── data/                 # 专家数据生成脚本与被忽略的生成数据
├── ml_baseline/          # KNN、Logistic Regression、Decision Tree baseline
├── bc_pytorch/           # PyTorch Behavior Cloning 训练
├── rollout/              # 已训练 BC policy 的 closed-loop 评估
├── rl_qlearning/         # 固定地图上的 Tabular Q-learning
├── cpp_astar/            # C++17 A*、CMake 配置与被忽略的构建目录
├── notes/                # 各阶段学习笔记与概念总结
├── reports/              # 阶段结果、最终汇总和汇报准备材料
├── figures/              # 仓库跟踪的关键实验图
└── checkpoints/          # 被忽略的模型 checkpoint
```

## 从零复现

以下命令按推荐顺序从仓库根目录运行。先准备与 `requirements.txt` 一致的 Python 环境；项目不会自动安装依赖。

```bash
cd ~/projects/gridnav
source .venv/bin/activate
```

| 顺序 | 命令 | 作用与主要输出 |
| ---: | --- | --- |
| 1 | `python scripts/run_astar_demo.py` | 运行固定地图 Python A*，打印路径并生成 `figures/astar_demo.png` |
| 2 | `python scripts/run_astar_demo_test.py` | 检查 A* demo 路径和无路径场景的基础断言 |
| 3 | `python data/generate_expert_data.py` | 生成 `data/expert_dataset.csv`，并更新 `reports/dataset_summary.md` |
| 4 | `python ml_baseline/train_baselines.py` | 更新 `reports/ml_baseline_result.md` 和 `figures/ml_baseline_confusion_matrix.png` |
| 5 | `python bc_pytorch/train_bc.py` | 更新 BC 报告、`figures/bc_training_curve.png` 和 `checkpoints/bc_policy.pt` |
| 6 | `python rollout/evaluate_rollout.py` | 更新 `reports/rollout_result.md` 和 `figures/rollout_success_summary.png` |
| 7 | `python rl_qlearning/train_qlearning.py` | 更新 Q-learning 报告、训练曲线图和 greedy success 汇总图 |

如果本机没有 `python` 命令，可将上述命令中的 `python` 替换为 `python3`。

C++ A* 使用 CMake 构建和运行：

```bash
cmake -S cpp_astar -B cpp_astar/build
cmake --build cpp_astar/build
./cpp_astar/build/cpp_astar_demo
```

这三条命令依次配置、编译并运行 C++ A* demo；程序会打印路径节点数、移动步数和路径合法性检查结果。

## 被忽略的生成文件

- `data/expert_dataset.csv` 被 `.gitignore` 忽略，需要运行 `python data/generate_expert_data.py` 重生成。
- `checkpoints/bc_policy.pt` 被 `.gitignore` 忽略，需要先生成专家数据，再运行 `python bc_pytorch/train_bc.py` 重生成。
- `cpp_astar/build/` 被 `.gitignore` 忽略，需要重新运行 CMake 配置和构建命令。
- `rollout/evaluate_rollout.py` 同时依赖专家数据和 BC checkpoint，应在 Stage 2、Stage 4 的生成命令之后运行。
- 结果文件、图片和报告是否随仓库提供，以仓库实际跟踪情况为准；当前六张关键结果图和各阶段 Markdown 报告已被跟踪。

## 结果边界

- Stage 4.5 只评估 Stage 4 已训练的 BC policy，不使用 reward，也不更新策略，因此不是强化学习。
- BC rollout 的 `65.00%` success rate 说明单步 accuracy 不能保证稳定导航，不代表已经形成可靠导航系统。
- Tabular Q-learning 的 `100%` greedy success rate 只适用于训练使用的固定 `6 x 6` 地图，不代表随机地图泛化能力。
- 当前项目是用于学习基础概念和工程流程的入门项目，不是工业级导航系统。
- 当前没有实现 CNN、DQN、复杂动态地图或 ROS/Gazebo 接入。

## 关键图表

| 阶段 | 图片 |
| --- | --- |
| Stage 1 A* | `figures/astar_demo.png` |
| Stage 3 ML baseline | `figures/ml_baseline_confusion_matrix.png` |
| Stage 4 PyTorch BC | `figures/bc_training_curve.png` |
| Stage 4.5 BC rollout | `figures/rollout_success_summary.png` |
| Stage 5 Q-learning 训练 | `figures/qlearning_training_curve.png` |
| Stage 5 Q-learning 评估 | `figures/qlearning_success_summary.png` |

## 文档入口

- [ROADMAP.md](ROADMAP.md)：完整阶段路线、验收标准和后续扩展方向。
- [reports/final_summary.md](reports/final_summary.md)：项目结果与学习结论总览。
- [reports/engineering_walkthrough.md](reports/engineering_walkthrough.md)：目录地图、数据流和推荐读代码顺序。
- [reports/teacher_update.md](reports/teacher_update.md)：发给老师前的项目进展与问题整理。
- [reports/presentation_outline.md](reports/presentation_outline.md)：最终汇报 PPT 结构建议。
- [notes/README.md](notes/README.md)：各阶段学习笔记索引。
