# GridNav 工程导读

## 1. 项目目录地图

| 目录 | 工程作用 |
| --- | --- |
| `env/` | 保存所有阶段共享的 GridWorld 地图表示和基础地图操作 |
| `planning_astar/` | 实现 Python A*，为 demo 和专家数据生成提供路径 |
| `scripts/` | 运行固定地图 A* demo、基础断言测试和辅助资源脚本 |
| `data/` | 把随机地图中的 A* 路径转换为专家 state-action 数据 |
| `ml_baseline/` | 使用传统分类模型学习专家单步动作 |
| `bc_pytorch/` | 使用 PyTorch MLP 训练 Behavior Cloning policy |
| `rollout/` | 加载已训练 BC policy，进行 closed-loop 导航评估 |
| `rl_qlearning/` | 在固定地图上通过 reward 训练 tabular Q-learning |
| `cpp_astar/` | 使用 C++17 和 CMake 重新实现并验证 A* |
| `notes/` | 记录各阶段学习理解、概念边界和待请教问题 |
| `reports/` | 保存各阶段结果、最终汇总和汇报准备材料 |
| `figures/` | 保存仓库跟踪的关键实验图 |
| `checkpoints/` | 保存被 `.gitignore` 忽略的 BC 模型 checkpoint |

## 2. 核心文件作用

| 文件 | 作用 |
| --- | --- |
| `env/grid_world.py` | 定义地图宽高、起终点、障碍物、邻居过滤和数组表示 |
| `planning_astar/astar.py` | 使用曼哈顿距离、优先队列和父节点回溯实现 A* |
| `scripts/run_astar_demo.py` | 构造固定地图，运行 A* 并生成 demo 图 |
| `scripts/run_astar_demo_test.py` | 检查有路径与无路径场景的基础行为 |
| `data/generate_expert_data.py` | 随机生成地图，将 A* 路径拆为 `state -> action` 样本 |
| `ml_baseline/train_baselines.py` | 按 `map_id` 划分数据，训练并比较三个传统分类模型 |
| `bc_pytorch/train_bc.py` | 训练 BC MLP，保存 checkpoint、训练曲线和分类结果 |
| `rollout/evaluate_rollout.py` | 重放测试地图，用 BC policy 连续导航并统计失败原因 |
| `rl_qlearning/train_qlearning.py` | 在固定 `6 x 6` 地图上训练 Q-table 并评估 greedy policy |
| `cpp_astar/astar.h` | 声明 C++ 坐标、网格和 A* 接口 |
| `cpp_astar/astar.cpp` | 实现 C++ Grid 邻居查询、优先队列搜索和路径回溯 |
| `cpp_astar/main.cpp` | 运行 C++ demo，打印结果并验证路径合法性 |
| `cpp_astar/CMakeLists.txt` | 配置 C++17 并生成 `cpp_astar_demo` |

## 3. 脚本输入与输出

| 脚本 | 主要输入 | 主要输出 |
| --- | --- | --- |
| `scripts/run_astar_demo.py` | 代码中定义的固定地图 | 控制台路径、`figures/astar_demo.png` |
| `scripts/run_astar_demo_test.py` | 两个代码内测试地图 | 基础断言通过信息 |
| `data/generate_expert_data.py` | 固定随机种子、随机地图参数、Python A* | `data/expert_dataset.csv`、`reports/dataset_summary.md` |
| `ml_baseline/train_baselines.py` | `data/expert_dataset.csv` | `reports/ml_baseline_result.md`、baseline 混淆矩阵图 |
| `bc_pytorch/train_bc.py` | `data/expert_dataset.csv` | BC 报告、训练曲线、`checkpoints/bc_policy.pt` |
| `rollout/evaluate_rollout.py` | 专家数据、BC checkpoint、固定随机种子重放地图 | rollout 报告和成功率汇总图 |
| `rl_qlearning/train_qlearning.py` | 代码中定义的固定地图和 reward | Q-learning 报告、训练曲线和成功率汇总图 |
| `cpp_astar_demo` | `main.cpp` 中定义的固定地图 | C++ 路径、节点数、步数和合法性检查 |

## 4. 完整数据流

```text
GridWorld + Python A*
        |
        v
随机地图中的 A* 专家路径
        |
        v
data/expert_dataset.csv
        |
        +----------------------+
        |                      |
        v                      v
Traditional ML baseline   PyTorch BC
                               |
                               v
                    checkpoints/bc_policy.pt
                               |
                               v
                    BC Rollout Evaluation

固定 6 x 6 GridWorld + reward ---> Tabular Q-learning

固定 demo GridWorld -----------> C++ A* 工程实现
```

Stage 3 和 Stage 4 都学习专家数据中的单步动作。Stage 4.5 只评估 Stage 4 policy，不使用 reward。Stage 5 不读取专家 action，而是通过环境 reward 更新 Q-table。

## 5. 推荐读代码顺序

1. `env/grid_world.py`：先理解地图坐标、障碍、边界和邻居。
2. `planning_astar/astar.py`：理解 A* 如何从地图得到路径。
3. `scripts/run_astar_demo.py` 与 `scripts/run_astar_demo_test.py`：观察 A* 如何被调用和检查。
4. `data/generate_expert_data.py`：理解路径如何变成监督学习样本。
5. `ml_baseline/train_baselines.py`：理解数据划分、baseline 和单步评估。
6. `bc_pytorch/train_bc.py`：对应理解 Dataset、DataLoader、训练循环和 checkpoint。
7. `rollout/evaluate_rollout.py`：理解单步 accuracy 与完整导航 success rate 的差距。
8. `rl_qlearning/train_qlearning.py`：比较 reward 学习与专家模仿的区别。
9. `cpp_astar/`：最后比较 Python 原型和 C++ 工程表达。

## 6. 必须能回答的工程问题

- 为什么项目坐标写作 `(x, y)`，NumPy 数组却使用 `grid[y, x]`？
- A* 的 `frontier`、`cost_so_far`、`came_from` 分别保存什么？
- 一条 A* 路径如何拆成多个 `state -> action` 样本？
- 为什么训练集和测试集按 `map_id` 划分，而不是随机拆 CSV 行？
- 为什么 BC single-step accuracy 为 `0.8763`，rollout success rate 却只有 `65.00%`？
- 为什么 Stage 4.5 不是强化学习？
- Q-learning 的 reward、Q-table、epsilon-greedy、alpha 和 gamma 分别起什么作用？
- 为什么固定 `6 x 6` 地图上的 `100%` 成功率不能说明随机地图泛化？
- C++ A* 与 Python A* 的算法结构如何一一对应？
- 哪些生成物被 `.gitignore` 忽略，克隆仓库后应如何重建？
