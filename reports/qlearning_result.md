# Stage 5：Tabular Q-learning 结果

## 阶段目标

在固定 GridWorld 中实现最小 tabular Q-learning。智能体不读取 A* 路径或专家 action，而是通过自己选择动作、获得 reward、更新 Q-table 来学习从 start 到 goal 的策略。

这是强化学习，因为训练数据来自智能体与环境的交互，策略会根据 reward 反馈逐步改变。它不是监督学习，也没有使用 PyTorch 或 DQN。

## 环境与定义

- 地图大小：`6 x 6`
- Start：`(0, 0)`
- Goal：`(5, 5)`
- Obstacles：`(1, 0), (1, 1), (1, 2), (2, 4), (3, 1), (3, 2), (3, 3), (3, 4)`
- State：智能体当前坐标 `(x, y)`
- Action：`0 = up`、`1 = down`、`2 = left`、`3 = right`
- Q-table：形状为 `(6, 6, 4)` 的表，`Q[y, x, action]` 表示在坐标 `(x, y)` 选择该动作的价值估计
- 到达 goal：`+100`
- 撞墙、越界或撞障碍：`-10`
- 普通移动一步：`-1`
- 训练时碰撞后留在原状态并继续 episode，让 Q-table 学习降低该动作的价值
- 最大步数：`100`，防止 episode 无限运行
- 随机种子：`42`

## Q-learning 参数

- Alpha 学习率：`0.1`
- Gamma 折扣因子：`0.95`
- 初始 epsilon：`1.0`
- 最小 epsilon：`0.05`
- Epsilon decay：每个 episode 后乘以 `0.995`
- 最终 epsilon：`0.0500`
- 训练 episode 数：`3000`

更新公式：

`Q(s,a) = Q(s,a) + alpha * [reward + gamma * max Q(s',a') - Q(s,a)]`

到达 goal 后 episode 结束，终止状态不再加入未来 Q-value。

## 训练结果

- 总训练 episode 数：3000
- 整体训练成功率：`99.50%`
- 最后 100 个 episode 成功率：`100.00%`
- 最后 100 个 episode 平均 total reward：`88.84`
- 最后 100 个 episode 平均 steps：`10.54`
- 每个 episode 都记录了 total reward、是否成功和 episode steps
- `figures/qlearning_training_curve.png` 展示 total reward 和最近 100 个 episode 的 success rate

### 训练过程摘要

| episode | 最近 100 个 episode 成功率 | 最近 100 个 episode 平均 reward |
| ---: | ---: | ---: |
| 100 | 85.00% | -131.87 |
| 500 | 100.00% | 85.70 |
| 1000 | 100.00% | 88.38 |
| 2000 | 100.00% | 88.12 |
| 3000 | 100.00% | 88.84 |

## Greedy Policy 评估

训练完成后关闭随机探索，每个 state 都选择 Q 值最大的动作。固定地图和 greedy policy 都是确定性的，因此重复评估 100 次用于统计结果。

- Success count：100
- Failure count：0
- Success rate：`100.00%`
- Average steps on success：`10.00`
- Greedy path：`(0, 0) -> (0, 1) -> (0, 2) -> (0, 3) -> (1, 3) -> (1, 4) -> (1, 5) -> (2, 5) -> (3, 5) -> (4, 5) -> (5, 5)`

### Failure reasons

| failure reason | count |
| --- | ---: |
| `hit_obstacle` | 0 |
| `out_of_bounds` | 0 |
| `loop_detected` | 0 |
| `max_steps_exceeded` | 0 |

## 与前面阶段的区别

- Stage 4 Behavior Cloning 使用专家 action 作为标签，学习专家给出的 `state -> action` 映射。
- Stage 4.5 BC Rollout Evaluation 只评估已经训练好的 BC policy，不使用 reward，也不训练或更新策略，因此不是强化学习。
- Stage 5 Q-learning 不依赖专家 action。智能体通过 epsilon-greedy 自己探索动作，根据 reward 和下一状态的 Q-value 更新 Q-table。

## 当前边界

- 当前是固定地图上的最小 tabular Q-learning，只使用坐标作为 state。
- Q-table 只适用于这张固定地图，不能直接泛化到不同 goal 或障碍布局。
- 当前没有使用 A* 专家 action、PyTorch 或 DQN。
