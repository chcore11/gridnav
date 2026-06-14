# BC Policy Rollout Evaluation 结果

## 评估目标

评估 Stage 4 训练出的 Behavior Cloning policy 能否在 GridWorld 中从 start 开始，根据自己的预测动作一步步走到 goal。这是 closed-loop rollout evaluation，不是训练，也不是强化学习。

## 评估设置

- 加载 checkpoint：`checkpoints/bc_policy.pt`
- Checkpoint PyTorch 版本：`2.4.1+cpu`
- BC single-step test accuracy：`0.8763`
- 评估地图：Stage 4 checkpoint 中记录的 test map IDs
- 使用测试地图数量：20
- 地图重放：使用 Stage 2 固定随机种子和原始地图生成规则
- 每张地图最大步数：`4 * width * height`
- A* 基准路径长度：`len(astar_search(world)) - 1`

## 汇总结果

- Evaluated maps：20
- Success count：13
- Failure count：7
- Success rate：`65.00%`
- Average model steps on success：`9.54`
- Average A* steps on success：`9.38`
- Average path length gap on success：`0.15`

### Failure reasons

| failure reason | count |
| --- | ---: |
| `out_of_bounds` | 0 |
| `hit_obstacle` | 2 |
| `loop_detected` | 5 |
| `max_steps_exceeded` | 0 |

## 每张测试地图结果

| map_id | success | model steps | A* steps | path gap | failure reason |
| ---: | --- | ---: | ---: | ---: | --- |
| 1 | True | 9 | 9 | 0 | - |
| 6 | True | 8 | 8 | 0 | - |
| 12 | True | 8 | 8 | 0 | - |
| 14 | True | 10 | 10 | 0 | - |
| 20 | True | 8 | 8 | 0 | - |
| 24 | True | 7 | 7 | 0 | - |
| 28 | False | 9 | 6 | - | loop_detected |
| 33 | False | 3 | 11 | - | loop_detected |
| 42 | False | 0 | 12 | - | hit_obstacle |
| 44 | False | 6 | 8 | - | loop_detected |
| 46 | True | 10 | 10 | 0 | - |
| 49 | True | 12 | 12 | 0 | - |
| 64 | False | 8 | 10 | - | loop_detected |
| 66 | True | 8 | 8 | 0 | - |
| 75 | False | 0 | 14 | - | hit_obstacle |
| 77 | True | 10 | 10 | 0 | - |
| 83 | False | 7 | 11 | - | loop_detected |
| 85 | True | 8 | 8 | 0 | - |
| 92 | True | 17 | 15 | 2 | - |
| 96 | True | 9 | 9 | 0 | - |

## 当前理解与边界

- Single-step accuracy 衡量独立测试状态上的动作预测正确率；rollout success rate 衡量模型连续使用自己预测的动作后，最终到达 goal 的比例。
- 单步错误会改变后续状态，模型可能进入专家数据中较少出现的位置，因此较高的 single-step accuracy 不保证较高的 rollout success rate。
- 当前只加载 Stage 4 BC policy 做评估，没有训练新策略。
- 当前不是强化学习，也没有实现 Q-learning。
