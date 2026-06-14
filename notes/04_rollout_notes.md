# Stage 4.5：BC Rollout Evaluation 学习笔记

## Rollout 是什么

我现在把 rollout 理解成让模型真正从起点开始运行。模型每一步都根据当前 state 预测 action，环境根据这个动作更新位置，然后模型再根据新的位置预测下一步，直到到达 goal 或失败。

Stage 4 的测试会把专家数据中的一个 state 单独交给模型，检查这一小步 action 是否预测正确。Stage 4.5 不再逐条检查独立样本，而是让同一个 BC policy 连续控制整个导航过程。

## 什么是 closed-loop evaluation

Closed-loop 表示模型的输出会影响下一步输入。模型预测动作后，智能体移动到新位置；新位置又会成为下一次预测的一部分。

这和只在固定测试样本上做 single-step action prediction 不同。单步分类时，一个样本预测错了不会改变下一条测试样本；closed-loop rollout 中，一步走错会让后面的状态全部发生变化。

## 这次如何评估

训练脚本先保存 `checkpoints/bc_policy.pt`。Checkpoint 中不仅有模型权重，还保存了训练时的特征均值、标准差、特征顺序和测试地图 ID。Rollout 推理必须使用相同的标准化参数，否则送进模型的输入分布会和训练时不同。

评估脚本使用 Stage 2 的固定随机种子和原始生成规则重放地图，只评估 Stage 4 中没有参加训练的 20 张测试地图。每张地图还会运行 A*，得到同一地图上的最短路径移动步数作为比较基准。

## 为什么 single-step accuracy 不等于导航成功率

Stage 4 的 single-step action accuracy 是 `0.8763`，但这次 rollout success rate 是 `65.00%`，20 张测试地图中有 13 张成功、7 张失败。

我认为差距的主要原因是错误累积。单步 accuracy 表示大部分独立状态可以预测正确，但完整导航通常需要连续正确很多步。模型一旦选错动作，就可能进入专家路径中没有出现过的位置，后续预测会变得更困难。

这次 7 次失败中有 5 次是 `loop_detected`，说明模型走错后可能在几个位置之间反复选择动作。另有 2 次是 `hit_obstacle`，说明模型虽然看到了局部障碍特征，仍然可能预测一个不可执行动作。

## Success rate 和路径长度

Success rate 表示成功到达 goal 的 episode 数量占全部评估 episode 的比例：

```text
success rate = success count / evaluated maps
             = 13 / 20
             = 65.00%
```

路径长度统一记录实际移动步数，不包含起点。成功地图中，模型平均移动 `9.54` 步，A* 平均移动 `9.38` 步。

Path length gap 表示：

```text
model steps - A* steps
```

成功地图上的平均 path length gap 是 `0.15`。这说明模型一旦成功，路径通常接近 A* 最短路径；但这个指标只统计成功 episode，不能掩盖另外 7 张地图的失败。

## Failure reasons 怎么理解

- `out_of_bounds`：模型预测动作后会走出地图边界。
- `hit_obstacle`：模型预测动作后会撞上障碍物。
- `loop_detected`：模型回到了已经访问过的位置。因为地图和 goal 不变，同一个位置对应的 state 也会重复，确定性 policy 很可能继续重复动作。
- `max_steps_exceeded`：在最大允许步数内没有成功，也没有提前触发其他失败条件。

这次结果是：

```text
out_of_bounds: 0
hit_obstacle: 2
loop_detected: 5
max_steps_exceeded: 0
```

## 为什么这不是强化学习

这次没有根据 rollout 成功或失败更新模型，也没有 reward、Q-value、探索或策略优化。评估脚本只是加载 Stage 4 已经训练好的 BC policy，观察它在环境中的表现，不会更新策略。

所以 Stage 4.5 是模型评估，不是训练，更不是强化学习。原始 Stage 5 仍然是 Q-learning；Stage 5 才会通过 reward 与环境交互并学习策略。

## Stage 3、Stage 4、Stage 4.5 和 Stage 5 的关系

Stage 3 使用传统机器学习模型做单步动作分类，建立 baseline。Stage 4 使用 PyTorch MLP 做同一个 `state -> action` 单步分类任务。Stage 4.5 把 Stage 4 的模型放回环境中，只检查单步预测能力能否转化成完整导航能力。Stage 5 Q-learning 则会使用 reward 学习新的策略。

这次结果让我更清楚地看到：单步 accuracy 很重要，但它不是导航任务的最终答案。Rollout success rate 和 failure reasons 能揭示分类报告中不容易直接看到的错误累积问题。
