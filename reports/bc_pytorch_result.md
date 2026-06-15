# PyTorch Behavior Cloning 结果

## 任务

使用一个简单的 PyTorch MLP 学习 A* 专家数据中的 `state -> action` 映射。当前评估仍是 single-step action prediction，没有进行完整导航 rollout。

## 数据与划分

- 数据来源：`data/expert_dataset.csv`
- 特征 X：`agent_x, agent_y, goal_x, goal_y, obs_up, obs_down, obs_left, obs_right`
- 标签 y：`action`，编码为 `0 = up`、`1 = down`、`2 = left`、`3 = right`
- 总地图数：98
- 总样本数：970
- 训练地图数：78
- 测试地图数：20
- 训练样本数：776
- 测试样本数：194
- 随机种子：42

训练集和测试集先按 `map_id` 划分，并断言两个 map ID 集合不相交。同一张地图的路径样本不会同时出现在训练集和测试集中。特征标准化参数只根据训练集计算。

## 模型与训练设置

```text
Linear(8, 64)
ReLU
Linear(64, 32)
ReLU
Linear(32, 4)
```

- PyTorch：`2.4.1+cpu`
- Dataset：`ExpertActionDataset`
- DataLoader batch size：`64`
- Loss function：`CrossEntropyLoss`
- Optimizer：`Adam`
- Learning rate：`0.001`
- Epoch 数：`100`
- Final train loss：`0.169001`

训练曲线保存于 `figures/bc_training_curve.png`。
用于 rollout 评估的 checkpoint 保存于 `checkpoints/bc_policy.pt`，其中包含模型权重、标准化参数、特征顺序和训练/测试地图 ID。

## 测试结果

- PyTorch BC single-step action accuracy：`0.8763`
- Stage 3 Logistic Regression accuracy：`0.8557`
- PyTorch BC 与 Logistic Regression 的 accuracy 差值：`+0.0206`

### Classification report

```text
              precision    recall  f1-score   support

          up       0.88      0.85      0.86        41
        down       0.88      0.94      0.91        78
        left       0.90      0.76      0.82        46
       right       0.84      0.93      0.89        29

    accuracy                           0.88       194
   macro avg       0.87      0.87      0.87       194
weighted avg       0.88      0.88      0.87       194
```

### Confusion matrix

| actual \ predicted | up | down | left | right |
| --- | ---: | ---: | ---: | ---: |
| up | 35 | 2 | 3 | 1 |
| down | 0 | 73 | 1 | 4 |
| left | 4 | 7 | 35 | 0 |
| right | 1 | 1 | 0 | 27 |

## 当前结论与边界

- 当前 PyTorch BC 使用神经网络模仿 A* 专家的单步动作。
- 当前结果可以与 Stage 3 Logistic Regression baseline 比较，但模型更复杂不保证 accuracy 一定更高。
- Stage 4 是 PyTorch Behavior Cloning 训练与 single-step action prediction 评估，本阶段本身没有进行 rollout。
- 完整导航能力由 Stage 4.5 BC Rollout Evaluation 单独评估；Stage 4.5 只加载已训练的 BC policy，不使用 reward，也不更新策略。
- 本阶段没有进入强化学习。
