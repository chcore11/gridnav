# 传统机器学习 Baseline 结果

## 任务

本阶段只进行单步 action prediction：输入当前 state 的八个特征，预测 A* 专家在该状态下选择的 action。没有进行完整导航 rollout。

## 数据与划分

当前数据来自 Stage 2 生成的 `data/expert_dataset.csv`，由 A* 在随机 GridWorld 地图中生成专家路径后拆分为 state-action 样本。

- 数据文件：`data/expert_dataset.csv`
- 特征：`agent_x, agent_y, goal_x, goal_y, obs_up, obs_down, obs_left, obs_right`
- 标签：`action`，编码为 `0 = up`、`1 = down`、`2 = left`、`3 = right`
- 总地图数：98
- 总样本数：970
- 训练地图数：78
- 测试地图数：20
- 训练样本数：776
- 测试样本数：194
- 随机种子：42
- 测试集比例：20% 的 `map_id`
- 训练 `map_id`：`[3, 4, 5, 7, 8, 9, 10, 11, 13, 15, 16, 17, 18, 19, 21, 22, 23, 25, 26, 27, 29, 30, 31, 32, 34, 35, 36, 37, 38, 39, 40, 41, 43, 45, 47, 48, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 65, 67, 68, 69, 70, 71, 72, 73, 74, 76, 78, 79, 80, 81, 82, 84, 86, 87, 88, 89, 90, 91, 93, 94, 95, 97, 98, 99]`
- 测试 `map_id`：`[1, 6, 12, 14, 20, 24, 28, 33, 42, 44, 46, 49, 64, 66, 75, 77, 83, 85, 92, 96]`

划分时先拆分 `map_id`，再选出对应样本。同一张地图产生的整条路径只会出现在训练集或测试集中的一边，避免相邻路径状态泄漏导致测试结果虚高。

## 模型设置

- KNN：`n_neighbors=5`，先使用 `StandardScaler` 标准化特征。
- Logistic Regression：`max_iter=1000`，先使用 `StandardScaler` 标准化特征。
- Decision Tree：使用默认树结构参数，并设置 `random_state=42`。

## Accuracy 汇总

| 模型 | Action accuracy |
| --- | ---: |
| KNN | 0.8144 |
| Logistic Regression | 0.8557 |
| Decision Tree | 0.7732 |

最佳模型是 **Logistic Regression**，测试集 action accuracy 为 **0.8557**。

混淆矩阵图保存于 `figures/ml_baseline_confusion_matrix.png`，图中同时包含三个模型，行表示真实动作，列表示预测动作。

## KNN

- Action accuracy：`0.8144`

### Classification report

```text
              precision    recall  f1-score   support

          up       0.80      0.88      0.84        41
        down       0.84      0.91      0.87        78
        left       0.80      0.72      0.76        46
       right       0.78      0.62      0.69        29

    accuracy                           0.81       194
   macro avg       0.81      0.78      0.79       194
weighted avg       0.81      0.81      0.81       194
```

### Confusion matrix

| actual \ predicted | up | down | left | right |
| --- | ---: | ---: | ---: | ---: |
| up | 36 | 0 | 3 | 2 |
| down | 1 | 71 | 3 | 3 |
| left | 5 | 8 | 33 | 0 |
| right | 3 | 6 | 2 | 18 |

## Logistic Regression

- Action accuracy：`0.8557`

### Classification report

```text
              precision    recall  f1-score   support

          up       0.84      0.90      0.87        41
        down       0.84      0.95      0.89        78
        left       0.92      0.72      0.80        46
       right       0.85      0.76      0.80        29

    accuracy                           0.86       194
   macro avg       0.86      0.83      0.84       194
weighted avg       0.86      0.86      0.85       194
```

### Confusion matrix

| actual \ predicted | up | down | left | right |
| --- | ---: | ---: | ---: | ---: |
| up | 37 | 2 | 1 | 1 |
| down | 0 | 74 | 1 | 3 |
| left | 4 | 9 | 33 | 0 |
| right | 3 | 3 | 1 | 22 |

## Decision Tree

- Action accuracy：`0.7732`

### Classification report

```text
              precision    recall  f1-score   support

          up       0.80      0.80      0.80        41
        down       0.83      0.86      0.84        78
        left       0.70      0.67      0.69        46
       right       0.68      0.66      0.67        29

    accuracy                           0.77       194
   macro avg       0.75      0.75      0.75       194
weighted avg       0.77      0.77      0.77       194
```

### Confusion matrix

| actual \ predicted | up | down | left | right |
| --- | ---: | ---: | ---: | ---: |
| up | 33 | 0 | 5 | 3 |
| down | 3 | 67 | 3 | 5 |
| left | 3 | 11 | 31 | 1 |
| right | 2 | 3 | 5 | 19 |

## 当前理解与局限

- Accuracy 表示测试样本中预测动作与 A* 专家动作相同的比例。
- Classification report 分动作展示 precision、recall 和 F1-score，可以帮助发现只看总 accuracy 时看不到的类别差异。
- Confusion matrix 可以直接看到每个真实动作容易被误判成哪个动作。
- 当前 state 只包含位置、目标位置和局部障碍，不包含完整地图；不同地图中相同特征可能对应不同的 A* 动作。
- 当前只评估单步分类，没有进行完整导航 rollout，因此 action accuracy 高不代表导航一定成功。

## 下一步建议

- 继续分析三个 baseline 的混淆矩阵，理解哪些动作容易混淆。
- 当前只完成 single-step action prediction，不代表模型已经能完整导航。
- 下一步可以做模型 rollout，测试模型从 start 到 goal 的实际导航成功率。
- 再之后才进入 PyTorch 行为克隆。
