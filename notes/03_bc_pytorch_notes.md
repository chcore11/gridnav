# PyTorch Behavior Cloning 学习笔记

## 这次阶段做了什么

这次我用 PyTorch 写了一个最小版本的 Behavior Cloning，也就是行为克隆模型。输入仍然是 Stage 3 使用的八个 state 特征，输出是 A* 专家选择的 up、down、left、right 四种 action。

训练数据来自 `data/expert_dataset.csv`。数据仍然按 `map_id` 划分，得到 78 张训练地图和 20 张测试地图。我在代码中增加了断言，确认训练和测试的 map ID 没有交集，避免同一张地图的相邻路径样本同时出现在两边。

## Behavior Cloning 是什么

我现在把 Behavior Cloning 理解成“看专家示范，然后模仿专家动作”。当前项目中的专家是 A*，A* 先生成路径，再把路径拆成很多 `state -> action` 样本。PyTorch 模型看到 state 后，学习预测 A* 在这个状态下会选择哪个 action。

BC 仍然属于监督学习，因为每条训练样本都有明确的正确标签 `action`。模型不是自己探索奖励，也不是通过成功或失败来学习，所以它和强化学习不同。

## Tensor、Dataset 和 DataLoader

Tensor 是 PyTorch 使用的数据形式，可以理解成适合神经网络计算的多维数组。这次特征会转换为 `float32` tensor，动作标签会转换为 `long` tensor。标签使用 `long`，是因为 `CrossEntropyLoss` 需要用整数类别编号表示正确答案。

Dataset 负责说明“一条样本怎么取出来”。`ExpertActionDataset` 保存所有特征和标签，`__getitem__()` 根据下标返回一条 `(features, label)`。

DataLoader 负责把 Dataset 中的样本组成 batch。训练时会打乱样本顺序，每次把 64 条样本交给模型；测试时不需要打乱。它让我不需要手动切 batch。

## MLP、forward 和 logits

MLP 是由多层全连接层组成的神经网络。这次使用的结构是：

```text
Linear(8, 64)
ReLU
Linear(64, 32)
ReLU
Linear(32, 4)
```

输入维度是 8，对应八个 state 特征；输出维度是 4，对应四种动作。

`forward` 表示输入数据如何经过网络得到输出。模型最后输出的四个数字叫 logits，它们是模型给四种动作的原始分数。logit 最大的动作会作为预测结果。logits 本身不要求是概率，也不需要在模型最后手动加 softmax。

## Loss、backward 和 optimizer

这次使用 `CrossEntropyLoss`。它会比较四个动作 logits 和正确 action 标签，预测越偏离正确动作，loss 通常越大。

`loss.backward()` 会从 loss 开始反向传播，计算每个模型参数应该往哪个方向调整，也就是计算梯度。它只负责算梯度，不会直接修改参数。

optimizer 是根据梯度更新参数的工具。这次使用 Adam。`optimizer.step()` 才会真正更新参数；每个 batch 开始前还要调用 `optimizer.zero_grad()`，清除上一次留下的梯度。

## Epoch 是什么

一个 epoch 表示模型完整看过一次训练集。这次训练 100 个 epoch，每个 epoch 记录平均 train loss。

训练开始时 loss 是 `1.342372`，最后下降到 `0.169001`。loss 持续下降说明模型越来越能拟合训练数据，但 loss 下降不自动代表模型在新地图上一定表现更好，所以还需要测试集评估。

## 这次结果怎么理解

测试集 single-step action accuracy 是 `0.8763`。Stage 3 中表现最好的 Logistic Regression accuracy 是 `0.8557`，这次 PyTorch BC 高了 `0.0206`。

这个结果说明当前 MLP 在相同数据划分上的单步动作预测表现更好一些，但不能只因为用了神经网络就认为它一定更好。模型结构、随机种子、训练参数和数据特征都会影响结果，传统 baseline 仍然是重要的比较参照。

## 为什么还不能说模型会完整导航

当前测试方式会从专家数据中拿出一个 state，让模型预测这一小步 action。这就是 single-step action prediction。

完整导航 rollout 则需要让模型从起点开始，连续使用自己的预测动作移动，直到到达目标、失败或超过步数限制。模型一旦走错，可能进入专家数据中很少出现的状态，之后错误还会继续累积。

这次没有实现 rollout，也没有进入强化学习。因此当前只能说模型对测试样本的单步动作预测 accuracy 是 `0.8763`，不能说它已经会完整导航。

## Stage 3 和 Stage 4 的关系

Stage 3 的 sklearn baseline 和 Stage 4 的 PyTorch BC 学习的是同一个 `state -> action` 分类任务，也使用相同的八个特征、标签和 `map_id` 数据划分。

区别主要在模型和训练方式。Stage 3 使用 KNN、Logistic Regression 和 Decision Tree；Stage 4 使用 MLP，通过 forward、loss、backward 和 optimizer 更新网络参数。因为任务和数据划分相同，两阶段的 accuracy 才有比较意义。
