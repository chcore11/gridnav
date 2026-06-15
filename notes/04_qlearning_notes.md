# Q-learning 学习笔记

## 当前状态

我完成了固定地图上的最小 tabular Q-learning 工程。训练脚本会让智能体从 start 出发，通过 epsilon-greedy 自己选择动作，再根据 reward 更新 Q-table。它没有读取 A* 路径或专家 action，也没有使用 PyTorch 或 DQN。

当前 state 只使用智能体坐标 `(x, y)`，所以这张 Q-table 只适用于固定地图。训练结束后使用纯 greedy policy 重复评估 100 次，成功率是 `100%`，平均成功步数是 `10`。

## 我对强化学习和 Q-learning 的理解

强化学习不是先准备好每个 state 的正确 action 答案，而是让智能体和环境互动。智能体执行动作后，环境返回 reward。智能体要从多次试错中学习哪些动作能带来更高的长期回报。

Q-learning 是一种不需要提前知道环境模型的强化学习方法。这里使用一张 Q-table 保存每个状态下四个动作的价值。`Q(s,a)` 越大，表示我目前认为在状态 `s` 选择动作 `a` 越值得。

Reward 和专家答案不同。专家 action 会直接告诉模型“这个状态应该选哪个动作”；reward 只评价动作产生的结果。这次到达 goal 得 `+100`，碰撞得 `-10`，普通移动得 `-1`。智能体需要自己从这些反馈中找到好策略。

## Episode 和探索

一个 episode 是一次从 start 开始的完整尝试。到达 goal 时成功结束；如果一直没有到达，超过 `max_steps` 后结束。每个 episode 都记录 total reward、是否成功和走过的 steps。

Epsilon-greedy 用来平衡 exploration 和 exploitation：

- exploration：以 epsilon 的概率随机选动作，尝试还不了解的路线。
- exploitation：选择当前 Q 值最大的动作，利用已经学到的经验。

训练开始时 epsilon 是 `1.0`，随机探索很多；每个 episode 后乘以 `0.995`，最低降到 `0.05`。评估时 epsilon 为 `0`，只使用 greedy policy。

## Q-table 更新

更新公式是：

`Q(s,a) = Q(s,a) + alpha * [reward + gamma * max Q(s',a') - Q(s,a)]`

我把中括号理解成“新观察到的目标价值”和“旧判断”之间的差。每走一步，只把这次差距的一部分写回 Q-table。

- alpha 是学习率。这次使用 `0.1`，表示每次用新信息修正旧 Q 值的一部分，不会一次全部覆盖。
- gamma 是折扣因子。这次使用 `0.95`，表示重视未来 reward，但距离越远的 reward 影响会逐渐减小。

## 和 BC、BC Rollout 的区别

Stage 4 Behavior Cloning 是监督学习，使用专家 action 作为标签。Stage 4.5 BC Rollout Evaluation 只评估已经训练好的 BC policy，不使用 reward，也不训练或更新策略，因此不是强化学习。

Stage 5 Q-learning 没有专家 action。它通过 epsilon-greedy 选择动作、接收 reward，并在每个 episode 中更新 Q-table。训练结果说明它在当前固定地图上学到了可行路径，但还不能说明它能泛化到新的 goal 或障碍布局。

## 当前边界

- 当前只实现固定地图上的 tabular Q-learning。
- state 只有坐标，没有包含 goal 或障碍布局。
- Q-table 只保存在训练脚本运行期间，没有单独保存模型文件。
- Stage 6 C++ A* 虽然已提前实现，但本次没有进入 Stage 6 学习或修改。
