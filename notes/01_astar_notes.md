# A* 路径规划笔记

## 1. 当前阶段目标

本阶段主要理解两件事：

1. GridWorld 如何表示一个二维导航环境；
2. A* 如何在已知地图中从起点搜索到目标点。

这一阶段还不是机器学习，而是在为后续专家轨迹数据生成、机器学习 baseline、行为克隆和强化学习打基础。

---

## 2. GridWorld 如何表示地图

`GridWorld` 使用 `(x, y)` 坐标表示地图位置。

* `x` 表示列，向右增大；
* `y` 表示行，向下增大；
* 左上角是 `(0, 0)`；
* numpy 数组访问顺序是 `grid[y, x]`。

当前环境中保存了：

* `width` / `height`：地图大小；
* `start`：起点；
* `goal`：目标点；
* `obstacles`：障碍物集合。

障碍物使用 `set` 存储，便于快速判断某个位置是否可通行：

```python
node not in self.obstacles
```

`to_array()` 会把地图转成数组，用于后续可视化。

---

## 3. neighbors() 的作用

`neighbors(node)` 用来返回当前节点上下左右所有可通行的位置。

四个候选方向是：

```text
(x + 1, y)  right
(x - 1, y)  left
(x, y + 1)  down
(x, y - 1)  up
```

然后通过：

```python
self.in_bounds(p)
self.passable(p)
```

过滤掉越界点和障碍物点。

当前代码还没有显式 action 编码，但动作已经隐含在相邻坐标变化中。后续生成专家数据时，需要把路径中的坐标变化转换成动作标签。

---

## 4. A* 核心逻辑

A* 的核心公式是：

```text
f(n) = g(n) + h(n)
```

含义：

* `g(n)`：从起点到当前节点的真实代价；
* `h(n)`：从当前节点到目标点的估计代价；
* `f(n)`：综合优先级，越小越优先搜索。

在本项目代码中：

```python
new_cost = cost_so_far[current] + 1
priority = new_cost + manhattan(nxt, goal)
```

对应关系是：

```text
cost_so_far[nxt]        -> g(n)
manhattan(nxt, goal)    -> h(n)
priority                -> f(n)
```

本项目使用曼哈顿距离作为启发式函数：

```python
abs(ax - bx) + abs(ay - by)
```

因为 GridWorld 只能上下左右移动，不能斜着走，所以曼哈顿距离比较适合。

---

## 5. frontier / cost_so_far / came_from

### frontier

`frontier` 是待搜索节点集合，也可以理解为 open set。

代码中用 `heapq` 实现优先队列：

```python
heapq.heappush(frontier, (priority, nxt))
```

这样每次都会优先取出 `priority` 最小的节点。

### cost_so_far

`cost_so_far` 记录从起点到某个节点的当前最小真实代价，也就是 `g(n)`。

如果找到一条更便宜的路径，就更新它：

```python
if nxt not in cost_so_far or new_cost < cost_so_far[nxt]:
```

### came_from

`came_from` 记录每个节点是从哪个节点来的。

例如：

```python
came_from[nxt] = current
```

意思是当前较优路径中，`nxt` 的前一个节点是 `current`。

---

## 6. 路径如何恢复

A* 搜索过程中没有一直保存完整路径，而是保存父节点关系。

找到目标点后，从 `goal` 开始不断查 `came_from[current]`，一路反向回到 `start`：

```python
while current is not None:
    path.append(current)
    current = came_from[current]
```

此时路径顺序是从目标点到起点，所以最后需要：

```python
path.reverse()
```

得到从起点到终点的路径。

---

## 7. 当前 demo 运行结果理解

当前 demo 的地图大小是 `10 x 10`，起点是 `(0, 0)`，目标点是 `(9, 9)`。

运行结果：

```text
Path length: 19
```

这里的 19 表示路径中有 19 个坐标点，包括起点和终点。

实际移动步数是：

```text
19 - 1 = 18
```

当前路径大致是先沿左侧向下走，再沿底部向右走到目标点。

生成的可视化图片保存到：

```text
figures/astar_demo.png
```

---

## 8. A* 为什么不是机器学习

A* 不是机器学习算法。

它不需要训练数据，不计算 loss，也不会通过梯度更新参数。它是在地图已知的情况下，根据搜索规则和启发式函数寻找路径。

所以 A* 更像是“规则规划”，不是“数据学习”。

---

## 9. A* 为什么可以生成专家数据

虽然 A* 不是机器学习，但它可以在已知地图中生成合理路径。

后续可以把路径拆成多个样本：

```text
state -> action
```

例如：

```text
(0, 0) -> (0, 1)  对应 down
(0, 9) -> (1, 9)  对应 right
```

这些样本可以作为专家演示数据，用于训练机器学习 baseline 或 PyTorch 行为克隆模型。

---

## 10. 当前代码的不足

当前阶段代码还能继续完善：

1. `GridWorld` 还没有 `reset()` 和 `step(action)`；
2. action 还没有显式编码；
3. 当前 demo 只有一张固定地图；
4. 后续需要随机生成多张地图；
5. 后续需要把 A* 路径转换成专家轨迹数据集。

---

## 11. 当前阶段小结

GridWorld 负责表示地图、边界、障碍物、起点和目标点。

A* 负责在已知地图中搜索路径，核心是：

```text
f(n) = g(n) + h(n)
```

其中 `g(n)` 是真实代价，`h(n)` 是估计代价。

这一阶段的意义不是直接做机器学习，而是先把环境和路径规划逻辑弄清楚。后续专家数据、机器学习 baseline、行为克隆和强化学习都要建立在这个基础上。
