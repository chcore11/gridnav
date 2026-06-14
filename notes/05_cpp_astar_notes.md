# C++ A* 学习笔记

## 为什么机器人系统常用 C++

我理解机器人系统常用 C++，主要是因为它运行效率高，也能更直接地控制内存、数据结构和硬件接口。路径规划、控制和传感器处理经常需要稳定、快速地反复运行，所以 C++ 很常见。

Python 更适合快速验证想法。这个项目先用 Python 把 A* 逻辑学清楚，再用 C++ 重写，可以让我把注意力放在两种语言如何表达同一个算法上。

## struct 和 class

`struct` 可以把几个相关数据组合成一个新类型。这次 `Point` 把 `x` 和 `y` 坐标放在一起，表示一个网格位置；`FrontierNode` 把优先级和位置放在一起，表示优先队列中的待搜索节点。

`class` 不只保存数据，还可以把相关操作放在一起。`Grid` 保存地图宽高、障碍物、起点和终点，同时提供 `in_bounds()`、`passable()` 和 `neighbors()`。

在 C++ 中，`struct` 默认成员是 public，`class` 默认成员是 private。这次选择主要是为了让代码表达更直观，不是因为它们能做完全不同的事情。

## vector、pair、priority_queue 和 unordered_map

`vector` 是可以动态增长的连续容器。这次用它保存邻居列表和最终路径，作用接近 Python 的 list。

`pair` 可以临时组合两个值，例如优先级和坐标。Python A* 的 heap 中直接放 `(priority, node)`，就很像 C++ 的 `pair`。这次为了让字段含义更清楚，我使用了 `FrontierNode` struct，没有直接使用 `pair`。

`priority_queue` 是优先队列。普通队列按进入顺序取元素，A* 的 open set 则需要优先取出 `f(n)=g(n)+h(n)` 最小的节点。C++ 默认的 `priority_queue` 会先取最大值，所以代码中还定义了比较规则，让较小优先级先被取出。

`unordered_map` 使用哈希快速查找键值。这次：

- `came_from` 保存一个节点是从哪个父节点来的。
- `cost_so_far` 保存从起点到节点的当前最小真实代价。

`map` 也能保存键值，但它会按键排序，底层通常是树；`unordered_map` 不保证顺序，底层通常是哈希表。因为 `Point` 是自定义类型，所以我还需要写 `PointHash`，告诉 `unordered_map` 和 `unordered_set` 如何计算坐标的哈希值。

## A* 中三个重要部分

Open set 是还需要继续搜索的节点集合。在代码中，它是 `priority_queue` 类型的 `frontier`。

`cost_so_far` 对应 `g(n)`，记录从 start 到当前节点已经走了多少步。如果后来找到更短的走法，就更新这个值。

`came_from` 记录父节点。找到 goal 后，代码从 goal 开始沿着父节点回到 start，再把路径反转，得到从 start 到 goal 的最终路径。

曼哈顿距离对应 `h(n)`。因为当前 GridWorld 只能上下左右移动，所以：

```text
h(n) = abs(x - goal_x) + abs(y - goal_y)
```

## C++ A* 和 Python A* 的关系

两份实现的算法逻辑基本相同：

1. 把 start 放进 frontier。
2. 每次取出优先级最低的节点。
3. 检查上下左右可通行邻居。
4. 计算新的 `g(n)` 和 `f(n)`。
5. 更新 `cost_so_far` 和 `came_from`。
6. 找到 goal 后回溯路径。

Demo 也使用了与 Python 相同的 `10 x 10` 地图、障碍物、起点和终点。两边都输出 19 个路径节点，也就是 18 个移动步数。

## 为什么 C++ 更啰嗦

C++ 需要显式写出类型、头文件、函数声明、自定义哈希、优先队列比较规则和 CMake 构建文件，所以同一个 A* 会比 Python 长很多。

这些额外内容让我更清楚地看到模块之间的接口，也更接近工程系统中需要考虑的编译、数据结构和类型约束。代码更啰嗦，但每一部分的责任也更明确。

## 当前阶段边界

Stage 6 只是用 C++ 重新实现 A* 路径规划。本阶段没有训练数据、loss、reward 或策略更新，因此不是机器学习，也不是强化学习。
