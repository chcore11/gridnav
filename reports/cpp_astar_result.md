# Stage 6：C++ A* 工程实现结果

## 阶段目标

用 C++17 重新实现 GridWorld 上的 A* 路径规划，理解路径规划算法在 C++ 工程中的数据结构、文件组织、编译和运行方式。

本阶段不是机器学习，不是行为克隆，也不是强化学习。

## 文件结构

- `cpp_astar/astar.h`：定义 `Point`、`PointHash`、`Grid` 和 A* 函数接口。
- `cpp_astar/astar.cpp`：实现 Grid 邻居查询、曼哈顿距离、优先队列搜索和路径回溯。
- `cpp_astar/main.cpp`：构造 demo 地图、调用 A*、打印路径并进行合法性检查。
- `cpp_astar/CMakeLists.txt`：设置 C++17 并生成 `cpp_astar_demo`。

## 编译与运行

```bash
mkdir -p cpp_astar/build
cd cpp_astar/build
cmake ..
cmake --build .
./cpp_astar_demo
```

CMake 配置和编译成功，可执行文件运行成功。

## Demo 地图

- 地图大小：`10 x 10`
- 起点：`(0, 0)`
- 终点：`(9, 9)`
- 四方向移动：up、down、left、right
- 障碍物与 `scripts/run_astar_demo.py` 完全相同
- 启发式函数：曼哈顿距离

## 输出路径

```text
(0, 0) -> (0, 1) -> (0, 2) -> (0, 3) -> (0, 4) -> (0, 5)
-> (0, 6) -> (0, 7) -> (0, 8) -> (0, 9) -> (1, 9) -> (2, 9)
-> (3, 9) -> (4, 9) -> (5, 9) -> (6, 9) -> (7, 9) -> (8, 9)
-> (9, 9)
```

- 路径节点数：`19`
- 移动步数：`18`
- Python A* demo 路径节点数：`19`
- Python A* demo 移动步数：`18`

## 合法性检查

- 路径起点是 start：通过
- 路径终点是 goal：通过
- 每一步都是上下左右相邻移动：通过
- 所有节点都在地图边界内：通过
- 路径不穿过障碍物：通过

如果 A* 找不到路径，`main.cpp` 会输出 `No path found.`。

## 与 Python A* 的关系

Python A* 用于早期学习、快速验证算法，并为后续监督学习生成专家路径。C++ A* 使用相同的 `frontier`、`came_from`、`cost_so_far`、曼哈顿距离和路径回溯逻辑，但需要显式定义类型、哈希函数、优先队列比较方式和构建流程。

C++ 版本不是为了替代 Python 版本，而是为了理解路径规划模块在更接近机器人工程系统的语言中如何组织。
