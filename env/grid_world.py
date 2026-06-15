import numpy as np


# GridWorld 负责保存二维网格地图，并提供 A*、数据生成和策略评估都会用到的基础地图操作。
class GridWorld:
    def __init__(self, width=10, height=10, obstacles=None, start=(0, 0), goal=(9, 9)):
        self.width = width  # 保存网格宽度，后面用于判断 x 坐标是否越界
        self.height = height  # 保存网格高度，后面用于判断 y 坐标是否越界
        self.start = start  # 保存起点坐标，坐标格式为 (x, y)
        self.goal = goal  # 保存终点坐标，路径规划的目标是到达这里
        # 将障碍物坐标保存为集合，便于快速判断某个位置是否被障碍占用。
        self.obstacles = set(obstacles) if obstacles else set()

    def in_bounds(self, node):
        x, y = node  # 拆出节点的横坐标 x 和纵坐标 y
        # x、y 都从 0 开始，必须分别小于网格宽度和高度才没有越界。
        return 0 <= x < self.width and 0 <= y < self.height

    def passable(self, node):
        # 不在障碍物集合中的节点才可以通行。
        return node not in self.obstacles

    def neighbors(self, node):
        x, y = node  # 以当前节点为中心，准备生成相邻位置
        # 每次只改变一个坐标，因此生成的是右、左、下、上四个方向的邻居。
        candidates = [
            (x + 1, y),
            (x - 1, y),
            (x, y + 1),
            (x, y - 1),
        ]
        # 先过滤掉越界位置，再过滤掉障碍位置，只返回真正可以移动到的邻居。
        return [p for p in candidates if self.in_bounds(p) and self.passable(p)]

    def to_array(self, path=None):
        # NumPy 数组形状按 (行数, 列数) 指定，所以这里先写 height，再写 width。
        grid = np.zeros((self.height, self.width), dtype=int)

        for x, y in self.obstacles:
            # 数组索引格式是 [行, 列]，对应坐标中的 [y, x]；数值 1 表示障碍物。
            grid[y, x] = 1

        if path:
            for x, y in path:
                # 起点和终点稍后会使用单独数值标记，因此这里不把它们覆盖成普通路径。
                if (x, y) != self.start and (x, y) != self.goal:
                    grid[y, x] = 2  # 数值 2 表示路径经过的位置

        sx, sy = self.start  # 拆出起点的 x、y 坐标
        gx, gy = self.goal  # 拆出终点的 x、y 坐标
        grid[sy, sx] = 3  # 数值 3 表示起点，数组索引仍然按 [y, x] 排列
        grid[gy, gx] = 4  # 数值 4 表示终点

        return grid  # 返回可用于显示、保存或后续处理的网格数组
