import numpy as np


class GridWorld:
    def __init__(self, width=10, height=10, obstacles=None, start=(0, 0), goal=(9, 9)):
        self.width = width
        self.height = height
        self.start = start
        self.goal = goal
        self.obstacles = set(obstacles) if obstacles else set()

    def in_bounds(self, node):
        x, y = node
        return 0 <= x < self.width and 0 <= y < self.height

    def passable(self, node):
        return node not in self.obstacles

    def neighbors(self, node):
        x, y = node
        candidates = [
            (x + 1, y),
            (x - 1, y),
            (x, y + 1),
            (x, y - 1),
        ]
        return [p for p in candidates if self.in_bounds(p) and self.passable(p)]

    def to_array(self, path=None):
        grid = np.zeros((self.height, self.width), dtype=int)

        for x, y in self.obstacles:
            grid[y, x] = 1

        if path:
            for x, y in path:
                if (x, y) != self.start and (x, y) != self.goal:
                    grid[y, x] = 2

        sx, sy = self.start
        gx, gy = self.goal
        grid[sy, sx] = 3
        grid[gy, gx] = 4

        return grid
