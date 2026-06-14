#include "astar.h"

#include <algorithm>
#include <cstdlib>
#include <queue>
#include <unordered_map>
#include <utility>


bool Point::operator==(const Point& other) const {
    return x == other.x && y == other.y;
}


std::size_t PointHash::operator()(const Point& point) const {
    const std::size_t x_hash = std::hash<int>{}(point.x);
    const std::size_t y_hash = std::hash<int>{}(point.y);
    return x_hash ^ (y_hash << 1);
}


Grid::Grid(
    int width_value,
    int height_value,
    std::unordered_set<Point, PointHash> obstacle_values,
    Point start_value,
    Point goal_value
)
    : width(width_value),
      height(height_value),
      obstacles(std::move(obstacle_values)),
      start(start_value),
      goal(goal_value) {}


bool Grid::in_bounds(const Point& point) const {
    return 0 <= point.x && point.x < width && 0 <= point.y && point.y < height;
}


bool Grid::passable(const Point& point) const {
    return obstacles.find(point) == obstacles.end();
}


std::vector<Point> Grid::neighbors(const Point& point) const {
    const std::vector<Point> candidates = {
        {point.x + 1, point.y},
        {point.x - 1, point.y},
        {point.x, point.y + 1},
        {point.x, point.y - 1},
    };

    std::vector<Point> result;
    for (const Point& candidate : candidates) {
        if (in_bounds(candidate) && passable(candidate)) {
            result.push_back(candidate);
        }
    }
    return result;
}


int manhattan(const Point& a, const Point& b) {
    return std::abs(a.x - b.x) + std::abs(a.y - b.y);
}


namespace {

struct FrontierNode {
    int priority;
    Point point;
};


struct FrontierNodeGreater {
    bool operator()(const FrontierNode& left, const FrontierNode& right) const {
        if (left.priority != right.priority) {
            return left.priority > right.priority;
        }
        if (left.point.x != right.point.x) {
            return left.point.x > right.point.x;
        }
        return left.point.y > right.point.y;
    }
};

}  // namespace


std::vector<Point> astar_search(const Grid& grid) {
    std::priority_queue<
        FrontierNode,
        std::vector<FrontierNode>,
        FrontierNodeGreater
    > frontier;
    frontier.push({0, grid.start});

    std::unordered_map<Point, Point, PointHash> came_from;
    std::unordered_map<Point, int, PointHash> cost_so_far;
    cost_so_far[grid.start] = 0;

    while (!frontier.empty()) {
        const Point current = frontier.top().point;
        frontier.pop();

        if (current == grid.goal) {
            break;
        }

        for (const Point& next : grid.neighbors(current)) {
            const int new_cost = cost_so_far[current] + 1;
            const auto known_cost = cost_so_far.find(next);

            if (known_cost == cost_so_far.end() || new_cost < known_cost->second) {
                cost_so_far[next] = new_cost;
                const int priority = new_cost + manhattan(next, grid.goal);
                frontier.push({priority, next});
                came_from[next] = current;
            }
        }
    }

    if (cost_so_far.find(grid.goal) == cost_so_far.end()) {
        return {};
    }

    std::vector<Point> path;
    Point current = grid.goal;
    path.push_back(current);

    while (!(current == grid.start)) {
        current = came_from.at(current);
        path.push_back(current);
    }

    std::reverse(path.begin(), path.end());
    return path;
}
