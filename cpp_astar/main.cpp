#include "astar.h"

#include <cstdlib>
#include <iostream>
#include <unordered_set>
#include <vector>


bool validate_path(const Grid& grid, const std::vector<Point>& path) {
    if (path.empty() || !(path.front() == grid.start) || !(path.back() == grid.goal)) {
        return false;
    }

    for (const Point& point : path) {
        if (!grid.in_bounds(point) || !grid.passable(point)) {
            return false;
        }
    }

    for (std::size_t index = 1; index < path.size(); ++index) {
        if (manhattan(path[index - 1], path[index]) != 1) {
            return false;
        }
    }

    return true;
}


void print_path(const std::vector<Point>& path) {
    for (std::size_t index = 0; index < path.size(); ++index) {
        const Point& point = path[index];
        std::cout << "(" << point.x << ", " << point.y << ")";
        if (index + 1 < path.size()) {
            std::cout << " -> ";
        }
    }
    std::cout << "\n";
}


int main() {
    const std::unordered_set<Point, PointHash> obstacles = {
        {3, 0}, {3, 1}, {3, 2}, {3, 3},
        {3, 4}, {3, 5}, {3, 6},
        {6, 3}, {6, 4}, {6, 5}, {6, 6},
    };
    const Grid grid(10, 10, obstacles, {0, 0}, {9, 9});

    const std::vector<Point> path = astar_search(grid);
    if (path.empty()) {
        std::cout << "No path found.\n";
        return 1;
    }

    const bool path_is_valid = validate_path(grid, path);
    std::cout << "Path:\n";
    print_path(path);
    std::cout << "Path nodes: " << path.size() << "\n";
    std::cout << "Movement steps: " << path.size() - 1 << "\n";
    std::cout << "Path starts at start: " << std::boolalpha
              << (path.front() == grid.start) << "\n";
    std::cout << "Path ends at goal: " << std::boolalpha
              << (path.back() == grid.goal) << "\n";
    std::cout << "Path validity check: " << std::boolalpha << path_is_valid << "\n";

    return path_is_valid ? 0 : 1;
}
