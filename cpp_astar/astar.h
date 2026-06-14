#ifndef GRIDNAV_CPP_ASTAR_ASTAR_H
#define GRIDNAV_CPP_ASTAR_ASTAR_H

#include <cstddef>
#include <unordered_set>
#include <vector>


struct Point {
    int x;
    int y;

    bool operator==(const Point& other) const;
};


struct PointHash {
    std::size_t operator()(const Point& point) const;
};


class Grid {
public:
    Grid(
        int width,
        int height,
        std::unordered_set<Point, PointHash> obstacles,
        Point start,
        Point goal
    );

    bool in_bounds(const Point& point) const;
    bool passable(const Point& point) const;
    std::vector<Point> neighbors(const Point& point) const;

    int width;
    int height;
    std::unordered_set<Point, PointHash> obstacles;
    Point start;
    Point goal;
};


int manhattan(const Point& a, const Point& b);
std::vector<Point> astar_search(const Grid& grid);

#endif
