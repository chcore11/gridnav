import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from env.grid_world import GridWorld
from planning_astar.astar import astar_search


def test_demo_path():
    obstacles = {
        (3, 0), (3, 1), (3, 2), (3, 3),
        (3, 4), (3, 5), (3, 6),
        (6, 3), (6, 4), (6, 5), (6, 6),
    }

    world = GridWorld(
        width=10,
        height=10,
        obstacles=obstacles,
        start=(0, 0),
        goal=(9, 9),
    )

    path = astar_search(world)

    assert path is not None
    assert path[0] == world.start
    assert path[-1] == world.goal
    assert len(path) == 19
    assert all(nxt in world.neighbors(current) for current, nxt in zip(path, path[1:]))


def test_no_path():
    world = GridWorld(
        width=3,
        height=3,
        obstacles={(1, 0), (1, 1), (1, 2)},
        start=(0, 1),
        goal=(2, 1),
    )

    assert astar_search(world) is None


def main():
    test_demo_path()
    test_no_path()
    print("A* simple assertion tests passed.")


if __name__ == "__main__":
    main()
