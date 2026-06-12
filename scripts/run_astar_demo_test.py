import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

import matplotlib.pyplot as plt

from env.grid_world import GridWorld
from planning_astar.astar import astar_search


def main():
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

    if path is None:
        print("No path found.")
        return

    print("Path length:", len(path))
    print("Path:")
    print(path)

    grid = world.to_array(path)

    plt.figure(figsize=(6, 6))
    plt.imshow(grid, origin="upper")
    plt.title("A* GridWorld Demo")
    plt.xticks(range(world.width))
    plt.yticks(range(world.height))
    plt.grid(True)

    out = ROOT / "figures" / "astar_demo.png"
    plt.savefig(out, dpi=150)
    print("Saved figure:", out)


if __name__ == "__main__":
    main()
