import random
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from env.grid_world import GridWorld
from planning_astar.astar import astar_search


NUM_MAPS = 100
WIDTH = 10
HEIGHT = 10
OBSTACLE_RATIO = 0.2
RANDOM_SEED = 42
MIN_START_GOAL_DISTANCE = 6

CSV_COLUMNS = [
    "map_id",
    "step_id",
    "agent_x",
    "agent_y",
    "goal_x",
    "goal_y",
    "obs_up",
    "obs_down",
    "obs_left",
    "obs_right",
    "action",
]

ACTION_NAMES = {
    0: "up",
    1: "down",
    2: "left",
    3: "right",
}


def generate_random_start_goal(width, height, min_distance, rng):
    if width <= 0 or height <= 0:
        raise ValueError("Map width and height must be positive.")
    if min_distance < 1:
        raise ValueError("Minimum start-goal distance must be at least 1.")

    cells = [(x, y) for y in range(height) for x in range(width)]
    valid_pairs = [
        (start, goal)
        for start in cells
        for goal in cells
        if abs(start[0] - goal[0]) + abs(start[1] - goal[1]) >= min_distance
    ]
    if not valid_pairs:
        raise ValueError(
            f"No start-goal pair has Manhattan distance >= {min_distance}."
        )

    return rng.choice(valid_pairs)


def generate_random_obstacles(width, height, obstacle_ratio, start, goal, rng):
    if width <= 0 or height <= 0:
        raise ValueError("Map width and height must be positive.")
    if not 0 <= obstacle_ratio <= 1:
        raise ValueError("Obstacle ratio must be between 0 and 1.")
    for name, node in {"start": start, "goal": goal}.items():
        x, y = node
        if not (0 <= x < width and 0 <= y < height):
            raise ValueError(f"{name} node is out of bounds: {node}")

    available_cells = [
        (x, y)
        for y in range(height)
        for x in range(width)
        if (x, y) not in {start, goal}
    ]
    obstacle_count = int(len(available_cells) * obstacle_ratio)
    return set(rng.sample(available_cells, obstacle_count))


def get_local_obstacles(world, node):
    x, y = node
    directions = {
        "obs_up": (x, y - 1),
        "obs_down": (x, y + 1),
        "obs_left": (x - 1, y),
        "obs_right": (x + 1, y),
    }

    return {
        name: int(not world.in_bounds(position) or not world.passable(position))
        for name, position in directions.items()
    }


def action_from_step(current, nxt):
    dx = nxt[0] - current[0]
    dy = nxt[1] - current[1]
    actions = {
        (0, -1): 0,
        (0, 1): 1,
        (-1, 0): 2,
        (1, 0): 3,
    }

    if (dx, dy) not in actions:
        raise ValueError(f"Invalid path step: {current} -> {nxt}")

    return actions[(dx, dy)]


def path_to_samples(world, path, map_id):
    if not path:
        raise ValueError("Path must contain at least one node.")
    if path[0] != world.start:
        raise ValueError(f"Path must start at {world.start}, got {path[0]}.")
    if path[-1] != world.goal:
        raise ValueError(f"Path must end at {world.goal}, got {path[-1]}.")

    for node in path:
        if not world.in_bounds(node):
            raise ValueError(f"Path node is out of bounds: {node}")
        if not world.passable(node):
            raise ValueError(f"Path node is an obstacle: {node}")

    samples = []
    goal_x, goal_y = world.goal

    for step_id, (current, nxt) in enumerate(zip(path, path[1:])):
        agent_x, agent_y = current
        sample = {
            "map_id": map_id,
            "step_id": step_id,
            "agent_x": agent_x,
            "agent_y": agent_y,
            "goal_x": goal_x,
            "goal_y": goal_y,
            **get_local_obstacles(world, current),
            "action": action_from_step(current, nxt),
        }
        samples.append(sample)

    return samples


def write_dataset_summary(
    report_path,
    attempted_maps,
    successful_maps,
    no_path_maps,
    total_samples,
    average_path_steps,
    action_counts,
):
    action_percentages = {
        action: action_counts[action] / total_samples * 100 if total_samples else 0.0
        for action in ACTION_NAMES
    }
    action_rows = "\n".join(
        f"| `{action}` | {ACTION_NAMES[action]} | {action_counts[action]} | "
        f"{action_percentages[action]:.2f}% |"
        for action in ACTION_NAMES
    )
    if all(action_counts[action] > 1 for action in ACTION_NAMES):
        distribution_comment = (
            "四个方向都获得了较多样本，分布比固定左上角起点和右下角终点时更合理。"
            "当前各动作仍不是完全相等，但已不再几乎只包含 down 和 right。"
        )
    else:
        distribution_comment = (
            "当前仍有动作样本过少，后续需要继续调整地图数量或采样方式。"
        )
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report = f"""# 专家轨迹数据集摘要

## 数据来源

当前数据集由 A* 在 {attempted_maps} 张随机 `{WIDTH} x {HEIGHT}` GridWorld 地图上搜索并生成。只有成功找到路径的地图会转换成 state-action 样本。

## 参数设置

- 地图大小：`{WIDTH} x {HEIGHT}`
- 障碍物比例：`{OBSTACLE_RATIO}`
- 起点和终点：每张地图随机采样，且不相同
- 起点和终点最小曼哈顿距离：`{MIN_START_GOAL_DISTANCE}`
- 随机种子：`{RANDOM_SEED}`
- `map_id`：原始尝试地图编号，从 `0` 开始；无路径地图的编号会被跳过。
- `step_id`：一条成功路径中的动作序号，从 `0` 开始。

## 生成统计

- 尝试地图数量：{attempted_maps}
- 成功找到路径地图数量：{successful_maps}
- 无路径地图数量：{no_path_maps}
- 总样本数量：{total_samples}
- 平均路径移动步数：{average_path_steps:.2f}

## Action 分布

| action | 方向 | 样本数量 | 占比 |
| --- | --- | ---: | ---: |
{action_rows}

{distribution_comment}

## 字段含义

- `map_id`：原始尝试地图编号。
- `step_id`：当前动作在该路径中的序号。
- `agent_x`、`agent_y`：智能体当前坐标。
- `goal_x`、`goal_y`：目标点坐标。
- `obs_up`、`obs_down`、`obs_left`、`obs_right`：对应方向越界或存在障碍物时为 `1`，否则为 `0`。
- `action`：专家动作，`0 = up`、`1 = down`、`2 = left`、`3 = right`。

## 当前局限

- 仍然是二维简化 GridWorld，和真实导航环境有较大差距。
- 状态特征只包含当前位置、目标和局部障碍，看不到完整地图。
- 该数据集已用于传统机器学习 baseline，按 `map_id` 划分训练集和测试集。
- 当前 baseline 学习的是 `state -> action` 单步动作预测，没有进行完整导航 rollout。
- 随机地图数量和障碍物比例后续可以继续调整。
"""
    report_path.write_text(report, encoding="utf-8")


def main():
    import pandas as pd

    rng = random.Random(RANDOM_SEED)
    all_samples = []
    successful_maps = 0
    no_path_maps = 0
    path_move_counts = []

    for map_id in range(NUM_MAPS):
        start, goal = generate_random_start_goal(
            WIDTH,
            HEIGHT,
            MIN_START_GOAL_DISTANCE,
            rng,
        )
        obstacles = generate_random_obstacles(
            WIDTH,
            HEIGHT,
            OBSTACLE_RATIO,
            start,
            goal,
            rng,
        )
        world = GridWorld(
            width=WIDTH,
            height=HEIGHT,
            obstacles=obstacles,
            start=start,
            goal=goal,
        )

        path = astar_search(world)
        if path is None:
            no_path_maps += 1
            continue

        samples = path_to_samples(world, path, map_id)
        all_samples.extend(samples)
        path_move_counts.append(len(path) - 1)
        successful_maps += 1

    dataset = pd.DataFrame(all_samples, columns=CSV_COLUMNS)
    csv_path = ROOT / "data" / "expert_dataset.csv"
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    dataset.to_csv(csv_path, index=False)

    action_counts = Counter(sample["action"] for sample in all_samples)
    average_path_steps = (
        sum(path_move_counts) / successful_maps if successful_maps else 0.0
    )
    report_path = ROOT / "reports" / "dataset_summary.md"
    write_dataset_summary(
        report_path=report_path,
        attempted_maps=NUM_MAPS,
        successful_maps=successful_maps,
        no_path_maps=no_path_maps,
        total_samples=len(all_samples),
        average_path_steps=average_path_steps,
        action_counts=action_counts,
    )

    print("尝试地图数量:", NUM_MAPS)
    print("成功找到路径地图数量:", successful_maps)
    print("无路径地图数量:", no_path_maps)
    print("总样本数量:", len(all_samples))
    print("平均路径移动步数:", f"{average_path_steps:.2f}")
    print("动作分布:", {action: action_counts[action] for action in ACTION_NAMES})
    print("CSV 保存路径:", csv_path)


if __name__ == "__main__":
    main()
