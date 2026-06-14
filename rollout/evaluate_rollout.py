import random
import sys
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from bc_pytorch.train_bc import BehaviorCloningMLP
from data.generate_expert_data import (
    HEIGHT,
    MIN_START_GOAL_DISTANCE,
    NUM_MAPS,
    OBSTACLE_RATIO,
    RANDOM_SEED as DATA_RANDOM_SEED,
    WIDTH,
    generate_random_obstacles,
    generate_random_start_goal,
    get_local_obstacles,
    path_to_samples,
)
from env.grid_world import GridWorld
from planning_astar.astar import astar_search


CHECKPOINT_PATH = ROOT / "checkpoints" / "bc_policy.pt"
DATA_PATH = ROOT / "data" / "expert_dataset.csv"
REPORT_PATH = ROOT / "reports" / "rollout_result.md"
FIGURE_PATH = ROOT / "figures" / "rollout_success_summary.png"

EXPECTED_FEATURE_COLUMNS = [
    "agent_x",
    "agent_y",
    "goal_x",
    "goal_y",
    "obs_up",
    "obs_down",
    "obs_left",
    "obs_right",
]
ACTION_DELTAS = {
    0: (0, -1),
    1: (0, 1),
    2: (-1, 0),
    3: (1, 0),
}
FAILURE_REASONS = [
    "out_of_bounds",
    "hit_obstacle",
    "loop_detected",
    "max_steps_exceeded",
]


def load_policy():
    if not CHECKPOINT_PATH.exists():
        raise FileNotFoundError(
            "未找到 checkpoints/bc_policy.pt，请先运行："
            "python bc_pytorch/train_bc.py"
        )

    checkpoint = torch.load(CHECKPOINT_PATH, map_location="cpu", weights_only=True)
    required_fields = {
        "model_state_dict",
        "feature_mean",
        "feature_std",
        "feature_columns",
        "test_map_ids",
        "model_config",
    }
    missing_fields = required_fields - set(checkpoint)
    if missing_fields:
        missing_text = ", ".join(sorted(missing_fields))
        raise ValueError(f"Checkpoint 缺少字段：{missing_text}")
    if checkpoint["feature_columns"] != EXPECTED_FEATURE_COLUMNS:
        raise ValueError("Checkpoint 特征顺序与 rollout 预期不一致。")

    model = BehaviorCloningMLP(checkpoint["model_config"])
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, checkpoint


def replay_successful_worlds():
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            "未找到 data/expert_dataset.csv，请先运行："
            "python data/generate_expert_data.py"
        )

    dataset = pd.read_csv(DATA_PATH)
    dataset_map_ids = sorted(dataset["map_id"].unique().tolist())
    rng = random.Random(DATA_RANDOM_SEED)
    worlds = {}
    expert_paths = {}

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
        expert_path = astar_search(world)
        if expert_path is not None:
            expected_samples = pd.DataFrame(path_to_samples(world, expert_path, map_id))
            actual_samples = dataset[dataset["map_id"] == map_id].reset_index(drop=True)
            if not expected_samples.equals(actual_samples):
                raise ValueError(f"重放地图 {map_id} 与 expert_dataset.csv 不一致。")
            worlds[map_id] = world
            expert_paths[map_id] = expert_path

    if sorted(worlds) != dataset_map_ids:
        raise ValueError("固定随机种子重放的地图与 expert_dataset.csv 不一致。")

    return worlds, expert_paths


def state_features(world, current, feature_columns):
    agent_x, agent_y = current
    goal_x, goal_y = world.goal
    feature_values = {
        "agent_x": agent_x,
        "agent_y": agent_y,
        "goal_x": goal_x,
        "goal_y": goal_y,
        **get_local_obstacles(world, current),
    }
    return np.array(
        [feature_values[column] for column in feature_columns],
        dtype=np.float32,
    )


def predict_action(model, features, feature_mean, feature_std):
    normalized = (features - feature_mean) / feature_std
    tensor = torch.tensor(normalized, dtype=torch.float32).unsqueeze(0)
    with torch.no_grad():
        logits = model(tensor)
    return int(torch.argmax(logits, dim=1).item())


def rollout_episode(
    map_id,
    world,
    expert_path,
    model,
    feature_mean,
    feature_std,
    feature_columns,
):
    current = world.start
    visited = {current}
    max_steps = 4 * world.width * world.height

    for step in range(1, max_steps + 1):
        features = state_features(world, current, feature_columns)
        action = predict_action(model, features, feature_mean, feature_std)
        dx, dy = ACTION_DELTAS[action]
        next_node = (current[0] + dx, current[1] + dy)

        if not world.in_bounds(next_node):
            return episode_result(map_id, False, step - 1, expert_path, "out_of_bounds")
        if not world.passable(next_node):
            return episode_result(map_id, False, step - 1, expert_path, "hit_obstacle")

        current = next_node
        if current == world.goal:
            return episode_result(map_id, True, step, expert_path, None)
        if current in visited:
            return episode_result(map_id, False, step, expert_path, "loop_detected")
        visited.add(current)

    return episode_result(
        map_id,
        False,
        max_steps,
        expert_path,
        "max_steps_exceeded",
    )


def episode_result(map_id, success, model_steps, expert_path, failure_reason):
    astar_steps = len(expert_path) - 1
    return {
        "map_id": map_id,
        "success": success,
        "model_steps": model_steps,
        "astar_steps": astar_steps,
        "path_gap": model_steps - astar_steps if success else None,
        "failure_reason": failure_reason,
    }


def summarize_results(results):
    successful = [result for result in results if result["success"]]
    failure_counts = Counter(
        result["failure_reason"] for result in results if not result["success"]
    )
    for reason in FAILURE_REASONS:
        failure_counts.setdefault(reason, 0)

    def average(values):
        return sum(values) / len(values) if values else 0.0

    return {
        "evaluated_maps": len(results),
        "success_count": len(successful),
        "failure_count": len(results) - len(successful),
        "success_rate": len(successful) / len(results) if results else 0.0,
        "average_model_steps_on_success": average(
            [result["model_steps"] for result in successful]
        ),
        "average_astar_steps_on_success": average(
            [result["astar_steps"] for result in successful]
        ),
        "average_path_gap_on_success": average(
            [result["path_gap"] for result in successful]
        ),
        "failure_reasons": dict(failure_counts),
    }


def save_summary_figure(summary):
    FIGURE_PATH.parent.mkdir(parents=True, exist_ok=True)
    labels = ["Success", "Failure"]
    values = [summary["success_count"], summary["failure_count"]]
    figure, axis = plt.subplots(figsize=(6, 4.5))
    bars = axis.bar(labels, values, color=["#2E8B57", "#C44E52"])
    axis.bar_label(bars)
    axis.set_ylabel("Episode count")
    axis.set_title("BC Policy Closed-Loop Rollout Results")
    axis.set_ylim(0, max(values + [1]) + 2)
    figure.tight_layout()
    figure.savefig(FIGURE_PATH, dpi=160)
    plt.close(figure)


def episode_rows(results):
    rows = [
        "| map_id | success | model steps | A* steps | path gap | failure reason |",
        "| ---: | --- | ---: | ---: | ---: | --- |",
    ]
    for result in results:
        gap = result["path_gap"] if result["path_gap"] is not None else "-"
        reason = result["failure_reason"] or "-"
        rows.append(
            f"| {result['map_id']} | {result['success']} | "
            f"{result['model_steps']} | {result['astar_steps']} | {gap} | {reason} |"
        )
    return "\n".join(rows)


def write_report(summary, results, checkpoint):
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    failure_rows = "\n".join(
        f"| `{reason}` | {summary['failure_reasons'][reason]} |"
        for reason in FAILURE_REASONS
    )
    report = f"""# BC Policy Rollout Evaluation 结果

## 评估目标

评估 Stage 4 训练出的 Behavior Cloning policy 能否在 GridWorld 中从 start 开始，根据自己的预测动作一步步走到 goal。这是 closed-loop rollout evaluation，不是训练，也不是强化学习。

## 评估设置

- 加载 checkpoint：`checkpoints/bc_policy.pt`
- Checkpoint PyTorch 版本：`{checkpoint["torch_version"]}`
- BC single-step test accuracy：`{checkpoint["test_accuracy"]:.4f}`
- 评估地图：Stage 4 checkpoint 中记录的 test map IDs
- 使用测试地图数量：{summary["evaluated_maps"]}
- 地图重放：使用 Stage 2 固定随机种子和原始地图生成规则
- 每张地图最大步数：`4 * width * height`
- A* 基准路径长度：`len(astar_search(world)) - 1`

## 汇总结果

- Evaluated maps：{summary["evaluated_maps"]}
- Success count：{summary["success_count"]}
- Failure count：{summary["failure_count"]}
- Success rate：`{summary["success_rate"]:.2%}`
- Average model steps on success：`{summary["average_model_steps_on_success"]:.2f}`
- Average A* steps on success：`{summary["average_astar_steps_on_success"]:.2f}`
- Average path length gap on success：`{summary["average_path_gap_on_success"]:.2f}`

### Failure reasons

| failure reason | count |
| --- | ---: |
{failure_rows}

## 每张测试地图结果

{episode_rows(results)}

## 当前理解与边界

- Single-step accuracy 衡量独立测试状态上的动作预测正确率；rollout success rate 衡量模型连续使用自己预测的动作后，最终到达 goal 的比例。
- 单步错误会改变后续状态，模型可能进入专家数据中较少出现的位置，因此较高的 single-step accuracy 不保证较高的 rollout success rate。
- 当前只加载 Stage 4 BC policy 做评估，没有训练新策略。
- 当前不是强化学习，也没有实现 Q-learning。
"""
    REPORT_PATH.write_text(report, encoding="utf-8")


def main():
    model, checkpoint = load_policy()
    feature_mean = checkpoint["feature_mean"].numpy()
    feature_std = checkpoint["feature_std"].numpy()
    feature_columns = checkpoint["feature_columns"]
    test_map_ids = checkpoint["test_map_ids"]

    worlds, expert_paths = replay_successful_worlds()
    missing_test_maps = sorted(set(test_map_ids) - set(worlds))
    if missing_test_maps:
        raise ValueError(f"无法重放测试地图：{missing_test_maps}")

    results = [
        rollout_episode(
            map_id,
            worlds[map_id],
            expert_paths[map_id],
            model,
            feature_mean,
            feature_std,
            feature_columns,
        )
        for map_id in test_map_ids
    ]
    summary = summarize_results(results)
    save_summary_figure(summary)
    write_report(summary, results, checkpoint)

    print(f"Evaluated maps: {summary['evaluated_maps']}")
    print(f"Success count: {summary['success_count']}")
    print(f"Failure count: {summary['failure_count']}")
    print(f"Success rate: {summary['success_rate']:.2%}")
    print(
        "Average model steps on success:",
        f"{summary['average_model_steps_on_success']:.2f}",
    )
    print(
        "Average A* steps on success:",
        f"{summary['average_astar_steps_on_success']:.2f}",
    )
    print(
        "Average path gap on success:",
        f"{summary['average_path_gap_on_success']:.2f}",
    )
    print("Failure reasons:", summary["failure_reasons"])
    print(f"报告：{REPORT_PATH.relative_to(ROOT)}")
    print(f"汇总图：{FIGURE_PATH.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
