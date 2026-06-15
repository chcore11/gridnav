import sys
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from env.grid_world import GridWorld


WIDTH = 6
HEIGHT = 6
START = (0, 0)
GOAL = (5, 5)
OBSTACLES = {
    (1, 0),
    (1, 1),
    (1, 2),
    (2, 4),
    (3, 1),
    (3, 2),
    (3, 3),
    (3, 4),
}

ACTION_NAMES = {
    0: "up",
    1: "down",
    2: "left",
    3: "right",
}
ACTION_DELTAS = {
    0: (0, -1),
    1: (0, 1),
    2: (-1, 0),
    3: (1, 0),
}
FAILURE_REASONS = [
    "hit_obstacle",
    "out_of_bounds",
    "loop_detected",
    "max_steps_exceeded",
]

GOAL_REWARD = 100
COLLISION_REWARD = -10
STEP_REWARD = -1

ALPHA = 0.1
GAMMA = 0.95
INITIAL_EPSILON = 1.0
MIN_EPSILON = 0.05
EPSILON_DECAY = 0.995
TRAIN_EPISODES = 3000
MAX_STEPS = 100
SUCCESS_WINDOW = 100
EVALUATION_EPISODES = 100
RANDOM_SEED = 42

REPORT_PATH = ROOT / "reports" / "qlearning_result.md"
TRAINING_CURVE_PATH = ROOT / "figures" / "qlearning_training_curve.png"
SUCCESS_SUMMARY_PATH = ROOT / "figures" / "qlearning_success_summary.png"


def create_world():
    return GridWorld(
        width=WIDTH,
        height=HEIGHT,
        obstacles=OBSTACLES,
        start=START,
        goal=GOAL,
    )


def transition(world, state, action):
    dx, dy = ACTION_DELTAS[action]
    next_state = (state[0] + dx, state[1] + dy)

    if not world.in_bounds(next_state):
        return state, COLLISION_REWARD, False, "out_of_bounds"
    if not world.passable(next_state):
        return state, COLLISION_REWARD, False, "hit_obstacle"
    if next_state == world.goal:
        return next_state, GOAL_REWARD, True, None
    return next_state, STEP_REWARD, False, None


def q_values(q_table, state):
    x, y = state
    return q_table[y, x]


def choose_training_action(q_table, state, epsilon, rng):
    if rng.random() < epsilon:
        return int(rng.integers(len(ACTION_NAMES)))

    values = q_values(q_table, state)
    best_actions = np.flatnonzero(values == values.max())
    return int(rng.choice(best_actions))


def train_qlearning(world):
    rng = np.random.default_rng(RANDOM_SEED)
    q_table = np.zeros((world.height, world.width, len(ACTION_NAMES)), dtype=float)
    episode_rewards = []
    episode_successes = []
    episode_steps = []
    rolling_success_rates = []
    epsilon = INITIAL_EPSILON

    for _ in range(TRAIN_EPISODES):
        state = world.start
        total_reward = 0
        success = False

        for step in range(1, MAX_STEPS + 1):
            action = choose_training_action(q_table, state, epsilon, rng)
            next_state, reward, done, _ = transition(world, state, action)

            current_q = q_values(q_table, state)[action]
            next_best_q = 0.0 if done else float(q_values(q_table, next_state).max())
            td_target = reward + GAMMA * next_best_q
            q_values(q_table, state)[action] = current_q + ALPHA * (
                td_target - current_q
            )

            total_reward += reward
            state = next_state
            if done:
                success = True
                break

        episode_rewards.append(total_reward)
        episode_successes.append(success)
        episode_steps.append(step)
        recent_successes = episode_successes[-SUCCESS_WINDOW:]
        rolling_success_rates.append(sum(recent_successes) / len(recent_successes))
        epsilon = max(MIN_EPSILON, epsilon * EPSILON_DECAY)

    return {
        "q_table": q_table,
        "episode_rewards": episode_rewards,
        "episode_successes": episode_successes,
        "episode_steps": episode_steps,
        "rolling_success_rates": rolling_success_rates,
        "final_epsilon": epsilon,
    }


def evaluate_episode(world, q_table):
    state = world.start
    visited = {state}
    path = [state]
    total_reward = 0

    for step in range(1, MAX_STEPS + 1):
        action = int(np.argmax(q_values(q_table, state)))
        next_state, reward, done, collision_reason = transition(world, state, action)
        total_reward += reward

        if collision_reason is not None:
            return {
                "success": False,
                "steps": step - 1,
                "total_reward": total_reward,
                "failure_reason": collision_reason,
                "path": path,
            }

        state = next_state
        path.append(state)
        if done:
            return {
                "success": True,
                "steps": step,
                "total_reward": total_reward,
                "failure_reason": None,
                "path": path,
            }
        if state in visited:
            return {
                "success": False,
                "steps": step,
                "total_reward": total_reward,
                "failure_reason": "loop_detected",
                "path": path,
            }
        visited.add(state)

    return {
        "success": False,
        "steps": MAX_STEPS,
        "total_reward": total_reward,
        "failure_reason": "max_steps_exceeded",
        "path": path,
    }


def evaluate_greedy_policy(world, q_table):
    results = [evaluate_episode(world, q_table) for _ in range(EVALUATION_EPISODES)]
    successful = [result for result in results if result["success"]]
    failure_counts = Counter(
        result["failure_reason"] for result in results if not result["success"]
    )
    for reason in FAILURE_REASONS:
        failure_counts.setdefault(reason, 0)

    return {
        "results": results,
        "success_count": len(successful),
        "failure_count": len(results) - len(successful),
        "success_rate": len(successful) / len(results),
        "average_steps_on_success": (
            sum(result["steps"] for result in successful) / len(successful)
            if successful
            else 0.0
        ),
        "failure_reasons": dict(failure_counts),
        "greedy_path": results[0]["path"],
    }


def save_training_curve(training):
    TRAINING_CURVE_PATH.parent.mkdir(parents=True, exist_ok=True)
    episodes = np.arange(1, TRAIN_EPISODES + 1)
    rewards = np.array(training["episode_rewards"], dtype=float)
    smoothing_window = 50
    smoothed_rewards = np.convolve(
        rewards,
        np.ones(smoothing_window) / smoothing_window,
        mode="valid",
    )

    figure, axes = plt.subplots(2, 1, figsize=(9, 7), sharex=True)
    axes[0].plot(episodes, rewards, color="#A9C5DA", linewidth=0.8, label="Reward")
    axes[0].plot(
        episodes[smoothing_window - 1 :],
        smoothed_rewards,
        color="#2F6690",
        linewidth=2,
        label=f"{smoothing_window}-episode mean",
    )
    axes[0].set_ylabel("Total reward")
    axes[0].set_title("Tabular Q-learning Training")
    axes[0].legend()

    axes[1].plot(
        episodes,
        np.array(training["rolling_success_rates"]) * 100,
        color="#2E8B57",
        linewidth=1.8,
    )
    axes[1].set_xlabel("Episode")
    axes[1].set_ylabel(f"Success rate, last {SUCCESS_WINDOW} (%)")
    axes[1].set_ylim(-2, 102)

    figure.tight_layout()
    figure.savefig(TRAINING_CURVE_PATH, dpi=160)
    plt.close(figure)


def save_success_summary(evaluation):
    SUCCESS_SUMMARY_PATH.parent.mkdir(parents=True, exist_ok=True)
    labels = ["Success", "Failure"]
    values = [evaluation["success_count"], evaluation["failure_count"]]
    figure, axis = plt.subplots(figsize=(6, 4.5))
    bars = axis.bar(labels, values, color=["#2E8B57", "#C44E52"])
    axis.bar_label(bars)
    axis.set_ylabel("Evaluation episode count")
    axis.set_title("Q-learning Greedy Policy Evaluation")
    axis.set_ylim(0, max(values + [1]) + 10)
    figure.tight_layout()
    figure.savefig(SUCCESS_SUMMARY_PATH, dpi=160)
    plt.close(figure)


def format_path(path):
    return " -> ".join(f"({x}, {y})" for x, y in path)


def write_report(training, evaluation):
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    overall_training_success_rate = (
        sum(training["episode_successes"]) / TRAIN_EPISODES
    )
    final_training_success_rate = sum(
        training["episode_successes"][-SUCCESS_WINDOW:]
    ) / SUCCESS_WINDOW
    final_average_reward = float(
        np.mean(training["episode_rewards"][-SUCCESS_WINDOW:])
    )
    final_average_steps = float(
        np.mean(training["episode_steps"][-SUCCESS_WINDOW:])
    )
    milestone_rows = []
    for episode in [100, 500, 1000, 2000, TRAIN_EPISODES]:
        start = max(0, episode - SUCCESS_WINDOW)
        recent_successes = training["episode_successes"][start:episode]
        recent_rewards = training["episode_rewards"][start:episode]
        milestone_rows.append(
            f"| {episode} | {sum(recent_successes) / len(recent_successes):.2%} | "
            f"{np.mean(recent_rewards):.2f} |"
        )
    milestone_table = "\n".join(milestone_rows)
    failure_rows = "\n".join(
        f"| `{reason}` | {evaluation['failure_reasons'][reason]} |"
        for reason in FAILURE_REASONS
    )
    obstacle_text = ", ".join(str(node) for node in sorted(OBSTACLES))

    report = f"""# Stage 5：Tabular Q-learning 结果

## 阶段目标

在固定 GridWorld 中实现最小 tabular Q-learning。智能体不读取 A* 路径或专家 action，而是通过自己选择动作、获得 reward、更新 Q-table 来学习从 start 到 goal 的策略。

这是强化学习，因为训练数据来自智能体与环境的交互，策略会根据 reward 反馈逐步改变。它不是监督学习，也没有使用 PyTorch 或 DQN。

## 环境与定义

- 地图大小：`{WIDTH} x {HEIGHT}`
- Start：`{START}`
- Goal：`{GOAL}`
- Obstacles：`{obstacle_text}`
- State：智能体当前坐标 `(x, y)`
- Action：`0 = up`、`1 = down`、`2 = left`、`3 = right`
- Q-table：形状为 `({HEIGHT}, {WIDTH}, 4)` 的表，`Q[y, x, action]` 表示在坐标 `(x, y)` 选择该动作的价值估计
- 到达 goal：`+{GOAL_REWARD}`
- 撞墙、越界或撞障碍：`{COLLISION_REWARD}`
- 普通移动一步：`{STEP_REWARD}`
- 训练时碰撞后留在原状态并继续 episode，让 Q-table 学习降低该动作的价值
- 最大步数：`{MAX_STEPS}`，防止 episode 无限运行
- 随机种子：`{RANDOM_SEED}`

## Q-learning 参数

- Alpha 学习率：`{ALPHA}`
- Gamma 折扣因子：`{GAMMA}`
- 初始 epsilon：`{INITIAL_EPSILON}`
- 最小 epsilon：`{MIN_EPSILON}`
- Epsilon decay：每个 episode 后乘以 `{EPSILON_DECAY}`
- 最终 epsilon：`{training['final_epsilon']:.4f}`
- 训练 episode 数：`{TRAIN_EPISODES}`

更新公式：

`Q(s,a) = Q(s,a) + alpha * [reward + gamma * max Q(s',a') - Q(s,a)]`

到达 goal 后 episode 结束，终止状态不再加入未来 Q-value。

## 训练结果

- 总训练 episode 数：{TRAIN_EPISODES}
- 整体训练成功率：`{overall_training_success_rate:.2%}`
- 最后 {SUCCESS_WINDOW} 个 episode 成功率：`{final_training_success_rate:.2%}`
- 最后 {SUCCESS_WINDOW} 个 episode 平均 total reward：`{final_average_reward:.2f}`
- 最后 {SUCCESS_WINDOW} 个 episode 平均 steps：`{final_average_steps:.2f}`
- 每个 episode 都记录了 total reward、是否成功和 episode steps
- `figures/qlearning_training_curve.png` 展示 total reward 和最近 {SUCCESS_WINDOW} 个 episode 的 success rate

### 训练过程摘要

| episode | 最近 {SUCCESS_WINDOW} 个 episode 成功率 | 最近 {SUCCESS_WINDOW} 个 episode 平均 reward |
| ---: | ---: | ---: |
{milestone_table}

## Greedy Policy 评估

训练完成后关闭随机探索，每个 state 都选择 Q 值最大的动作。固定地图和 greedy policy 都是确定性的，因此重复评估 {EVALUATION_EPISODES} 次用于统计结果。

- Success count：{evaluation['success_count']}
- Failure count：{evaluation['failure_count']}
- Success rate：`{evaluation['success_rate']:.2%}`
- Average steps on success：`{evaluation['average_steps_on_success']:.2f}`
- Greedy path：`{format_path(evaluation['greedy_path'])}`

### Failure reasons

| failure reason | count |
| --- | ---: |
{failure_rows}

## 与前面阶段的区别

- Stage 4 Behavior Cloning 使用专家 action 作为标签，学习专家给出的 `state -> action` 映射。
- Stage 4.5 BC Rollout Evaluation 只评估已经训练好的 BC policy，不使用 reward，也不训练或更新策略，因此不是强化学习。
- Stage 5 Q-learning 不依赖专家 action。智能体通过 epsilon-greedy 自己探索动作，根据 reward 和下一状态的 Q-value 更新 Q-table。

## 当前边界

- 当前是固定地图上的最小 tabular Q-learning，只使用坐标作为 state。
- Q-table 只适用于这张固定地图，不能直接泛化到不同 goal 或障碍布局。
- 当前没有使用 A* 专家 action、PyTorch 或 DQN。
"""
    REPORT_PATH.write_text(report, encoding="utf-8")


def main():
    world = create_world()
    training = train_qlearning(world)
    evaluation = evaluate_greedy_policy(world, training["q_table"])
    save_training_curve(training)
    save_success_summary(evaluation)
    write_report(training, evaluation)

    final_training_success_rate = sum(
        training["episode_successes"][-SUCCESS_WINDOW:]
    ) / SUCCESS_WINDOW
    print("Stage 5 tabular Q-learning complete.")
    print(f"Training episodes: {TRAIN_EPISODES}")
    print(f"Final training success rate: {final_training_success_rate:.2%}")
    print(f"Greedy evaluation success rate: {evaluation['success_rate']:.2%}")
    print(
        "Average steps on success: "
        f"{evaluation['average_steps_on_success']:.2f}"
    )
    print(f"Failure reasons: {evaluation['failure_reasons']}")
    print(f"Greedy path: {format_path(evaluation['greedy_path'])}")


if __name__ == "__main__":
    main()
