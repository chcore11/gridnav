from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
)
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier


ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "expert_dataset.csv"
REPORT_PATH = ROOT / "reports" / "ml_baseline_result.md"
FIGURE_PATH = ROOT / "figures" / "ml_baseline_confusion_matrix.png"

FEATURE_COLUMNS = [
    "agent_x",
    "agent_y",
    "goal_x",
    "goal_y",
    "obs_up",
    "obs_down",
    "obs_left",
    "obs_right",
]
LABEL_COLUMN = "action"
ACTION_LABELS = [0, 1, 2, 3]
ACTION_NAMES = ["up", "down", "left", "right"]
RANDOM_SEED = 42
TEST_SIZE = 0.2


def load_dataset():
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            "未找到 data/expert_dataset.csv，请先运行："
            "python data/generate_expert_data.py"
        )

    dataset = pd.read_csv(DATA_PATH)
    required_columns = {"map_id", *FEATURE_COLUMNS, LABEL_COLUMN}
    missing_columns = required_columns - set(dataset.columns)
    if missing_columns:
        missing_text = ", ".join(sorted(missing_columns))
        raise ValueError(f"数据集缺少字段：{missing_text}")
    if dataset[list(required_columns)].isnull().any().any():
        raise ValueError("数据集中存在空值，请先检查专家数据。")

    invalid_actions = sorted(set(dataset[LABEL_COLUMN]) - set(ACTION_LABELS))
    if invalid_actions:
        raise ValueError(f"数据集中存在未知 action：{invalid_actions}")

    return dataset


def split_by_map_id(dataset):
    map_ids = sorted(dataset["map_id"].unique())
    if len(map_ids) < 2:
        raise ValueError("至少需要两张地图才能划分训练集和测试集。")

    train_map_ids, test_map_ids = train_test_split(
        map_ids,
        test_size=TEST_SIZE,
        random_state=RANDOM_SEED,
    )
    train_mask = dataset["map_id"].isin(train_map_ids)
    test_mask = dataset["map_id"].isin(test_map_ids)

    return (
        dataset.loc[train_mask, FEATURE_COLUMNS],
        dataset.loc[test_mask, FEATURE_COLUMNS],
        dataset.loc[train_mask, LABEL_COLUMN],
        dataset.loc[test_mask, LABEL_COLUMN],
        sorted(train_map_ids),
        sorted(test_map_ids),
    )


def build_models():
    return {
        "KNN": make_pipeline(
            StandardScaler(),
            KNeighborsClassifier(n_neighbors=5),
        ),
        "Logistic Regression": make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=1000, random_state=RANDOM_SEED),
        ),
        "Decision Tree": DecisionTreeClassifier(random_state=RANDOM_SEED),
    }


def evaluate_models(models, x_train, y_train, x_test, y_test):
    results = {}

    for name, model in models.items():
        model.fit(x_train, y_train)
        predictions = model.predict(x_test)
        results[name] = {
            "accuracy": accuracy_score(y_test, predictions),
            "classification_report": classification_report(
                y_test,
                predictions,
                labels=ACTION_LABELS,
                target_names=ACTION_NAMES,
                zero_division=0,
            ),
            "confusion_matrix": confusion_matrix(
                y_test,
                predictions,
                labels=ACTION_LABELS,
            ),
        }

    return results


def save_confusion_matrices(results):
    FIGURE_PATH.parent.mkdir(parents=True, exist_ok=True)
    figure, axes = plt.subplots(1, len(results), figsize=(15, 4.5))

    for axis, (name, result) in zip(axes, results.items()):
        display = ConfusionMatrixDisplay(
            confusion_matrix=result["confusion_matrix"],
            display_labels=ACTION_NAMES,
        )
        display.plot(ax=axis, cmap="Blues", colorbar=False)
        axis.set_title(f"{name}\naccuracy={result['accuracy']:.4f}")

    figure.suptitle("GridNav ML Baseline Confusion Matrices")
    figure.tight_layout()
    figure.savefig(FIGURE_PATH, dpi=160, bbox_inches="tight")
    plt.close(figure)


def confusion_matrix_markdown(matrix):
    rows = [
        "| actual \\ predicted | up | down | left | right |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for action_name, values in zip(ACTION_NAMES, matrix):
        cells = " | ".join(str(value) for value in values)
        rows.append(f"| {action_name} | {cells} |")
    return "\n".join(rows)


def write_report(dataset, train_map_ids, test_map_ids, y_train, y_test, results):
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    best_model = max(results, key=lambda name: results[name]["accuracy"])

    summary_rows = "\n".join(
        f"| {name} | {result['accuracy']:.4f} |"
        for name, result in results.items()
    )
    detail_sections = []
    for name, result in results.items():
        detail_sections.append(
            f"""## {name}

- Action accuracy：`{result["accuracy"]:.4f}`

### Classification report

```text
{result["classification_report"].rstrip()}
```

### Confusion matrix

{confusion_matrix_markdown(result["confusion_matrix"])}
"""
        )
    detail_text = "\n".join(detail_sections)

    report = f"""# 传统机器学习 Baseline 结果

## 任务

本阶段只进行单步 action prediction：输入当前 state 的八个特征，预测 A* 专家在该状态下选择的 action。没有进行完整导航 rollout。

## 数据与划分

- 数据文件：`data/expert_dataset.csv`
- 特征：`{", ".join(FEATURE_COLUMNS)}`
- 标签：`action`，编码为 `0 = up`、`1 = down`、`2 = left`、`3 = right`
- 总地图数：{dataset["map_id"].nunique()}
- 总样本数：{len(dataset)}
- 训练地图数：{len(train_map_ids)}
- 测试地图数：{len(test_map_ids)}
- 训练样本数：{len(y_train)}
- 测试样本数：{len(y_test)}
- 随机种子：{RANDOM_SEED}
- 测试集比例：{TEST_SIZE:.0%} 的 `map_id`
- 训练 `map_id`：`{train_map_ids}`
- 测试 `map_id`：`{test_map_ids}`

划分时先拆分 `map_id`，再选出对应样本。同一张地图产生的整条路径只会出现在训练集或测试集中的一边，避免相邻路径状态泄漏导致测试结果虚高。

## 模型设置

- KNN：`n_neighbors=5`，先使用 `StandardScaler` 标准化特征。
- Logistic Regression：`max_iter=1000`，先使用 `StandardScaler` 标准化特征。
- Decision Tree：使用默认树结构参数，并设置 `random_state={RANDOM_SEED}`。

## Accuracy 汇总

| 模型 | Action accuracy |
| --- | ---: |
{summary_rows}

最佳模型是 **{best_model}**，测试集 action accuracy 为 **{results[best_model]["accuracy"]:.4f}**。

混淆矩阵图保存于 `figures/ml_baseline_confusion_matrix.png`，图中同时包含三个模型，行表示真实动作，列表示预测动作。

{detail_text}
## 当前理解与局限

- Accuracy 表示测试样本中预测动作与 A* 专家动作相同的比例。
- Classification report 分动作展示 precision、recall 和 F1-score，可以帮助发现只看总 accuracy 时看不到的类别差异。
- Confusion matrix 可以直接看到每个真实动作容易被误判成哪个动作。
- 当前 state 只包含位置、目标位置和局部障碍，不包含完整地图；不同地图中相同特征可能对应不同的 A* 动作。
- 当前只评估单步分类，没有进行完整导航 rollout，因此 action accuracy 高不代表导航一定成功。
"""
    REPORT_PATH.write_text(report, encoding="utf-8")

    return best_model


def main():
    dataset = load_dataset()
    x_train, x_test, y_train, y_test, train_map_ids, test_map_ids = split_by_map_id(
        dataset
    )
    results = evaluate_models(build_models(), x_train, y_train, x_test, y_test)
    save_confusion_matrices(results)
    best_model = write_report(
        dataset,
        train_map_ids,
        test_map_ids,
        y_train,
        y_test,
        results,
    )

    print(f"按 map_id 划分：{len(train_map_ids)} 张训练地图，{len(test_map_ids)} 张测试地图")
    for name, result in results.items():
        print(f"{name}: accuracy={result['accuracy']:.4f}")
    print(f"最佳模型：{best_model}")
    print(f"报告：{REPORT_PATH.relative_to(ROOT)}")
    print(f"混淆矩阵图：{FIGURE_PATH.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
