import random
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import DataLoader, Dataset


ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "expert_dataset.csv"
REPORT_PATH = ROOT / "reports" / "bc_pytorch_result.md"
FIGURE_PATH = ROOT / "figures" / "bc_training_curve.png"
CHECKPOINT_PATH = ROOT / "checkpoints" / "bc_policy.pt"

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
BATCH_SIZE = 64
EPOCHS = 100
LEARNING_RATE = 0.001
LR_BASELINE_ACCURACY = 0.8557
MODEL_CONFIG = {
    "input_dim": 8,
    "hidden_dims": [64, 32],
    "output_dim": 4,
    "layers": ["Linear(8, 64)", "ReLU", "Linear(64, 32)", "ReLU", "Linear(32, 4)"],
}


class ExpertActionDataset(Dataset):
    def __init__(self, features, labels):
        self.features = torch.tensor(features, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        return self.features[index], self.labels[index]


class BehaviorCloningMLP(nn.Module):
    def __init__(self, model_config=MODEL_CONFIG):
        super().__init__()
        input_dim = model_config["input_dim"]
        hidden_dim_1, hidden_dim_2 = model_config["hidden_dims"]
        output_dim = model_config["output_dim"]
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim_1),
            nn.ReLU(),
            nn.Linear(hidden_dim_1, hidden_dim_2),
            nn.ReLU(),
            nn.Linear(hidden_dim_2, output_dim),
        )

    def forward(self, features):
        return self.network(features)


def set_random_seed():
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    torch.manual_seed(RANDOM_SEED)


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
    train_map_ids = sorted(train_map_ids)
    test_map_ids = sorted(test_map_ids)
    assert set(train_map_ids).isdisjoint(test_map_ids)

    train_data = dataset[dataset["map_id"].isin(train_map_ids)]
    test_data = dataset[dataset["map_id"].isin(test_map_ids)]
    return train_data, test_data, train_map_ids, test_map_ids


def prepare_datasets(train_data, test_data):
    train_features = train_data[FEATURE_COLUMNS].to_numpy(dtype=np.float32)
    test_features = test_data[FEATURE_COLUMNS].to_numpy(dtype=np.float32)

    feature_mean = train_features.mean(axis=0)
    feature_std = train_features.std(axis=0)
    feature_std[feature_std == 0] = 1.0

    train_features = (train_features - feature_mean) / feature_std
    test_features = (test_features - feature_mean) / feature_std

    train_dataset = ExpertActionDataset(
        train_features,
        train_data[LABEL_COLUMN].to_numpy(),
    )
    test_dataset = ExpertActionDataset(
        test_features,
        test_data[LABEL_COLUMN].to_numpy(),
    )
    return train_dataset, test_dataset, feature_mean, feature_std


def build_data_loaders(train_dataset, test_dataset):
    generator = torch.Generator().manual_seed(RANDOM_SEED)
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        generator=generator,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
    )
    return train_loader, test_loader


def train_model(model, train_loader):
    loss_function = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    train_losses = []

    model.train()
    for epoch in range(1, EPOCHS + 1):
        total_loss = 0.0
        total_samples = 0

        for features, labels in train_loader:
            optimizer.zero_grad()
            logits = model(features)
            loss = loss_function(logits, labels)
            loss.backward()
            optimizer.step()

            batch_size = labels.size(0)
            total_loss += loss.item() * batch_size
            total_samples += batch_size

        average_loss = total_loss / total_samples
        train_losses.append(average_loss)
        if epoch == 1 or epoch % 10 == 0:
            print(f"Epoch {epoch:3d}/{EPOCHS}: train_loss={average_loss:.6f}")

    return train_losses


def evaluate_model(model, test_loader):
    predictions = []
    labels = []

    model.eval()
    with torch.no_grad():
        for features, batch_labels in test_loader:
            logits = model(features)
            batch_predictions = torch.argmax(logits, dim=1)
            predictions.extend(batch_predictions.tolist())
            labels.extend(batch_labels.tolist())

    return {
        "accuracy": accuracy_score(labels, predictions),
        "classification_report": classification_report(
            labels,
            predictions,
            labels=ACTION_LABELS,
            target_names=ACTION_NAMES,
            zero_division=0,
        ),
        "confusion_matrix": confusion_matrix(
            labels,
            predictions,
            labels=ACTION_LABELS,
        ),
    }


def save_training_curve(train_losses):
    FIGURE_PATH.parent.mkdir(parents=True, exist_ok=True)
    figure, axis = plt.subplots(figsize=(7, 4.5))
    axis.plot(range(1, EPOCHS + 1), train_losses)
    axis.set_xlabel("Epoch")
    axis.set_ylabel("Average train loss")
    axis.set_title("PyTorch Behavior Cloning Training Curve")
    axis.grid(True, alpha=0.3)
    figure.tight_layout()
    figure.savefig(FIGURE_PATH, dpi=160)
    plt.close(figure)


def save_checkpoint(
    model,
    feature_mean,
    feature_std,
    train_map_ids,
    test_map_ids,
    final_train_loss,
    test_accuracy,
):
    CHECKPOINT_PATH.parent.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "feature_mean": torch.tensor(feature_mean, dtype=torch.float32),
        "feature_std": torch.tensor(feature_std, dtype=torch.float32),
        "feature_columns": list(FEATURE_COLUMNS),
        "train_map_ids": [int(map_id) for map_id in train_map_ids],
        "test_map_ids": [int(map_id) for map_id in test_map_ids],
        "random_seed": int(RANDOM_SEED),
        "model_config": MODEL_CONFIG,
        "torch_version": str(torch.__version__),
        "final_train_loss": float(final_train_loss),
        "test_accuracy": float(test_accuracy),
    }
    torch.save(checkpoint, CHECKPOINT_PATH)


def confusion_matrix_markdown(matrix):
    rows = [
        "| actual \\ predicted | up | down | left | right |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for action_name, values in zip(ACTION_NAMES, matrix):
        cells = " | ".join(str(value) for value in values)
        rows.append(f"| {action_name} | {cells} |")
    return "\n".join(rows)


def write_report(
    dataset,
    train_data,
    test_data,
    train_map_ids,
    test_map_ids,
    train_losses,
    evaluation,
):
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    comparison = evaluation["accuracy"] - LR_BASELINE_ACCURACY
    report = f"""# PyTorch Behavior Cloning 结果

## 任务

使用一个简单的 PyTorch MLP 学习 A* 专家数据中的 `state -> action` 映射。当前评估仍是 single-step action prediction，没有进行完整导航 rollout。

## 数据与划分

- 数据来源：`data/expert_dataset.csv`
- 特征 X：`{", ".join(FEATURE_COLUMNS)}`
- 标签 y：`action`，编码为 `0 = up`、`1 = down`、`2 = left`、`3 = right`
- 总地图数：{dataset["map_id"].nunique()}
- 总样本数：{len(dataset)}
- 训练地图数：{len(train_map_ids)}
- 测试地图数：{len(test_map_ids)}
- 训练样本数：{len(train_data)}
- 测试样本数：{len(test_data)}
- 随机种子：{RANDOM_SEED}

训练集和测试集先按 `map_id` 划分，并断言两个 map ID 集合不相交。同一张地图的路径样本不会同时出现在训练集和测试集中。特征标准化参数只根据训练集计算。

## 模型与训练设置

```text
Linear(8, 64)
ReLU
Linear(64, 32)
ReLU
Linear(32, 4)
```

- PyTorch：`{torch.__version__}`
- Dataset：`ExpertActionDataset`
- DataLoader batch size：`{BATCH_SIZE}`
- Loss function：`CrossEntropyLoss`
- Optimizer：`Adam`
- Learning rate：`{LEARNING_RATE}`
- Epoch 数：`{EPOCHS}`
- Final train loss：`{train_losses[-1]:.6f}`

训练曲线保存于 `figures/bc_training_curve.png`。
用于 rollout 评估的 checkpoint 保存于 `checkpoints/bc_policy.pt`，其中包含模型权重、标准化参数、特征顺序和训练/测试地图 ID。

## 测试结果

- PyTorch BC single-step action accuracy：`{evaluation["accuracy"]:.4f}`
- Stage 3 Logistic Regression accuracy：`{LR_BASELINE_ACCURACY:.4f}`
- PyTorch BC 与 Logistic Regression 的 accuracy 差值：`{comparison:+.4f}`

### Classification report

```text
{evaluation["classification_report"].rstrip()}
```

### Confusion matrix

{confusion_matrix_markdown(evaluation["confusion_matrix"])}

## 当前结论与边界

- 当前 PyTorch BC 使用神经网络模仿 A* 专家的单步动作。
- 当前结果可以与 Stage 3 Logistic Regression baseline 比较，但模型更复杂不保证 accuracy 一定更高。
- 当前 Stage 4 训练与单步分类评估本身没有进行 rollout；完整导航能力由 Stage 5 单独评估。
- 本阶段没有进入强化学习。
"""
    REPORT_PATH.write_text(report, encoding="utf-8")


def main():
    set_random_seed()
    dataset = load_dataset()
    train_data, test_data, train_map_ids, test_map_ids = split_by_map_id(dataset)
    train_dataset, test_dataset, feature_mean, feature_std = prepare_datasets(
        train_data,
        test_data,
    )
    train_loader, test_loader = build_data_loaders(train_dataset, test_dataset)

    model = BehaviorCloningMLP()
    train_losses = train_model(model, train_loader)
    evaluation = evaluate_model(model, test_loader)
    save_training_curve(train_losses)
    save_checkpoint(
        model,
        feature_mean,
        feature_std,
        train_map_ids,
        test_map_ids,
        train_losses[-1],
        evaluation["accuracy"],
    )
    write_report(
        dataset,
        train_data,
        test_data,
        train_map_ids,
        test_map_ids,
        train_losses,
        evaluation,
    )

    print(f"PyTorch version: {torch.__version__}")
    print(f"按 map_id 划分：{len(train_map_ids)} 张训练地图，{len(test_map_ids)} 张测试地图")
    print(f"Final train loss: {train_losses[-1]:.6f}")
    print(f"Test single-step action accuracy: {evaluation['accuracy']:.4f}")
    print("Classification report:")
    print(evaluation["classification_report"])
    print("Confusion matrix:")
    print(evaluation["confusion_matrix"])
    print(f"训练曲线：{FIGURE_PATH.relative_to(ROOT)}")
    print(f"报告：{REPORT_PATH.relative_to(ROOT)}")
    print(f"Checkpoint：{CHECKPOINT_PATH.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
