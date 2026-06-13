from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "figures" / "xiaohongshu_stage3"
ASTAR_IMAGE = ROOT / "figures" / "astar_demo.png"
CONFUSION_IMAGE = ROOT / "figures" / "ml_baseline_confusion_matrix.png"

WIDTH = 1080
HEIGHT = 1440

BG = "#F6F8FC"
CARD = "#FFFFFF"
INK = "#172033"
MUTED = "#667085"
LINE = "#DDE4EF"
BLUE = "#3B6FF5"
BLUE_LIGHT = "#EAF0FF"
GREEN = "#18A675"
GREEN_LIGHT = "#E7F7F1"
ORANGE = "#F59E42"
ORANGE_LIGHT = "#FFF2E4"
PURPLE = "#7957D5"
PURPLE_LIGHT = "#F1ECFF"
RED = "#D95D67"

FONT_REGULAR_PATH = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
FONT_BOLD_PATH = "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc"


def font(size, bold=False):
    return ImageFont.truetype(FONT_BOLD_PATH if bold else FONT_REGULAR_PATH, size)


def canvas():
    image = Image.new("RGB", (WIDTH, HEIGHT), BG)
    draw = ImageDraw.Draw(image)
    draw.rounded_rectangle((60, 55, 1020, 1385), radius=42, fill=CARD)
    return image, draw


def text(draw, xy, content, size, fill=INK, bold=False, anchor=None, spacing=10):
    draw.multiline_text(
        xy,
        content,
        font=font(size, bold),
        fill=fill,
        anchor=anchor,
        spacing=spacing,
    )


def pill(draw, xy, content, fill=BLUE_LIGHT, color=BLUE):
    x, y = xy
    label_font = font(28, True)
    box = draw.textbbox((0, 0), content, font=label_font)
    width = box[2] - box[0] + 42
    draw.rounded_rectangle((x, y, x + width, y + 54), radius=27, fill=fill)
    draw.text((x + 21, y + 27), content, font=label_font, fill=color, anchor="lm")


def section_title(draw, kicker, title, subtitle=None):
    pill(draw, (105, 100), kicker)
    text(draw, (105, 190), title, 57, bold=True, spacing=12)
    if subtitle:
        text(draw, (105, 335), subtitle, 30, fill=MUTED, spacing=8)


def rounded_card(draw, box, fill="#FFFFFF", outline=LINE, radius=28, width=2):
    draw.rounded_rectangle(box, radius=radius, fill=fill, outline=outline, width=width)


def footer(draw, content="GridNav · Stage 3"):
    draw.line((105, 1310, 975, 1310), fill=LINE, width=2)
    text(draw, (105, 1340), content, 25, fill=MUTED)
    text(draw, (975, 1340), "学习项目阶段记录", 25, fill=MUTED, anchor="ra")


def paste_contain(base, source, box, background=CARD):
    x1, y1, x2, y2 = box
    target_w = x2 - x1
    target_h = y2 - y1
    copy = source.convert("RGB")
    copy.thumbnail((target_w, target_h), Image.Resampling.LANCZOS)
    x = x1 + (target_w - copy.width) // 2
    y = y1 + (target_h - copy.height) // 2
    ImageDraw.Draw(base).rectangle(box, fill=background)
    base.paste(copy, (x, y))


def save(image, name):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    image.save(OUTPUT_DIR / name, format="PNG", optimize=True)


def make_project_overview():
    image, draw = canvas()
    section_title(
        draw,
        "阶段性更新",
        "GridNav 学习项目\n阶段记录",
        "从 A* 到专家数据，再到传统机器学习 baseline",
    )

    stages = [
        ("01", "Stage 1", "GridWorld + A* 路径规划", BLUE, BLUE_LIGHT),
        ("02", "Stage 2", "随机地图专家数据生成", GREEN, GREEN_LIGHT),
        ("03", "Stage 3", "ML baseline 单步动作预测", PURPLE, PURPLE_LIGHT),
    ]
    y = 510
    for number, stage, label, color, light in stages:
        rounded_card(draw, (105, y, 975, y + 180), fill=light, outline=light)
        draw.ellipse((145, y + 42, 241, y + 138), fill=color)
        text(draw, (193, y + 90), number, 32, fill="#FFFFFF", bold=True, anchor="mm")
        text(draw, (285, y + 43), stage, 28, fill=color, bold=True)
        text(draw, (285, y + 91), label, 35, bold=True)
        y += 215

    rounded_card(draw, (105, 1170, 975, 1280), fill="#FFF8E8", outline="#FFF8E8")
    text(
        draw,
        (540, 1225),
        "目标不是堆复杂系统，而是把机器人学习基础一层层打穿。",
        29,
        fill="#7A5417",
        bold=True,
        anchor="mm",
    )
    footer(draw)
    save(image, "01_project_overview.png")


def make_astar_path():
    image, draw = canvas()
    section_title(
        draw,
        "Stage 1",
        "GridWorld + A*\n路径规划",
        "先理解搜索算法如何在已知地图中找到路径",
    )

    rounded_card(draw, (105, 445, 975, 1160), fill="#FAFBFF")
    source = Image.open(ASTAR_IMAGE)
    paste_contain(image, source, (145, 475, 935, 1120), background="#FAFBFF")

    rounded_card(draw, (105, 1190, 975, 1280), fill=BLUE_LIGHT, outline=BLUE_LIGHT)
    text(
        draw,
        (540, 1235),
        "A* 在已知 GridWorld 地图中搜索从 start 到 goal 的路径",
        29,
        fill=BLUE,
        bold=True,
        anchor="mm",
    )
    footer(draw)
    save(image, "02_astar_path.png")


def draw_arrow(draw, start, end, color=BLUE):
    draw.line((start, end), fill=color, width=8)
    x, y = end
    draw.polygon([(x, y), (x - 20, y - 14), (x - 20, y + 14)], fill=color)


def make_expert_dataset():
    image, draw = canvas()
    section_title(
        draw,
        "Stage 2",
        "把 A* 路径变成\n专家训练数据",
        "从规划结果中拆出可以监督学习的 state-action 样本",
    )

    pipeline = [
        (105, 450, 345, 600, "A* path", BLUE, BLUE_LIGHT),
        (420, 450, 660, 600, "state-action\nsamples", GREEN, GREEN_LIGHT),
        (735, 450, 975, 600, "expert_\ndataset.csv", PURPLE, PURPLE_LIGHT),
    ]
    for x1, y1, x2, y2, label, color, light in pipeline:
        rounded_card(draw, (x1, y1, x2, y2), fill=light, outline=light)
        text(draw, ((x1 + x2) // 2, (y1 + y2) // 2), label, 31, fill=color, bold=True, anchor="mm")
    draw_arrow(draw, (360, 525), (405, 525))
    draw_arrow(draw, (675, 525), (720, 525))

    stats = [
        ("100", "随机地图数量", BLUE, BLUE_LIGHT),
        ("98", "成功地图数量", GREEN, GREEN_LIGHT),
        ("970", "总样本数量", PURPLE, PURPLE_LIGHT),
    ]
    x = 105
    for value, label, color, light in stats:
        rounded_card(draw, (x, 665, x + 270, 825), fill=light, outline=light)
        text(draw, (x + 135, 715), value, 48, fill=color, bold=True, anchor="mm")
        text(draw, (x + 135, 775), label, 26, fill=MUTED, anchor="mm")
        x += 300

    text(draw, (105, 885), "Action 分布", 31, bold=True)
    actions = [
        ("up", 226, BLUE),
        ("down", 269, GREEN),
        ("left", 275, ORANGE),
        ("right", 200, PURPLE),
    ]
    y = 945
    for label, value, color in actions:
        text(draw, (105, y), label, 26, fill=MUTED)
        draw.rounded_rectangle((225, y + 5, 850, y + 34), radius=14, fill="#EEF1F6")
        bar_width = int(625 * value / 300)
        draw.rounded_rectangle((225, y + 5, 225 + bar_width, y + 34), radius=14, fill=color)
        text(draw, (925, y), str(value), 27, fill=INK, bold=True, anchor="ra")
        y += 65

    rounded_card(draw, (105, 1210, 975, 1285), fill=ORANGE_LIGHT, outline=ORANGE_LIGHT)
    text(
        draw,
        (540, 1247),
        "随机 start / goal，减少固定起终点造成的 down / right 偏置",
        27,
        fill="#965A18",
        bold=True,
        anchor="mm",
    )
    footer(draw)
    save(image, "03_expert_dataset.png")


def make_ml_accuracy():
    image, draw = canvas()
    section_title(
        draw,
        "Stage 3",
        "传统机器学习\n单步动作预测",
        "同一份专家数据，比较三个简单 baseline",
    )

    rounded_card(draw, (105, 445, 975, 1070), fill="#FAFBFF")
    chart_left = 190
    chart_bottom = 970
    chart_height = 400
    values = [
        ("KNN", 0.8144, BLUE),
        ("Logistic\nRegression", 0.8557, GREEN),
        ("Decision\nTree", 0.7732, PURPLE),
    ]
    positions = [280, 540, 800]
    for (label, value, color), center_x in zip(values, positions):
        bar_h = int(chart_height * value)
        draw.rounded_rectangle(
            (center_x - 75, chart_bottom - bar_h, center_x + 75, chart_bottom),
            radius=22,
            fill=color,
        )
        text(draw, (center_x, chart_bottom - bar_h - 45), f"{value:.4f}", 32, fill=color, bold=True, anchor="mm")
        text(draw, (center_x, chart_bottom + 45), label, 27, fill=INK, bold=True, anchor="ma", spacing=5)

    draw.line((160, chart_bottom, 920, chart_bottom), fill=LINE, width=3)
    pill(draw, (360, 1105), "最佳模型：Logistic Regression", fill=GREEN_LIGHT, color=GREEN)

    rounded_card(draw, (105, 1190, 975, 1285), fill="#FFF2F3", outline="#FFF2F3")
    text(
        draw,
        (540, 1238),
        "single-step action prediction accuracy\n≠ 完整导航成功率",
        29,
        fill=RED,
        bold=True,
        anchor="mm",
        spacing=6,
    )
    footer(draw)
    save(image, "04_ml_accuracy.png")


def crop_panel(source, index):
    panel_width = source.width // 3
    left = index * panel_width
    right = source.width if index == 2 else (index + 1) * panel_width
    return source.crop((left, 45, right, source.height))


def make_confusion_matrix():
    image, draw = canvas()
    section_title(
        draw,
        "Stage 3 · 评估",
        "混淆矩阵",
        "查看不同动作类别的预测情况",
    )

    source = Image.open(CONFUSION_IMAGE)
    panels = [crop_panel(source, index) for index in range(3)]
    boxes = [
        (105, 430, 520, 835),
        (560, 430, 975, 835),
        (330, 850, 750, 1255),
    ]
    for panel, box in zip(panels, boxes):
        rounded_card(draw, box, fill="#FAFBFF")
        x1, y1, x2, y2 = box
        paste_contain(image, panel, (x1 + 12, y1 + 12, x2 - 12, y2 - 12), background="#FAFBFF")

    text(
        draw,
        (540, 1280),
        "行是真实动作，列是预测动作；对角线越集中，分类越准确。",
        27,
        fill=MUTED,
        anchor="mm",
    )
    footer(draw)
    save(image, "05_confusion_matrix.png")


def make_next_steps():
    image, draw = canvas()
    section_title(
        draw,
        "继续推进",
        "下一步计划",
        "先验证当前阶段，再逐步进入更复杂的方法",
    )

    steps = [
        ("01", "继续理解 train/test split 和 map_id 划分", BLUE, BLUE_LIGHT),
        ("02", "分析混淆矩阵中哪些动作容易错", GREEN, GREEN_LIGHT),
        ("03", "做 rollout，测试模型能否从 start 走到 goal", ORANGE, ORANGE_LIGHT),
        ("04", "再进入 PyTorch 行为克隆", PURPLE, PURPLE_LIGHT),
        ("05", "后续补 Q-learning 和 C++ A*", "#4D718C", "#EAF3F8"),
    ]
    y = 450
    for number, label, color, light in steps:
        rounded_card(draw, (105, y, 975, y + 125), fill=light, outline=light)
        draw.ellipse((140, y + 27, 211, y + 98), fill=color)
        text(draw, (175, y + 62), number, 24, fill="#FFFFFF", bold=True, anchor="mm")
        text(draw, (245, y + 62), label, 29, fill=INK, bold=True, anchor="lm")
        y += 145

    rounded_card(draw, (105, 1190, 975, 1285), fill="#FFF2F3", outline="#FFF2F3")
    text(
        draw,
        (540, 1238),
        "单步预测 ≠ 完整导航，后续需要连续决策验证",
        30,
        fill=RED,
        bold=True,
        anchor="mm",
    )
    footer(draw)
    save(image, "06_next_steps.png")


def main():
    if not ASTAR_IMAGE.exists():
        raise FileNotFoundError(f"Missing source image: {ASTAR_IMAGE}")
    if not CONFUSION_IMAGE.exists():
        raise FileNotFoundError(f"Missing source image: {CONFUSION_IMAGE}")

    make_project_overview()
    make_astar_path()
    make_expert_dataset()
    make_ml_accuracy()
    make_confusion_matrix()
    make_next_steps()

    for path in sorted(OUTPUT_DIR.glob("*.png")):
        print(path.relative_to(ROOT))


if __name__ == "__main__":
    main()
