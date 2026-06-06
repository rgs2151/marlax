from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib import animation
from matplotlib.animation import PillowWriter
from matplotlib.patches import Circle, Ellipse, Polygon, Rectangle
from tqdm import tqdm


OUTPUT = Path("docs/source/_static/gallery/cooperative_grid.gif")


def setup_style():
    sns.set_theme(context="talk", style="ticks", palette="dark")
    plt.rcParams["font.family"] = "serif"
    plt.rcParams["mathtext.fontset"] = "cm"
    plt.rcParams["axes.spines.top"] = False
    plt.rcParams["axes.spines.right"] = False
    plt.rcParams["lines.linewidth"] = 1
    plt.rcParams["patch.linewidth"] = 0
    plt.rcParams["image.interpolation"] = "none"
    plt.rcParams["legend.frameon"] = False
    plt.rcParams["figure.figsize"] = [4.0, 4.0]
    plt.rcParams["figure.dpi"] = 120
    plt.rcParams["savefig.dpi"] = 120
    plt.rcParams["savefig.transparent"] = True


def interpolate_path(points, repeats=3):
    frames = []
    for start, end in zip(points[:-1], points[1:]):
        start = np.array(start, dtype=float)
        end = np.array(end, dtype=float)
        for step in range(repeats):
            t = step / repeats
            frames.append((1 - t) * start + t * end)
    frames.append(np.array(points[-1], dtype=float))
    return np.array(frames)


def demo_paths():
    center = (5, 5)
    reward = (10, 5)
    agent_1 = [(1, 2), (2, 2), (3, 3), (4, 4), center, (6, 5), (7, 5), (8, 5), (9, 5), reward]
    agent_2 = [(2, 8), (3, 8), (4, 7), (5, 6), center, (6, 5), (7, 5), (8, 5), (9, 5), reward]
    path_1 = interpolate_path(agent_1)
    path_2 = interpolate_path(agent_2)
    return path_1, path_2


def heading(path, index):
    if index == 0:
        delta = path[1] - path[0]
    else:
        delta = path[index] - path[index - 1]
    if np.linalg.norm(delta) == 0:
        return 0
    return np.degrees(np.arctan2(delta[1], delta[0])) - 90


def make_mouse(ax, color):
    body = Ellipse((0, 0), 0.58, 0.9, color=color, alpha=0.95, zorder=4)
    head = Circle((0, 0), 0.28, color=color, alpha=0.95, zorder=5)
    ear_1 = Circle((0, 0), 0.13, color=color, alpha=0.95, zorder=5)
    ear_2 = Circle((0, 0), 0.13, color=color, alpha=0.95, zorder=5)
    eye_1 = Circle((0, 0), 0.035, color="black", zorder=6)
    eye_2 = Circle((0, 0), 0.035, color="black", zorder=6)
    nose = Polygon([(0, 0), (0, 0), (0, 0)], color=color, zorder=6)
    tail, = ax.plot([], [], color=color, linewidth=2, alpha=0.9, zorder=3)
    for patch in [body, head, ear_1, ear_2, eye_1, eye_2, nose]:
        ax.add_patch(patch)
    return {
        "body": body,
        "head": head,
        "ear_1": ear_1,
        "ear_2": ear_2,
        "eye_1": eye_1,
        "eye_2": eye_2,
        "nose": nose,
        "tail": tail,
    }


def point(x, y, angle, distance, side=0):
    radians = np.radians(angle + 90 + side)
    return x + distance * np.cos(radians), y + distance * np.sin(radians)


def set_mouse(mouse, x, y, angle):
    mouse["body"].set_center((x, y))
    mouse["body"].angle = angle
    head_x, head_y = point(x, y, angle, 0.5)
    mouse["head"].set_center((head_x, head_y))
    mouse["ear_1"].set_center(point(head_x, head_y, angle, 0.27, 125))
    mouse["ear_2"].set_center(point(head_x, head_y, angle, 0.27, -125))
    mouse["eye_1"].set_center(point(head_x, head_y, angle, 0.22, 55))
    mouse["eye_2"].set_center(point(head_x, head_y, angle, 0.22, -55))
    nose_tip = point(head_x, head_y, angle, 0.43)
    nose_left = point(head_x, head_y, angle, 0.24, 12)
    nose_right = point(head_x, head_y, angle, 0.24, -12)
    mouse["nose"].set_xy([nose_tip, nose_left, nose_right])
    tail_start = point(x, y, angle, -0.38)
    tail_end = point(x, y, angle, -0.9)
    mouse["tail"].set_data([tail_start[0], tail_end[0]], [tail_start[1], tail_end[1]])


def heart_xy(cx, cy, size):
    t = np.linspace(0, 2 * np.pi, 100)
    x = cx + size * (16 * np.sin(t) ** 3) / 20
    y = cy + size * (13 * np.cos(t) - 5 * np.cos(2 * t) - 2 * np.cos(3 * t) - np.cos(4 * t)) / 20
    return np.column_stack([x, y + 0.4])


def build_animation(path_1, path_2):
    setup_style()
    fig, ax = plt.subplots(figsize=(4.8, 4.8), dpi=120)
    fig.subplots_adjust(left=0.02, right=0.98, bottom=0.02, top=0.98)
    ax.set_xlim(-0.7, 10.7)
    ax.set_ylim(-0.7, 10.7)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)

    for x in range(11):
        for y in range(11):
            ax.add_patch(Rectangle((x - 0.5, y - 0.5), 1, 1, facecolor="white", edgecolor="0.88", linewidth=0.7, zorder=0))

    center = ax.scatter([5], [5], marker="s", s=240, color="gold", alpha=0.75, edgecolors="white", linewidths=1.5, zorder=1)
    reward = ax.scatter([], [], marker="o", s=280, color="darkgreen", alpha=0.75, edgecolors="white", linewidths=1.5, zorder=1)
    heart = Polygon(np.empty((0, 2)), color="purple", alpha=0.0, zorder=8)
    ax.add_patch(heart)

    mouse_1 = make_mouse(ax, "darkred")
    mouse_2 = make_mouse(ax, "midnightblue")

    def update(frame):
        active = frame >= 13
        collected = frame >= len(path_1) - 4
        reward.set_offsets(np.array([[10, 5]]) if active else np.empty((0, 2)))
        center.set_offsets(np.empty((0, 2)) if active else np.array([[5, 5]]))

        set_mouse(mouse_1, path_1[frame, 0], path_1[frame, 1], heading(path_1, frame))
        set_mouse(mouse_2, path_2[frame, 0], path_2[frame, 1], heading(path_2, frame))

        if collected:
            cx = (path_1[frame, 0] + path_2[frame, 0]) / 2
            cy = (path_1[frame, 1] + path_2[frame, 1]) / 2
            heart.set_xy(heart_xy(cx, cy, 0.85))
            heart.set_alpha(0.95)
        else:
            heart.set_xy(np.empty((0, 2)))
            heart.set_alpha(0)

        return [center, reward, heart] + list(mouse_1.values()) + list(mouse_2.values())

    return animation.FuncAnimation(fig, update, frames=len(path_1), interval=140, blit=False)


def save_animation(ani, output, total):
    output.parent.mkdir(parents=True, exist_ok=True)
    with tqdm(total=total, desc="Saving cooperative grid GIF", unit="frame") as progress:
        ani.save(output, writer=PillowWriter(fps=7), progress_callback=lambda frame, count: progress.update(1))


def main():
    path_1, path_2 = demo_paths()
    ani = build_animation(path_1, path_2)
    save_animation(ani, OUTPUT, len(path_1))


if __name__ == "__main__":
    main()
