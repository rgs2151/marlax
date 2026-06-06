from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib import animation
from matplotlib.animation import PillowWriter
from matplotlib.patches import Circle, Ellipse, Polygon, Rectangle
from tqdm import tqdm


OUTPUT = Path("docs/source/_static/gallery/cooperative_grid.gif")
GRID_SIZE = 11
CENTER = np.array([5.0, 5.0])
TARGETS = [
    np.array([10.0, 5.0]),
    np.array([5.0, 10.0]),
    np.array([0.0, 5.0]),
    np.array([5.0, 0.0]),
]
ACTIONS = np.array([
    [0.0, 0.0],
    [0.0, -1.0],
    [0.0, 1.0],
    [-1.0, 0.0],
    [1.0, 0.0],
])


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


def random_edge_start(random):
    side = random.integers(4)
    value = random.uniform(1.0, 9.0)
    if side == 0:
        return np.array([value, 0.8])
    if side == 1:
        return np.array([10.2, value])
    if side == 2:
        return np.array([value, 10.2])
    return np.array([0.8, value])


def sample_traits(random):
    return {
        "temperature": random.uniform(0.11, 0.22),
        "motor_noise": random.uniform(0.018, 0.045),
        "cohesion": random.uniform(0.05, 0.13),
        "inertia": random.uniform(0.04, 0.16),
        "curvature": random.uniform(-0.22, 0.22),
        "wander_rate": random.uniform(0.28, 0.58),
        "action_bias": random.normal(0, 0.05, len(ACTIONS)),
    }


def q_value(position, velocity, action, partner, goal, traits):
    next_position = np.clip(position + action * 0.34, 0, GRID_SIZE - 1)
    progress = -np.linalg.norm(next_position - goal) * 1.8
    pair_distance = -np.linalg.norm(next_position - partner) * traits["cohesion"]
    inertia = np.dot(action, velocity) * traits["inertia"]
    to_goal = goal - position
    goal_distance = max(0.1, np.linalg.norm(to_goal))
    tangent = np.array([-to_goal[1], to_goal[0]]) / goal_distance
    curvature = np.dot(action, tangent) * traits["curvature"]
    return progress + pair_distance + inertia + curvature


def choose_action(position, velocity, partner, goal, traits, random):
    scores = np.array([
        q_value(position, velocity, action, partner, goal, traits)
        for action in ACTIONS
    ])
    scores = scores + traits["action_bias"]
    weights = np.exp((scores - scores.max()) / traits["temperature"])
    probabilities = weights / weights.sum()
    return ACTIONS[random.choice(len(ACTIONS), p=probabilities)]


def simulate_trial(episode, random):
    target = TARGETS[random.integers(len(TARGETS))]
    positions = np.array([random_edge_start(random), random_edge_start(random)])
    velocities = np.zeros((2, 2))
    wander = np.zeros((2, 2))
    traits = [sample_traits(random), sample_traits(random)]
    frames = []
    active = False
    phase = 0
    collected_frames = 0

    for step in range(90):
        if step in (12, 31, 52):
            velocities[random.integers(2)] += random.normal(0, 0.85, 2)

        for agent in range(2):
            partner = 1 - agent
            goal = target if active else CENTER
            action = choose_action(
                positions[agent],
                velocities[agent],
                positions[partner],
                goal,
                traits[agent],
                random,
            )
            noise = random.normal(0, traits[agent]["motor_noise"], 2)
            wander[agent] = wander[agent] * 0.92 + random.normal(0, traits[agent]["motor_noise"], 2) * traits[agent]["wander_rate"]
            desired = action * 2.45 + noise + wander[agent]
            velocities[agent] = velocities[agent] * 0.72 + desired * 0.28

        positions = np.clip(positions + velocities * 0.2, 0, GRID_SIZE - 1)
        phase_start = False

        if not active and np.all(np.linalg.norm(positions - CENTER, axis=1) < 0.72):
            active = True
            phase = 1
            phase_start = True

        if active and np.all(np.linalg.norm(positions - target, axis=1) < 0.72):
            collected_frames += 1
        else:
            collected_frames = 0

        frames.append({
            "episode": episode,
            "phase": phase,
            "agent_1": positions[0].copy(),
            "agent_2": positions[1].copy(),
            "target": target,
            "active": active,
            "collected": collected_frames > 0,
            "episode_start": step == 0,
            "phase_start": phase_start,
        })

        if collected_frames > 5:
            break

    return frames


def build_trials():
    random = np.random.default_rng(25)
    frames = []
    for episode in tqdm(range(5), desc="Simulating cooperative grid", unit="trial"):
        trial_frames = simulate_trial(episode, random)
        frames.extend(trial_frames)
        last = trial_frames[-1]
        frames.extend([{
            **last,
            "episode_start": False,
            "phase_start": False,
        }] * 2)
    return frames


def heading(frames, key, index):
    if index == 0 or frames[index]["episode_start"] or frames[index]["phase_start"]:
        delta = frames[index + 1][key] - frames[index][key]
    else:
        delta = frames[index][key] - frames[index - 1][key]
    if np.linalg.norm(delta) == 0:
        return 0
    return np.degrees(np.arctan2(delta[1], delta[0])) - 90


def make_mouse(ax, color):
    body = Ellipse((0, 0), 0.58, 0.9, color=color, alpha=0.95, zorder=5)
    head = Circle((0, 0), 0.28, color=color, alpha=0.95, zorder=6)
    ear_1 = Circle((0, 0), 0.13, color=color, alpha=0.95, zorder=6)
    ear_2 = Circle((0, 0), 0.13, color=color, alpha=0.95, zorder=6)
    eye_1 = Circle((0, 0), 0.035, color="black", zorder=7)
    eye_2 = Circle((0, 0), 0.035, color="black", zorder=7)
    nose = Polygon([(0, 0), (0, 0), (0, 0)], color=color, zorder=7)
    tail, = ax.plot([], [], color=color, linewidth=2, alpha=0.9, zorder=4)
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
    t = np.linspace(0, 2 * np.pi, 120)
    x = cx + size * (16 * np.sin(t) ** 3) / 20
    y = cy + size * (13 * np.cos(t) - 5 * np.cos(2 * t) - 2 * np.cos(3 * t) - np.cos(4 * t)) / 20
    return np.column_stack([x, y + 0.4])


def build_animation(frames):
    setup_style()
    fig, ax = plt.subplots(figsize=(4.8, 4.8), dpi=120)
    fig.subplots_adjust(left=0.02, right=0.98, bottom=0.02, top=0.98)
    ax.set_xlim(-0.7, GRID_SIZE - 0.3)
    ax.set_ylim(-0.7, GRID_SIZE - 0.3)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)

    for x in range(GRID_SIZE):
        for y in range(GRID_SIZE):
            ax.add_patch(Rectangle((x - 0.5, y - 0.5), 1, 1, facecolor="white", edgecolor="0.88", linewidth=0.7, zorder=0))

    trail_1, = ax.plot([], [], color="darkred", alpha=0.24, linewidth=2, zorder=2)
    trail_2, = ax.plot([], [], color="midnightblue", alpha=0.24, linewidth=2, zorder=2)
    center = ax.scatter([CENTER[0]], [CENTER[1]], marker="s", s=250, color="gold", alpha=0.8, edgecolors="white", linewidths=1.4, zorder=1)
    reward = ax.scatter([], [], marker="o", s=300, color="darkgreen", alpha=0.78, edgecolors="white", linewidths=1.4, zorder=1)
    heart = Polygon(np.empty((0, 2)), color="purple", alpha=0.0, zorder=9)
    ax.add_patch(heart)

    mouse_1 = make_mouse(ax, "darkred")
    mouse_2 = make_mouse(ax, "midnightblue")

    def recent_points(index, key):
        start = max(0, index - 12)
        episode = frames[index]["episode"]
        phase = frames[index]["phase"]
        window = []
        for item in frames[start:index + 1]:
            if item["episode"] == episode and item["phase"] == phase:
                window.append(item[key])
        return np.array(window)

    def update(index):
        frame = frames[index]
        reward.set_offsets(np.array([frame["target"]]) if frame["active"] else np.empty((0, 2)))
        center.set_offsets(np.empty((0, 2)) if frame["active"] else np.array([CENTER]))

        points_1 = recent_points(index, "agent_1")
        points_2 = recent_points(index, "agent_2")
        trail_1.set_data(points_1[:, 0], points_1[:, 1])
        trail_2.set_data(points_2[:, 0], points_2[:, 1])

        set_mouse(mouse_1, frame["agent_1"][0], frame["agent_1"][1], heading(frames, "agent_1", index))
        set_mouse(mouse_2, frame["agent_2"][0], frame["agent_2"][1], heading(frames, "agent_2", index))

        if frame["collected"]:
            center_point = (frame["agent_1"] + frame["agent_2"]) / 2
            heart.set_xy(heart_xy(center_point[0], center_point[1], 0.9))
            heart.set_alpha(0.9)
        else:
            heart.set_xy(np.empty((0, 2)))
            heart.set_alpha(0)

        artists = [trail_1, trail_2, center, reward, heart]
        artists.extend(mouse_1.values())
        artists.extend(mouse_2.values())
        return artists

    return animation.FuncAnimation(fig, update, frames=len(frames), interval=115, blit=False)


def save_animation(ani, output, total):
    output.parent.mkdir(parents=True, exist_ok=True)
    with tqdm(total=total, desc="Saving cooperative grid GIF", unit="frame") as progress:
        ani.save(output, writer=PillowWriter(fps=9), progress_callback=lambda frame, count: progress.update(1))


def main():
    frames = build_trials()
    ani = build_animation(frames)
    save_animation(ani, OUTPUT, len(frames))


if __name__ == "__main__":
    main()
