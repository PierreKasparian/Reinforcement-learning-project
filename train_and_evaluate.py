"""
Axel Bechtel
Jesper Wentzell
Pieree Kasparian
Ahmet Alper Basturk
"""

import os
import csv
import argparse
import random
import numpy as np
import torch
import matplotlib.pyplot as plt

from DQN import DQNAgent
from DDQN import DDQNAgent
from rainbow import RainbowAgent
from forest_fire_env import ForestFireEnv, CellState
from actions import ActionType, decode_action, line_cells
from buffer import device

ALGO_ORDER = ["dqn", "ddqn", "rainbow"]
ALGO_LABELS = {
    "dqn": "Pure DQN",
    "ddqn": "Pure DDQN",
    "rainbow": "Rainbow DQN",
}
ALGO_COLORS = {
    "dqn": "#0072B2",      # Blue
    "ddqn": "#E69F00",     # Orange
    "rainbow": "#009E73",  # Green
}

plt.rcParams.update({
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 11,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "figure.titlesize": 14,
})


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train and evaluate one RL agent at a time (or combine saved runs into report plots)"
    )
    parser.add_argument(
        "--algo",
        type=str,
        required=True,
        choices=["dqn", "ddqn", "rainbow", "all", "plot-only"],
        help="Run a single agent ('dqn', 'ddqn', 'rainbow'), 'all' sequentially, or 'plot-only' from saved CSVs",
    )
    parser.add_argument("--episodes", type=int, default=20000, help="Training episodes")
    parser.add_argument("--eval-episodes", type=int, default=200, help="Evaluation episodes")
    parser.add_argument("--grid-size", type=int, default=15, help="Grid dimension GxG")
    parser.add_argument("--window", type=int, default=100, help="Rolling window for training curves")
    parser.add_argument("--seed", type=int, default=42, help="Base random seed for reproducibility")
    parser.add_argument(
        "--skip-train",
        action="store_true",
        help="Skip training for the selected --algo and only run greedy evaluation + plotting",
    )
    return parser.parse_args()


def set_global_seeds(seed: int):
    # Sets global random seeds across Python, NumPy, and PyTorch for exact reproducibility
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def rolling_stats(data, window=100):
    """Computes rolling mean and rolling standard deviation for shaded confidence bands."""
    arr = np.asarray(data, dtype=np.float64)
    means = np.empty_like(arr)
    stds = np.empty_like(arr)
    for i in range(len(arr)):
        start = max(0, i - window + 1)
        segment = arr[start : i + 1]
        means[i] = np.mean(segment)
        stds[i] = np.std(segment)
    return means, stds


def clean_valid(values):
    """Filter out NaNs when computing statistics on conditional metrics."""
    arr = np.asarray(values, dtype=np.float64)
    return arr[~np.isnan(arr)]


def make_agent(algo: str, action_size: int, grid_size: int, num_spatial: int = 5, num_global: int = 6):
    if algo == "rainbow":
        return RainbowAgent(action_size, grid_size, num_spatial, num_global)
    elif algo == "ddqn":
        return DDQNAgent(action_size, grid_size, num_spatial, num_global)
    elif algo == "dqn":
        return DQNAgent(action_size, grid_size, num_spatial, num_global)
    raise ValueError(f"Unknown algorithm: {algo}")


@torch.no_grad()
def select_greedy_action(agent, algo: str, state: dict, action_mask: np.ndarray) -> int:
    """Performs deterministic greedy action selection during evaluation (disables epsilon exploration and NoisyNet noise)."""
    sp_tensor = torch.tensor(np.array(state["spatial"]), dtype=torch.float32, device=device).unsqueeze(0)
    gl_tensor = torch.tensor(np.array(state["global"]), dtype=torch.float32, device=device).unsqueeze(0)
    mask_tensor = torch.tensor(action_mask, dtype=torch.bool, device=device)

    agent.q_network.eval()
    if algo == "rainbow":
        dist = agent.q_network(sp_tensor, gl_tensor)
        q_values = (dist * agent.support).sum(2)
        q_values[0, ~mask_tensor] = -float("inf")
    else:
        q_values = agent.q_network(sp_tensor, gl_tensor)
        q_values[0, ~mask_tensor] = -1e8
    agent.q_network.train()

    return int(torch.argmax(q_values).item())


def check_spatial_precision(env: ForestFireEnv, spec) -> tuple[bool, bool]:
    """
    Evaluate whether an active intervention targeted the fire accurately:
      - WATER_DROP / FIRE_TRUCK: did it hit at least one actively BURNING cell?
      - FIREBREAK: was it placed within Chebyshev distance <= 2 of an actively BURNING cell?
    """
    burning_coords = np.argwhere(env.grid == CellState.BURNING)
    if len(burning_coords) == 0:
        return False, False

    if spec.type == ActionType.WATER_DROP:
        cells = line_cells(spec, env.grid_size, env.line_half_length)
        direct_hit = any(env.grid[r, c] == CellState.BURNING for r, c in cells)
        return direct_hit, False

    if spec.type == ActionType.FIRE_TRUCK:
        direct_hit = bool(env.grid[spec.r, spec.c] == CellState.BURNING)
        return direct_hit, False

    if spec.type == ActionType.FIREBREAK:
        cells = line_cells(spec, env.grid_size, env.line_half_length)
        for r, c in cells:
            dists = np.max(np.abs(burning_coords - np.array([r, c])), axis=1)
            if np.any(dists <= 2):
                return False, True
        return False, False

    return False, False


def train_single_agent(algo: str, args) -> dict:
    """Trains a single agent, saves checkpoints and training CSV telemetry, and returns the history dict."""
    set_global_seeds(args.seed)
    env = ForestFireEnv(grid_size=args.grid_size)
    action_size = env.action_space.n
    agent = make_agent(algo, action_size, args.grid_size, num_spatial=5, num_global=6)

    target_update_freq = 10
    save_freq = max(500, args.episodes // 10)
    w = args.window

    history = {
        "reward": [],
        "steps": [],
        "success": [],
        "preserved_fuel_pct": [],
        "burnt_area_pct": [],
        "budget_spent": [],
        "noop_pct": [],
        "water_drop_pct": [],
        "firebreak_pct": [],
        "fire_truck_pct": [],
        "rew_fire_damage": [],
        "rew_action_cost": [],
        "rew_suppression": [],
        "rew_terminal": [],
    }

    print(f"\n{'='*70}")
    print(f"Training {ALGO_LABELS[algo]} ({algo.upper()}) for {args.episodes} episodes...")
    print(f"{'='*70}")

    for episode in range(args.episodes):
        state, _ = env.reset(seed=args.seed + episode)
        total_reward = 0.0
        ep_steps = 0
        ep_budget = 0.0
        ep_dmg = 0.0
        ep_cost = 0.0
        ep_supp = 0.0
        ep_term = 0.0

        counts = {t: 0 for t in ActionType}
        done = False

        while not done:
            mask = env.action_masks()
            action = agent.act(state, action_mask=mask)

            spec = decode_action(int(action), args.grid_size)
            counts[spec.type] += 1

            next_state, reward, terminated, truncated, next_info = env.step(action)
            done = terminated or truncated
            next_mask = env.action_masks()

            if algo == "rainbow" or done or np.sum(mask) > 1:
                agent.memory.push(state, action, reward, next_state, done, next_mask)
            agent.learn()

            state = next_state
            total_reward += reward
            ep_steps += 1

            rm = next_info.get("reward_metrics", {})
            rc = next_info.get("reward_components", {})
            ep_budget += rm.get("budget_spent", 0.0)
            ep_dmg += rc.get("fire_damage", 0.0)
            ep_cost += rc.get("action_cost", 0.0)
            ep_supp += rc.get("suppression", 0.0)
            ep_term += rc.get("terminal", 0.0)

        warmup = getattr(agent, "warmup_steps", 0)
        if hasattr(agent, "epsilon") and len(agent.memory) >= warmup:
            agent.epsilon = max(agent.epsilon_min, agent.epsilon * agent.epsilon_decay)

        if not hasattr(agent, "target_update_steps") and (episode + 1) % target_update_freq == 0:
            agent.target_network.load_state_dict(agent.q_network.state_dict())

        rm = next_info.get("reward_metrics", {})
        outcome = rm.get("outcome", "unknown")
        preserved_pct = rm.get("preserved_fraction", 0.0) * 100.0
        burnt_pct = (next_info.get("burnt_count", 0) / float(args.grid_size ** 2)) * 100.0

        history["reward"].append(total_reward)
        history["steps"].append(ep_steps)
        history["success"].append(100.0 if outcome == "success" else 0.0)
        history["preserved_fuel_pct"].append(preserved_pct)
        history["burnt_area_pct"].append(burnt_pct)
        history["budget_spent"].append(ep_budget)
        history["noop_pct"].append((counts[ActionType.NOOP] / max(1, ep_steps)) * 100.0)
        history["water_drop_pct"].append((counts[ActionType.WATER_DROP] / max(1, ep_steps)) * 100.0)
        history["firebreak_pct"].append((counts[ActionType.FIREBREAK] / max(1, ep_steps)) * 100.0)
        history["fire_truck_pct"].append((counts[ActionType.FIRE_TRUCK] / max(1, ep_steps)) * 100.0)
        history["rew_fire_damage"].append(ep_dmg)
        history["rew_action_cost"].append(ep_cost)
        history["rew_suppression"].append(ep_supp)
        history["rew_terminal"].append(ep_term)

        if (episode + 1) % 50 == 0 or episode == 0:
            rec_rew = np.mean(history["reward"][-w:])
            rec_win = np.mean(history["success"][-w:])
            rec_fuel = np.mean(history["preserved_fuel_pct"][-w:])
            eps_str = f" | Eps: {agent.epsilon:.3f}" if hasattr(agent, "epsilon") else ""
            print(
                f"[{algo.upper()}] Ep {episode + 1:5d}/{args.episodes} | "
                f"Rew (Avg): {rec_rew:6.2f} | Win%: {rec_win:5.1f}% | "
                f"FuelSaved: {rec_fuel:5.1f}%{eps_str}"
            )

        if (episode + 1) % save_freq == 0:
            agent.save_checkpoint(f"./checkpoints/{algo}_wildfire_ep{episode + 1}.pth")

    final_ckpt = f"./checkpoints/{algo}_wildfire_final.pth"
    agent.save_checkpoint(final_ckpt)
    env.close()

    csv_path = f"./results/train_{algo}_{args.grid_size}x{args.grid_size}.csv"
    keys = list(history.keys())
    with open(csv_path, mode="w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["episode"] + keys)
        for i in range(args.episodes):
            writer.writerow([i + 1] + [history[k][i] for k in keys])
    print(f"Saved {ALGO_LABELS[algo]} checkpoint to {final_ckpt} and training CSV to {csv_path}")

    return history


def evaluate_single_agent(algo: str, args) -> dict:
    """Evaluates a trained agent greedily across fixed paired test seeds and saves evaluation CSV metrics."""
    env = ForestFireEnv(grid_size=args.grid_size)
    action_size = env.action_space.n
    agent = make_agent(algo, action_size, args.grid_size, num_spatial=5, num_global=6)

    final_ckpt = f"./checkpoints/{algo}_wildfire_final.pth"
    if not os.path.exists(final_ckpt):
        ckpts = sorted(
            [f for f in os.listdir("./checkpoints") if f.startswith(f"{algo}_wildfire_ep")],
            key=lambda x: int(x.split("ep")[1].split(".")[0]),
        )
        if not ckpts:
            raise FileNotFoundError(f"No checkpoint found for {algo} in ./checkpoints/")
        final_ckpt = os.path.join("./checkpoints", ckpts[-1])

    agent.load_checkpoint(final_ckpt)
    print(f"Evaluating {ALGO_LABELS[algo]} from {final_ckpt} over {args.eval_episodes} paired seeds...")

    eval_metrics = {
        "reward": [],
        "steps": [],
        "steps_on_success": [],
        "success": [],
        "preserved_fuel_pct": [],
        "burnt_area_pct": [],
        "budget_spent": [],
        "noop_pct": [],
        "water_drop_pct": [],
        "firebreak_pct": [],
        "fire_truck_pct": [],
        "water_truck_hit_rate": [],
        "firebreak_near_front_rate": [],
        "rew_fire_damage": [],
        "rew_action_cost": [],
        "rew_suppression": [],
        "rew_terminal": [],
    }

    eval_base_seed = args.seed + 500_000

    for ep in range(args.eval_episodes):
        state, _ = env.reset(seed=eval_base_seed + ep)
        done = False
        total_reward = 0.0
        ep_steps = 0
        ep_budget = 0.0
        ep_dmg = 0.0
        ep_cost = 0.0
        ep_supp = 0.0
        ep_term = 0.0

        counts = {t: 0 for t in ActionType}
        water_truck_attempts = 0
        water_truck_hits = 0
        fb_attempts = 0
        fb_near_front = 0

        while not done:
            mask = env.action_masks()
            action = select_greedy_action(agent, algo, state, mask)
            spec = decode_action(action, args.grid_size)
            counts[spec.type] += 1

            wt_hit, fb_near = check_spatial_precision(env, spec)
            if spec.type in (ActionType.WATER_DROP, ActionType.FIRE_TRUCK):
                water_truck_attempts += 1
                if wt_hit:
                    water_truck_hits += 1
            elif spec.type == ActionType.FIREBREAK:
                fb_attempts += 1
                if fb_near:
                    fb_near_front += 1

            next_state, reward, terminated, truncated, next_info = env.step(action)
            done = terminated or truncated
            state = next_state

            total_reward += reward
            ep_steps += 1
            rm = next_info.get("reward_metrics", {})
            rc = next_info.get("reward_components", {})
            ep_budget += rm.get("budget_spent", 0.0)
            ep_dmg += rc.get("fire_damage", 0.0)
            ep_cost += rc.get("action_cost", 0.0)
            ep_supp += rc.get("suppression", 0.0)
            ep_term += rc.get("terminal", 0.0)

        rm = next_info.get("reward_metrics", {})
        outcome = rm.get("outcome", "unknown")
        is_win = outcome == "success"
        preserved_pct = rm.get("preserved_fraction", 0.0) * 100.0
        burnt_pct = (next_info.get("burnt_count", 0) / float(args.grid_size ** 2)) * 100.0

        eval_metrics["reward"].append(total_reward)
        eval_metrics["steps"].append(ep_steps)
        eval_metrics["steps_on_success"].append(float(ep_steps) if is_win else float("nan"))
        eval_metrics["success"].append(100.0 if is_win else 0.0)
        eval_metrics["preserved_fuel_pct"].append(preserved_pct)
        eval_metrics["burnt_area_pct"].append(burnt_pct)
        eval_metrics["budget_spent"].append(ep_budget)

        eval_metrics["noop_pct"].append((counts[ActionType.NOOP] / max(1, ep_steps)) * 100.0)
        eval_metrics["water_drop_pct"].append((counts[ActionType.WATER_DROP] / max(1, ep_steps)) * 100.0)
        eval_metrics["firebreak_pct"].append((counts[ActionType.FIREBREAK] / max(1, ep_steps)) * 100.0)
        eval_metrics["fire_truck_pct"].append((counts[ActionType.FIRE_TRUCK] / max(1, ep_steps)) * 100.0)

        wt_rate = (water_truck_hits / water_truck_attempts) * 100.0 if water_truck_attempts > 0 else float("nan")
        fb_rate = (fb_near_front / fb_attempts) * 100.0 if fb_attempts > 0 else float("nan")
        eval_metrics["water_truck_hit_rate"].append(wt_rate)
        eval_metrics["firebreak_near_front_rate"].append(fb_rate)

        eval_metrics["rew_fire_damage"].append(ep_dmg)
        eval_metrics["rew_action_cost"].append(ep_cost)
        eval_metrics["rew_suppression"].append(ep_supp)
        eval_metrics["rew_terminal"].append(ep_term)

    env.close()

    # Save per-episode evaluation metrics to CSV so results persist across separate runs
    eval_csv_path = f"./results/eval_{algo}_{args.grid_size}x{args.grid_size}.csv"
    keys = list(eval_metrics.keys())
    with open(eval_csv_path, mode="w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["eval_episode"] + keys)
        for i in range(args.eval_episodes):
            writer.writerow([i + 1] + [eval_metrics[k][i] for k in keys])
    print(f"Saved {ALGO_LABELS[algo]} evaluation CSV to {eval_csv_path}")

    return eval_metrics


def load_csv_dict(csv_path: str, index_col: str) -> dict:
    data = {}
    with open(csv_path, mode="r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            for k, v in row.items():
                if k == index_col:
                    continue
                data.setdefault(k, []).append(float(v))
    return data


def load_available_results(args) -> tuple[dict, dict]:
    """Scan ./results/ and load all completed training and evaluation CSVs."""
    all_train = {}
    all_eval = {}
    for algo in ALGO_ORDER:
        t_path = f"./results/train_{algo}_{args.grid_size}x{args.grid_size}.csv"
        e_path = f"./results/eval_{algo}_{args.grid_size}x{args.grid_size}.csv"
        if os.path.exists(t_path):
            all_train[algo] = load_csv_dict(t_path, index_col="episode")
        if os.path.exists(e_path):
            all_eval[algo] = load_csv_dict(e_path, index_col="eval_episode")
    return all_train, all_eval


# REPORT-READY STANDALONE PLOTTING FUNCTIONS (300 DPI)

def plot_training_curve(all_train: dict, key: str, ylabel: str, title: str, filename: str, window: int, ylim=None):
    """Generates a publication-ready training curve plot with rolling means and shaded standard deviation bands."""
    if not all_train:
        return
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    for algo in ALGO_ORDER:
        if algo not in all_train:
            continue
        data = all_train[algo][key]
        episodes_x = np.arange(1, len(data) + 1)
        means, stds = rolling_stats(data, window=window)
        ax.plot(episodes_x, means, label=ALGO_LABELS[algo], color=ALGO_COLORS[algo], linewidth=2.0)
        ax.fill_between(
            episodes_x,
            means - 0.5 * stds,
            means + 0.5 * stds,
            color=ALGO_COLORS[algo],
            alpha=0.15,
        )

    ax.set_xlabel("Training Episode")
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontweight="bold", pad=10)
    if ylim is not None:
        ax.set_ylim(*ylim)
    ax.legend(loc="best", frameon=True)
    ax.grid(True, linestyle="--", alpha=0.35)
    fig.tight_layout()
    fig.savefig(f"./report_figures/{filename}", dpi=300)
    plt.close(fig)


def plot_training_action_evolution(all_train: dict, window: int):
    """Plots stacked area charts showing how action type selection evolves over training episodes."""
    avail = [a for a in ALGO_ORDER if a in all_train]
    if not avail:
        return
    n = len(avail)
    fig, axes = plt.subplots(1, n, figsize=(4.6 * n, 4.2), sharey=True)
    if n == 1:
        axes = [axes]

    for idx, algo in enumerate(avail):
        ax = axes[idx]
        tr = all_train[algo]
        episodes_x = np.arange(1, len(tr["reward"]) + 1)
        noop_m, _ = rolling_stats(tr["noop_pct"], window)
        wd_m, _ = rolling_stats(tr["water_drop_pct"], window)
        fb_m, _ = rolling_stats(tr["firebreak_pct"], window)
        ft_m, _ = rolling_stats(tr["fire_truck_pct"], window)

        ax.stackplot(
            episodes_x,
            noop_m,
            wd_m,
            fb_m,
            ft_m,
            labels=["NOOP", "Water Drop", "Firebreak", "Fire Truck"],
            colors=["#9e9e9e", "#1f77b4", "#ff7f0e", "#d62728"],
            alpha=0.85,
        )
        ax.set_title(ALGO_LABELS[algo], fontweight="bold")
        ax.set_xlabel("Training Episode")
        ax.set_ylim(0, 100)
        ax.grid(True, linestyle="--", alpha=0.25)
        if idx == 0:
            ax.set_ylabel(f"Action Share ({window}-Ep Rolling %)")
        if idx == n - 1:
            ax.legend(loc="upper right", frameon=True)

    fig.suptitle("Evolution of Action Selection During Training", fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig("./report_figures/fig4_training_action_evolution.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_eval_outcome_bars(all_eval: dict):
    """Generates grouped bar charts comparing evaluation success rates and fuel preservation."""
    avail = [a for a in ALGO_ORDER if a in all_eval]
    if not avail:
        return
    metrics = ["success", "preserved_fuel_pct", "burnt_area_pct"]
    metric_labels = ["Containment Success (%)", "Preserved Fuel (%)", "Burnt Area (%)"]

    x = np.arange(len(metrics))
    width = 0.25

    fig, ax = plt.subplots(figsize=(7.5, 4.6))
    for i, algo in enumerate(avail):
        means = [np.mean(all_eval[algo][m]) for m in metrics]
        cis = [1.96 * np.std(all_eval[algo][m]) / np.sqrt(len(all_eval[algo][m])) for m in metrics]
        offset = (i - (len(avail) - 1) / 2.0) * width
        bars = ax.bar(
            x + offset,
            means,
            width,
            yerr=cis,
            capsize=4,
            label=ALGO_LABELS[algo],
            color=ALGO_COLORS[algo],
            edgecolor="black",
            linewidth=0.6,
            alpha=0.9,
        )
        for bar in bars:
            h = bar.get_height()
            ax.annotate(
                f"{h:.1f}%",
                xy=(bar.get_x() + bar.get_width() / 2, h),
                xytext=(0, 6),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=8.5,
            )

    ax.set_xticks(x)
    ax.set_xticklabels(metric_labels, fontweight="bold")
    ax.set_ylabel("Percentage (%)")
    ax.set_ylim(0, 112)
    ax.set_title("Greedy Evaluation Performance Across Paired Test Seeds", fontweight="bold", pad=10)
    ax.legend(loc="upper right", frameon=True)
    ax.grid(True, axis="y", linestyle="--", alpha=0.35)
    fig.tight_layout()
    fig.savefig("./report_figures/fig5_eval_outcome_summary.png", dpi=300)
    plt.close(fig)


def plot_eval_fuel_distribution(all_eval: dict):
    """Generates box plots illustrating the distribution of preserved forest fuel across algorithms."""
    avail = [a for a in ALGO_ORDER if a in all_eval]
    if not avail:
        return
    fig, ax = plt.subplots(figsize=(7.0, 4.5))
    positions = list(range(1, len(avail) + 1))
    data = [all_eval[algo]["preserved_fuel_pct"] for algo in avail]
    labels = [ALGO_LABELS[algo] for algo in avail]

    bp = ax.boxplot(
        data,
        positions=positions,
        widths=0.45,
        patch_artist=True,
        showmeans=True,
        meanprops={"marker": "D", "markerfacecolor": "white", "markeredgecolor": "black"},
        medianprops={"color": "black", "linewidth": 2.0},
    )
    for patch, algo in zip(bp["boxes"], avail):
        patch.set_facecolor(ALGO_COLORS[algo])
        patch.set_alpha(0.65)

    rng = np.random.default_rng(42)
    for i, algo in zip(positions, avail):
        pts = all_eval[algo]["preserved_fuel_pct"]
        jitter = rng.uniform(-0.12, 0.12, size=len(pts))
        ax.scatter(
            np.full(len(pts), i) + jitter,
            pts,
            color=ALGO_COLORS[algo],
            edgecolor="black",
            linewidth=0.3,
            alpha=0.5,
            s=18,
        )

    ax.axhline(40.0, color="crimson", linestyle="--", linewidth=1.2, label="Success Threshold (40% Fuel)")
    ax.set_xticks(positions)
    ax.set_xticklabels(labels, fontweight="bold")
    ax.set_ylabel("Preserved Forest Fuel (%)")
    ax.set_ylim(-3, 105)
    ax.set_title("Distribution of Preserved Forest Fuel in Evaluation", fontweight="bold", pad=10)
    ax.legend(loc="lower right", frameon=True)
    ax.grid(True, axis="y", linestyle="--", alpha=0.35)
    fig.tight_layout()
    fig.savefig("./report_figures/fig6_eval_fuel_distribution.png", dpi=300)
    plt.close(fig)


def plot_eval_tactical_precision(all_eval: dict):
    """Generates comparative plots for action share distribution and spatial targeting precision."""
    avail = [a for a in ALGO_ORDER if a in all_eval]
    if not avail:
        return
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.0, 4.4))

    action_keys = ["noop_pct", "water_drop_pct", "firebreak_pct", "fire_truck_pct"]
    action_names = ["NOOP", "Water Drop", "Firebreak", "Fire Truck"]
    action_colors = ["#9e9e9e", "#1f77b4", "#ff7f0e", "#d62728"]

    x = np.arange(len(avail))
    bottoms = np.zeros(len(avail))
    for k, name, col in zip(action_keys, action_names, action_colors):
        vals = np.array([np.mean(all_eval[algo][k]) for algo in avail])
        ax1.bar(x, vals, width=0.5, bottom=bottoms, label=name, color=col, edgecolor="black", linewidth=0.5)
        bottoms += vals

    ax1.set_xticks(x)
    ax1.set_xticklabels([ALGO_LABELS[a] for a in avail], fontweight="bold")
    ax1.set_ylabel("Mean Action Share (%)")
    ax1.set_ylim(0, 100)
    ax1.set_title("Evaluation Action Distribution", fontweight="bold")
    ax1.legend(loc="upper right", frameon=True)
    ax1.grid(True, axis="y", linestyle="--", alpha=0.3)

    prec_metrics = ["water_truck_hit_rate", "firebreak_near_front_rate"]
    prec_labels = ["Water/Truck Direct\nFire Hit Rate (%)", "Firebreak Placed\nNear Fire Front (%)"]
    px = np.arange(len(prec_metrics))
    width = 0.25

    for i, algo in enumerate(avail):
        means = []
        for m in prec_metrics:
            valid = clean_valid(all_eval[algo][m])
            means.append(np.mean(valid) if len(valid) > 0 else 0.0)

        offset = (i - (len(avail) - 1) / 2.0) * width
        bars = ax2.bar(
            px + offset,
            means,
            width,
            label=ALGO_LABELS[algo],
            color=ALGO_COLORS[algo],
            edgecolor="black",
            linewidth=0.6,
        )
        for bar in bars:
            h = bar.get_height()
            ax2.annotate(
                f"{h:.1f}%",
                xy=(bar.get_x() + bar.get_width() / 2, h),
                xytext=(0, 4),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=8.5,
            )

    ax2.set_xticks(px)
    ax2.set_xticklabels(prec_labels, fontweight="bold")
    ax2.set_ylabel("Spatial Targeting Accuracy (%)")
    ax2.set_ylim(0, 110)
    ax2.set_title("Spatial Targeting Precision", fontweight="bold")
    ax2.legend(loc="upper right", frameon=True)
    ax2.grid(True, axis="y", linestyle="--", alpha=0.3)

    fig.tight_layout()
    fig.savefig("./report_figures/fig7_eval_tactical_precision.png", dpi=300)
    plt.close(fig)


def export_summary_table(all_eval: dict, args):
    """Exports structured evaluation summary tables in both CSV and Markdown formats."""
    avail = [a for a in ALGO_ORDER if a in all_eval]
    if not avail:
        return
    rows = []
    for algo in avail:
        ev = all_eval[algo]
        win_steps = clean_valid(ev["steps_on_success"])
        win_steps_str = f"{np.mean(win_steps):.1f} ± {np.std(win_steps):.1f}" if len(win_steps) > 0 else "N/A"
        wt_hit = clean_valid(ev["water_truck_hit_rate"])
        fb_near = clean_valid(ev["firebreak_near_front_rate"])

        rows.append({
            "Algorithm": ALGO_LABELS[algo],
            "Episode Return": f"{np.mean(ev['reward']):.2f} ± {np.std(ev['reward']):.2f}",
            "Success Rate (%)": f"{np.mean(ev['success']):.1f}%",
            "Preserved Fuel (%)": f"{np.mean(ev['preserved_fuel_pct']):.1f} ± {np.std(ev['preserved_fuel_pct']):.1f}%",
            "Burnt Area (%)": f"{np.mean(ev['burnt_area_pct']):.1f} ± {np.std(ev['burnt_area_pct']):.1f}%",
            "Steps (Successful Eps)": win_steps_str,
            "Budget Spent": f"{np.mean(ev['budget_spent']):.1f} ± {np.std(ev['budget_spent']):.1f}",
            "Direct Fire Hit (%)": f"{np.mean(wt_hit):.1f}%" if len(wt_hit) > 0 else "0.0%",
            "Firebreak Near Front (%)": f"{np.mean(fb_near):.1f}%" if len(fb_near) > 0 else "0.0%",
        })

    csv_out = f"./results/eval_summary_{args.grid_size}x{args.grid_size}.csv"
    headers = list(rows[0].keys())
    with open(csv_out, mode="w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        writer.writerows(rows)

    md_out = f"./results/eval_summary_{args.grid_size}x{args.grid_size}.md"
    with open(md_out, mode="w") as f:
        f.write(f"# Wildfire Suppression Evaluation Summary ({args.eval_episodes} Paired Seeds)\n\n")
        f.write("| " + " | ".join(headers) + " |\n")
        f.write("|" + "|".join(["---"] * len(headers)) + "|\n")
        for r in rows:
            f.write("| " + " | ".join(r[h] for h in headers) + " |\n")

    print(f"\nUpdated evaluation summary tables ({', '.join(a.upper() for a in avail)}):")
    print(f"  - {csv_out}\n  - {md_out}\n")
    with open(md_out, "r") as f:
        print(f.read())


def main():
    args = parse_args()
    os.makedirs("./checkpoints", exist_ok=True)
    os.makedirs("./results", exist_ok=True)
    os.makedirs("./report_figures", exist_ok=True)

    # Execute training and evaluation pipeline for selected algorithm(s)
    if args.algo != "plot-only":
        target_algos = ALGO_ORDER if args.algo == "all" else [args.algo]
        for algo in target_algos:
            if not args.skip_train:
                train_single_agent(algo, args)
            evaluate_single_agent(algo, args)

    # Load all completed CSVs from ./results/ and generate publication-ready figures
    all_train, all_eval = load_available_results(args)
    completed_train = [ALGO_LABELS[a] for a in ALGO_ORDER if a in all_train]
    completed_eval = [ALGO_LABELS[a] for a in ALGO_ORDER if a in all_eval]
    print(f"\nGenerating report figures for available training runs: {completed_train}")
    print(f"Generating report figures for available evaluation runs: {completed_eval}")

    plot_training_curve(
        all_train,
        key="reward",
        ylabel=f"Episode Return ({args.window}-Ep Moving Avg)",
        title="Training Learning Curve: Episode Return",
        filename="fig1_training_returns.png",
        window=args.window,
    )
    plot_training_curve(
        all_train,
        key="success",
        ylabel=f"Containment Success Rate ({args.window}-Ep %)",
        title="Training Progression: Fire Containment Success Rate",
        filename="fig2_training_success_rate.png",
        window=args.window,
        ylim=(-2, 105),
    )
    plot_training_curve(
        all_train,
        key="preserved_fuel_pct",
        ylabel=f"Preserved Forest Fuel ({args.window}-Ep %)",
        title="Training Progression: Preserved Forest Fuel",
        filename="fig3_training_preserved_fuel.png",
        window=args.window,
        ylim=(-2, 105),
    )
    plot_training_action_evolution(all_train, window=args.window)

    plot_eval_outcome_bars(all_eval)
    plot_eval_fuel_distribution(all_eval)
    plot_eval_tactical_precision(all_eval)
    export_summary_table(all_eval, args)

    print("Saved updated figures to ./report_figures/")


if __name__ == "__main__":
    main()
