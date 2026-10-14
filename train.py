import os
import csv
import argparse
import numpy as np
import matplotlib.pyplot as plt
from DQN import DQNAgent
from DDQN import DDQNAgent
from rainbow import RainbowAgent
from forest_fire_env import ForestFireEnv
from actions import ActionType, decode_action


def parse_args():
    parser = argparse.ArgumentParser(description="Train RL agent on Wildfire Environment")
    parser.add_argument(
        '--algo',
        type=str,
        default='rainbow',
        choices=['dqn', 'ddqn', 'rainbow'],
        help='Algorithm to train: dqn, ddqn, or rainbow',
    )
    parser.add_argument('--episodes', type=int, default=10000)
    parser.add_argument('--grid-size', type=int, default=15)
    parser.add_argument('--window', type=int, default=100, help='Rolling window size for moving averages')
    return parser.parse_args()


def rolling_mean(data, window=100):
    """Compute a 1D rolling average with partial-window warmup for early episodes."""
    arr = np.asarray(data, dtype=np.float64)
    out = np.empty_like(arr)
    for i in range(len(arr)):
        start = max(0, i - window + 1)
        out[i] = np.mean(arr[start : i + 1])
    return out


def main():
    args = parse_args()

    os.makedirs("./checkpoints", exist_ok=True)
    os.makedirs("./plots", exist_ok=True)

    env = ForestFireEnv(grid_size=args.grid_size)

    action_size = env.action_space.n 
    num_spatial = 5 
    num_global = 5 
    
    if args.algo == 'rainbow':
        agent = RainbowAgent(action_size, args.grid_size, num_spatial, num_global)
    elif args.algo == 'ddqn':
        agent = DDQNAgent(action_size, args.grid_size, num_spatial, num_global)
    else:
        agent = DQNAgent(action_size, args.grid_size, num_spatial, num_global)

    target_update_freq = 10
    save_freq = 500
    w = args.window

    # Per-episode metric trackers
    episode_rewards = []
    episode_steps = []
    success_flags = []          # 1.0 if outcome == 'success' else 0.0
    preserved_fuel_pcts = []    # % of initial forest fuel remaining at episode end
    burnt_area_pcts = []        # % of grid cells burnt at episode end
    budget_spent_totals = []    # Total budget spent in episode

    # Decomposed reward component trackers
    comp_fire_damage = []
    comp_action_cost = []
    comp_terminal = []

    # Per-episode action type percentages: [NOOP, WATER_DROP, FIREBREAK, FIRE_TRUCK]
    action_Dist_history = {
        ActionType.NOOP: [],
        ActionType.WATER_DROP: [],
        ActionType.FIREBREAK: [],
        ActionType.FIRE_TRUCK: [],
    }

    print(f"Starting {args.algo.upper()} training on Wildfire Environment ({args.grid_size}x{args.grid_size})...")

    for episode in range(args.episodes):
        state, info = env.reset()
        total_reward = 0.0
        ep_steps = 0
        ep_budget_spent = 0.0
        ep_dmg_reward = 0.0
        ep_cost_reward = 0.0
        ep_term_reward = 0.0

        action_counts = {
            ActionType.NOOP: 0,
            ActionType.WATER_DROP: 0,
            ActionType.FIREBREAK: 0,
            ActionType.FIRE_TRUCK: 0,
        }

        done = False
        while not done:
            mask = env.action_masks()
            action = agent.act(state, action_mask=mask)

            # Record decoded action type
            spec = decode_action(int(action), args.grid_size)
            action_counts[spec.type] += 1

            next_state, reward, terminated, truncated, next_info = env.step(action)
            done = terminated or truncated
            next_mask = env.action_masks()

            if args.algo == 'rainbow' or done or np.sum(mask) > 1:
                agent.memory.push(state, action, reward, next_state, done, next_mask)
            agent.learn()

            state = next_state
            total_reward += reward
            ep_steps += 1

            # Accumulate step metrics from info
            rm = next_info.get("reward_metrics", {})
            rc = next_info.get("reward_components", {})
            ep_budget_spent += rm.get("budget_spent", 0.0)
            ep_dmg_reward += rc.get("fire_damage", 0.0)
            ep_cost_reward += rc.get("action_cost", 0.0)
            ep_term_reward += rc.get("terminal", 0.0)

        # Decay epsilon (if applicable and past warmup)
        warmup = getattr(agent, 'warmup_steps', 0)
        if hasattr(agent, 'epsilon') and len(agent.memory) >= warmup:
            agent.epsilon = max(agent.epsilon_min, agent.epsilon * agent.epsilon_decay)

        # Fallback episode-based target sync if step-based target sync isn't configured
        if not hasattr(agent, 'target_update_steps') and (episode + 1) % target_update_freq == 0:
            agent.target_network.load_state_dict(agent.q_network.state_dict())

        # Record episode-level outcomes
        rm = next_info.get("reward_metrics", {})
        outcome = rm.get("outcome", "unknown")
        preserved_pct = rm.get("preserved_fraction", 0.0) * 100.0
        burnt_pct = (next_info.get("burnt_count", 0) / float(args.grid_size ** 2)) * 100.0

        episode_rewards.append(total_reward)
        episode_steps.append(ep_steps)
        success_flags.append(100.0 if outcome == "success" else 0.0)
        preserved_fuel_pcts.append(preserved_pct)
        burnt_area_pcts.append(burnt_pct)
        budget_spent_totals.append(ep_budget_spent)

        comp_fire_damage.append(ep_dmg_reward)
        comp_action_cost.append(ep_cost_reward)
        comp_terminal.append(ep_term_reward)

        for act_type in ActionType:
            pct = (action_counts[act_type] / max(1, ep_steps)) * 100.0
            action_Dist_history[act_type].append(pct)

        if (episode + 1) % 10 == 0:
            recent_rew = np.mean(episode_rewards[-w:])
            recent_win = np.mean(success_flags[-w:])
            recent_fuel = np.mean(preserved_fuel_pcts[-w:])
            noop_pct = action_Dist_history[ActionType.NOOP][-1]
            wd_pct = action_Dist_history[ActionType.WATER_DROP][-1]
            fb_pct = action_Dist_history[ActionType.FIREBREAK][-1]
            ft_pct = action_Dist_history[ActionType.FIRE_TRUCK][-1]

            eps_str = f" | Eps: {agent.epsilon:.3f}" if hasattr(agent, 'epsilon') else ""
            print(
                f"Ep: {episode + 1:5d} | Rew: {total_reward:6.2f} (Avg: {recent_rew:6.2f}) | "
                f"Win%: {recent_win:5.1f}% | FuelSaved: {recent_fuel:5.1f}% | Steps: {ep_steps:3d} | "
                f"Acts [N:{noop_pct:4.0f}% W:{wd_pct:4.0f}% FB:{fb_pct:4.0f}% FT:{ft_pct:4.0f}%]"
                f"{eps_str} | {outcome}"
            )

        if (episode + 1) % save_freq == 0:
            agent.save_checkpoint(f"./checkpoints/{args.algo}_wildfire_ep{episode + 1}.pth")

    env.close()

    # Save per-episode metrics to CSV for cross-algorithm plotting
    csv_path = f"./plots/{args.algo}_{args.grid_size}x{args.grid_size}_metrics.csv"
    with open(csv_path, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            "episode", "reward", "steps", "success", "preserved_fuel_pct",
            "burnt_area_pct", "budget_spent", "noop_pct", "water_drop_pct",
            "firebreak_pct", "fire_truck_pct", "rew_fire_damage",
            "rew_action_cost", "rew_terminal"
        ])
        for i in range(args.episodes):
            writer.writerow([
                i + 1,
                episode_rewards[i],
                episode_steps[i],
                success_flags[i],
                preserved_fuel_pcts[i],
                burnt_area_pcts[i],
                budget_spent_totals[i],
                action_Dist_history[ActionType.NOOP][i],
                action_Dist_history[ActionType.WATER_DROP][i],
                action_Dist_history[ActionType.FIREBREAK][i],
                action_Dist_history[ActionType.FIRE_TRUCK][i],
                comp_fire_damage[i],
                comp_action_cost[i],
                comp_terminal[i],
            ])
    print(f"Saved training metrics CSV to {csv_path}")

    # Generate 6-panel comparative diagnostic dashboard
    episodes_x = np.arange(1, args.episodes + 1)
    fig, axes = plt.subplots(3, 2, figsize=(16, 14), sharex=True)
    fig.suptitle(
        f"{args.algo.upper()} Wildfire Training Diagnostics ({args.grid_size}x{args.grid_size} Grid)",
        fontsize=15,
        fontweight="bold",
    )

    # Panel (0, 0): Episode Reward & Rolling Average
    ax = axes[0, 0]
    ax.plot(episodes_x, episode_rewards, alpha=0.3, color="royalblue", label="Episode Reward")
    ax.plot(episodes_x, rolling_mean(episode_rewards, w), linewidth=2, color="darkorange", label=f"{w}-Ep Moving Avg")
    ax.set_ylabel("Total Reward")
    ax.set_title("Episode Return")
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)

    # Panel (0, 1): Containment Success Rate & Preserved Fuel (%)
    ax = axes[0, 1]
    ax.plot(episodes_x, rolling_mean(success_flags, w), linewidth=2, color="forestgreen", label=f"Success Rate ({w}-Ep %)")
    ax.plot(episodes_x, rolling_mean(preserved_fuel_pcts, w), linewidth=2, color="teal", linestyle="--", label=f"Preserved Fuel ({w}-Ep %)")
    ax.plot(episodes_x, rolling_mean(burnt_area_pcts, w), linewidth=1.5, color="crimson", linestyle=":", label=f"Burnt Area ({w}-Ep %)")
    ax.set_ylabel("Percentage (%)")
    ax.set_ylim(-2, 105)
    ax.set_title("Containment Success Rate & Forest Preservation")
    ax.legend(loc="center right")
    ax.grid(True, alpha=0.3)

    # Panel (1, 0): Per-Episode Action Breakdown (Rolling %)
    ax = axes[1, 0]
    noop_ma = rolling_mean(action_Dist_history[ActionType.NOOP], w)
    wd_ma = rolling_mean(action_Dist_history[ActionType.WATER_DROP], w)
    fb_ma = rolling_mean(action_Dist_history[ActionType.FIREBREAK], w)
    ft_ma = rolling_mean(action_Dist_history[ActionType.FIRE_TRUCK], w)

    ax.stackplot(
        episodes_x,
        noop_ma,
        wd_ma,
        fb_ma,
        ft_ma,
        labels=["NOOP", "WATER_DROP", "FIREBREAK", "FIRE_TRUCK"],
        colors=["#9e9e9e", "#1f77b4", "#ff7f0e", "#d62728"],
        alpha=0.85,
    )
    ax.set_ylabel(f"Action Share ({w}-Ep %)")
    ax.set_ylim(0, 100)
    ax.set_title("Action Type Breakdown Over Training")
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3)

    # Panel (1, 1): Episode Length (Steps to Containment / Burnout)
    ax = axes[1, 1]
    ax.plot(episodes_x, episode_steps, alpha=0.25, color="slategray", label="Episode Steps")
    ax.plot(episodes_x, rolling_mean(episode_steps, w), linewidth=2, color="purple", label=f"{w}-Ep Moving Avg")
    ax.set_ylabel("Steps")
    ax.set_title("Episode Duration (Suppression Speed vs. Burnout)")
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3)

    # Panel (2, 0): Decomposed Reward Components
    ax = axes[2, 0]
    ax.plot(episodes_x, rolling_mean(comp_fire_damage, w), linewidth=2, color="firebrick", label="Fire Damage Penalty")
    ax.plot(episodes_x, rolling_mean(comp_action_cost, w), linewidth=2, color="goldenrod", label="Action Cost Penalty")
    ax.plot(episodes_x, rolling_mean(comp_terminal, w), linewidth=2, color="mediumseagreen", label="Terminal Bonus/Penalty")
    ax.set_xlabel("Episode")
    ax.set_ylabel(f"Component Return ({w}-Ep Avg)")
    ax.set_title("Decomposed Reward Components")
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)

    # Panel (2, 1): Resource Expenditure (Total Budget Spent per Episode)
    ax = axes[2, 1]
    ax.plot(episodes_x, budget_spent_totals, alpha=0.25, color="cadetblue", label="Episode Budget Spent")
    ax.plot(episodes_x, rolling_mean(budget_spent_totals, w), linewidth=2, color="navy", label=f"{w}-Ep Moving Avg")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Budget Units Spent")
    ax.set_title("Suppression Resource Expenditure per Episode")
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3)

    plt.tight_layout(rect=[0, 0.01, 1, 0.96])
    dashboard_path = f"./plots/{args.algo}_{args.grid_size}x{args.grid_size}_dashboard.png"
    plt.savefig(dashboard_path, dpi=150)
    print(f"Saved 6-panel training dashboard to {dashboard_path}")
    plt.close(fig)


if __name__ == "__main__":
    main()