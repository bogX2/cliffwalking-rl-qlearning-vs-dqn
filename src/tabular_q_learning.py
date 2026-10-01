import gymnasium as gym
import numpy as np
import pickle
import matplotlib.pyplot as plt
from dataclasses import dataclass
from datetime import datetime
import time
import os
import csv
import random


@dataclass
class Config:
    learning_rate: float = 0.1
    epsilon_decay: float = 1.2
    starting_e: float = 1.0
    discount_factor: float = 0.99
    episodes: int = 5000
    max_episode_steps: int = 300
    is_slippery: bool = True
    eval_episodes: int = 100
    exp_name: str = "baseline"  # short label used in folder name + summary table


def train(cfg: Config):
    env = gym.make("CliffWalking-v1", max_episode_steps=cfg.max_episode_steps, is_slippery=cfg.is_slippery)
    q_table = np.zeros((env.observation_space.n, env.action_space.n))
    epsilon = cfg.starting_e

    metrics = {
        'episode_rewards': [],
        'episode_lengths': [],
        'successful_episode': [],
        'exploration_rate': []
    }

    for i in range(cfg.episodes):
        state = env.reset()[0]
        terminated = False
        truncated = False
        episode_reward = 0
        episode_length = 0

        while not terminated and not truncated:
            # epsilon-greedy action selection
            if random.random() < epsilon:
                action = env.action_space.sample()
            else:
                action = np.argmax(q_table[state, :])

            new_state, reward, terminated, truncated, _ = env.step(action)

            # non-deterministic Bellman update (TD-update with learning rate, slide 48)
            q_table[state, action] = q_table[state, action] + cfg.learning_rate * (
                reward + cfg.discount_factor * np.max(q_table[new_state, :]) - q_table[state, action]
            )

            episode_reward += reward
            episode_length += 1
            state = new_state

        # linear epsilon decay
        epsilon = max(epsilon - (cfg.epsilon_decay / cfg.episodes), 0.001)

        metrics['episode_rewards'].append(episode_reward)
        metrics['episode_lengths'].append(episode_length)
        # state 47 is the goal state
        metrics['successful_episode'].append(1 if terminated and state == 47 else 0)
        metrics['exploration_rate'].append(epsilon)

        if i % 250 == 0:
            print(f"Episode {i}: epsilon={epsilon:.3f}, episode_reward={episode_reward}")

    env.close()
    return q_table, metrics


def evaluate(q_table, cfg: Config):
    """Greedy-policy evaluation (no exploration), separate from training."""
    env = gym.make("CliffWalking-v1", max_episode_steps=cfg.max_episode_steps, is_slippery=cfg.is_slippery)
    rewards = []
    successes = []

    for _ in range(cfg.eval_episodes):
        state = env.reset()[0]
        terminated = False
        truncated = False
        episode_reward = 0

        while not terminated and not truncated:
            action = np.argmax(q_table[state, :])
            state, reward, terminated, truncated, _ = env.step(action)
            episode_reward += reward

        rewards.append(episode_reward)
        successes.append(1 if terminated and state == 47 else 0)

    env.close()
    return {
        'avg_reward': float(np.mean(rewards)),
        'success_rate': float(np.mean(successes) * 100)
    }


def plot_metrics(metrics, cfg: Config, path):
    plt.style.use('seaborn-v0_8' if 'seaborn-v0_8' in plt.style.available else 'default')
    fig = plt.figure(figsize=(16, 10))
    fig.suptitle(
        f"learning_rate={cfg.learning_rate}, epsilon_decay={cfg.epsilon_decay}, "
        f"discount_factor={cfg.discount_factor}, episodes={cfg.episodes}, "
        f"is_slippery={cfg.is_slippery}",
        fontsize=16, y=1.02
    )

    # 1. Reward per episode
    ax1 = plt.subplot(2, 2, 1)
    rewards = np.array(metrics['episode_rewards'])
    plt.plot(rewards, alpha=0.5, label='Episode Reward', color='blue')
    if len(rewards) > 50:
        w = 50
        mov_avg = np.convolve(rewards, np.ones(w) / w, mode='valid')
        plt.plot(range(w - 1, len(rewards)), mov_avg, color='red', linewidth=2,
                 label=f'Moving Avg ({w}): {np.mean(rewards[-w:]):.1f}')
    plt.axhline(y=-13, color='orange', linestyle='--', linewidth=2, label='Optimal value (-13)')
    plt.xlabel('Episode'); plt.ylabel('Reward'); plt.title('Reward per Episode')
    plt.legend(); plt.grid(True, alpha=0.3)

    # 2. Success rate
    ax2 = plt.subplot(2, 2, 2)
    success = np.array(metrics['successful_episode'])
    plt.plot(success, color='green', alpha=0.4, label='Success (raw)')
    if len(success) > 50:
        w = 50
        mov_avg = np.convolve(success, np.ones(w) / w, mode='valid')
        plt.plot(range(w - 1, len(success)), mov_avg, color='darkgreen', linewidth=2,
                  label=f'Moving Avg ({w})')
    plt.xlabel('Episode'); plt.ylabel('Success'); plt.title('Successful Episodes')
    plt.legend(); plt.grid(True, alpha=0.3)

    # 3. Episode length
    ax3 = plt.subplot(2, 2, 3)
    lengths = np.array(metrics['episode_lengths'])
    plt.plot(lengths, alpha=0.4, color='orange', label='Length (raw)')
    if len(lengths) > 50:
        w = 50
        mov_avg = np.convolve(lengths, np.ones(w) / w, mode='valid')
        plt.plot(range(w - 1, len(lengths)), mov_avg, color='red', linewidth=2,
                 label=f'Moving Avg ({w})')
    plt.xlabel('Episode'); plt.ylabel('Steps'); plt.title('Episode Length')
    plt.legend(); plt.grid(True, alpha=0.3)

    # 4. Epsilon decay
    ax4 = plt.subplot(2, 2, 4)
    plt.plot(metrics['exploration_rate'], color='red', linewidth=2)
    plt.xlabel('Episode'); plt.ylabel('Epsilon'); plt.title('Exploration Rate Decay')
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f'{path}/training_metrics.png', dpi=300, bbox_inches='tight')
    plt.close(fig)
    print("Plots saved")


def save_results(cfg: Config, metrics, eval_stats, elapsed, path):
    with open(f"{path}/results.txt", "w") as f:
        f.write("=== Parameters ===\n")
        f.write(f"learning_rate={cfg.learning_rate}\n")
        f.write(f"epsilon_decay={cfg.epsilon_decay}\n")
        f.write(f"starting_e={cfg.starting_e}\n")
        f.write(f"discount_factor={cfg.discount_factor}\n")
        f.write(f"episodes={cfg.episodes}\n")
        f.write(f"max_episode_steps={cfg.max_episode_steps}\n")
        f.write(f"is_slippery={cfg.is_slippery}\n")
        f.write(f"eval_episodes={cfg.eval_episodes}\n")
        f.write(f"\nTime to finish: {elapsed:.4f} s\n")

        f.write("\n=== Training (last 100 episodes) ===\n")
        f.write(f"Average reward: {np.mean(metrics['episode_rewards'][-100:]):.2f}\n")
        f.write(f"Success rate: {np.mean(metrics['successful_episode'][-100:]) * 100:.1f}%\n")

        f.write(f"\n=== Evaluation (greedy policy, {cfg.eval_episodes} episodes) ===\n")
        f.write(f"Average reward: {eval_stats['avg_reward']:.2f}\n")
        f.write(f"Success rate: {eval_stats['success_rate']:.1f}%\n")

    print(f"Results saved to {path}/results.txt")


def append_summary(cfg: Config, metrics, eval_stats, elapsed, path, summary_path="Q_table/summary.csv"):
    """Append one row to a cumulative CSV so all runs can be compared in a single table."""
    row = {
        'exp_name': cfg.exp_name,
        'folder': path,
        'learning_rate': cfg.learning_rate,
        'epsilon_decay': cfg.epsilon_decay,
        'discount_factor': cfg.discount_factor,
        'episodes': cfg.episodes,
        'max_episode_steps': cfg.max_episode_steps,
        'is_slippery': cfg.is_slippery,
        'train_avg_reward_last100': round(float(np.mean(metrics['episode_rewards'][-100:])), 2),
        'train_success_rate_last100': round(float(np.mean(metrics['successful_episode'][-100:]) * 100), 1),
        'eval_avg_reward': round(eval_stats['avg_reward'], 2),
        'eval_success_rate': round(eval_stats['success_rate'], 1),
        'time_s': round(elapsed, 2),
    }

    file_exists = os.path.isfile(summary_path)
    os.makedirs(os.path.dirname(summary_path), exist_ok=True)
    with open(summary_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)

    print(f"Summary row appended to {summary_path}")


if __name__ == "__main__":
    # === set parameters here manually for each test run ===
    cfg = Config(
        learning_rate=0.1, epsilon_decay=1.2, starting_e=1.0, discount_factor=0.9,
        episodes=5000, max_episode_steps=300, is_slippery=True, eval_episodes=100,
        exp_name="final_best_v2",
    )
    now = datetime.now().strftime("%Y-%m-%d__%H-%M-%S")
    path = f"Q_table/{cfg.exp_name}__{now}"
    os.makedirs(path, exist_ok=True)

    start = time.perf_counter()
    q_table, metrics = train(cfg)
    elapsed = time.perf_counter() - start

    eval_stats = evaluate(q_table, cfg)

    plot_metrics(metrics, cfg, path)
    save_results(cfg, metrics, eval_stats, elapsed, path)
    append_summary(cfg, metrics, eval_stats, elapsed, path)

    with open(f"{path}/q_table.pkl", "wb") as f:
        pickle.dump(q_table, f)

    print(f"\nDone. Results saved in {path}")
    print(f"Eval -> avg reward: {eval_stats['avg_reward']:.2f}, success rate: {eval_stats['success_rate']:.1f}%")