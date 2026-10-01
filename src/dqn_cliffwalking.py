import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
from datetime import datetime
from collections import deque
import random
import time
import os
import csv
import torch
from torch import nn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


@dataclass
class Config:
    learning_rate: float = 0.001
    epsilon_decay: float = 1.5
    starting_e: float = 1.0
    discount_factor: float = 0.99
    episodes: int = 2500
    max_episode_steps: int = 300
    is_slippery: bool = True
    eval_episodes: int = 100
    batch_size: int = 256
    buffer_size: int = 8192
    nn_nodes: int = 32
    exp_name: str = "baseline"  # short label used in folder name + summary table


class DQN(nn.Module):
    """
    Single Q-network (no target network, no D-DQN: not required by the project).
    Two hidden layers, tunable width (nn_nodes).
    """
    def __init__(self, n_states, nn_nodes, n_actions):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(n_states, nn_nodes),
            nn.ReLU(),
            nn.Linear(nn_nodes, nn_nodes),
            nn.ReLU(),
            nn.Linear(nn_nodes, n_actions)
        )

    def forward(self, x):
        return self.network(x)


class ReplayBuffer:
    """Fixed-size experience buffer (deque-based)."""
    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def store(self, transition):
        self.memory.append(transition)

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


def optimize(q_network, optimizer, loss_fn, batch, identity, cfg: Config):
    """
    One gradient step on a sampled minibatch.

    Non-deterministic Bellman target (slide 48), approximated with a NN instead of a table:
        Y_t = R_t + gamma * max_a' Q(S_{t+1}, a'; theta)
    Since the project requires a single Q-network (no target network / no D-DQN),
    the max over next actions is computed with the SAME network, under torch.no_grad()
    so that gradients only flow through the "predicted" side of the loss, not the target.
    """
    states, actions, new_states, rewards, dones = zip(*batch)

    states_t = identity[list(states)]
    new_states_t = identity[list(new_states)]
    actions_t = torch.tensor(actions, dtype=torch.long, device=device)
    rewards_t = torch.tensor(rewards, dtype=torch.float32, device=device)
    dones_t = torch.tensor(dones, dtype=torch.bool, device=device)

    with torch.no_grad():
        next_q = q_network(new_states_t).max(dim=1).values
        targets = rewards_t + cfg.discount_factor * next_q * (~dones_t).float()

    predicted_q = q_network(states_t).gather(1, actions_t.unsqueeze(1)).squeeze(1)

    loss = loss_fn(predicted_q, targets)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return loss.item()


def train(cfg: Config):
    env = gym.make("CliffWalking-v1", max_episode_steps=cfg.max_episode_steps, is_slippery=cfg.is_slippery)
    n_states = env.observation_space.n
    n_actions = env.action_space.n

    # one-hot lookup table computed once (instead of rebuilding torch.eye every step)
    identity = torch.eye(n_states, device=device)

    q_network = DQN(n_states, cfg.nn_nodes, n_actions).to(device)
    optimizer = torch.optim.Adam(q_network.parameters(), lr=cfg.learning_rate)
    loss_fn = nn.MSELoss()
    buffer = ReplayBuffer(cfg.buffer_size)

    epsilon = cfg.starting_e

    metrics = {
        'episode_rewards': [],
        'episode_lengths': [],
        'successful_episode': [],
        'exploration_rate': [],
        'loss_history': []
    }

    for i in range(cfg.episodes):
        state = env.reset()[0]
        terminated = False
        truncated = False
        episode_reward = 0
        episode_length = 0
        episode_losses = []

        while not terminated and not truncated:
            # epsilon-greedy action selection
            if random.random() < epsilon:
                action = env.action_space.sample()
            else:
                with torch.no_grad():
                    action = q_network(identity[state]).argmax().item()

            new_state, reward, terminated, truncated, _ = env.step(action)
            buffer.store((state, action, new_state, reward, terminated))

            # start learning only once the buffer has at least one full batch
            if len(buffer) > cfg.batch_size:
                batch = buffer.sample(cfg.batch_size)
                loss = optimize(q_network, optimizer, loss_fn, batch, identity, cfg)
                episode_losses.append(loss)

            episode_reward += reward
            episode_length += 1
            state = new_state

        # linear epsilon decay (same formula/style as the tabular agent)
        epsilon = max(epsilon - (cfg.epsilon_decay / cfg.episodes), 0.001)

        metrics['episode_rewards'].append(episode_reward)
        metrics['episode_lengths'].append(episode_length)
        # state 47 is the goal state
        metrics['successful_episode'].append(1 if terminated and state == 47 else 0)
        metrics['exploration_rate'].append(epsilon)
        metrics['loss_history'].append(float(np.mean(episode_losses)) if episode_losses else 0.0)

        if i % 250 == 0:
            print(f"Episode {i}: epsilon={epsilon:.3f}, episode_reward={episode_reward}")

    env.close()
    return q_network, metrics


def evaluate(q_network, cfg: Config):
    """Greedy-policy evaluation (no exploration), separate from training."""
    env = gym.make("CliffWalking-v1", max_episode_steps=cfg.max_episode_steps, is_slippery=cfg.is_slippery)
    n_states = env.observation_space.n
    identity = torch.eye(n_states, device=device)
    rewards = []
    successes = []

    for _ in range(cfg.eval_episodes):
        state = env.reset()[0]
        terminated = False
        truncated = False
        episode_reward = 0

        while not terminated and not truncated:
            with torch.no_grad():
                action = q_network(identity[state]).argmax().item()
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
    """Same 4-panel layout as the tabular script, so the two are directly comparable."""
    plt.style.use('seaborn-v0_8' if 'seaborn-v0_8' in plt.style.available else 'default')
    fig = plt.figure(figsize=(16, 10))
    fig.suptitle(
        f"learning_rate={cfg.learning_rate}, epsilon_decay={cfg.epsilon_decay}, "
        f"discount_factor={cfg.discount_factor}, episodes={cfg.episodes}, "
        f"batch_size={cfg.batch_size}, buffer_size={cfg.buffer_size}, nn_nodes={cfg.nn_nodes}, "
        f"is_slippery={cfg.is_slippery}",
        fontsize=14, y=1.02
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

    # separate loss plot (useful for the oral exam on DQN internals, not needed for the
    # main 4-panel comparison against the tabular agent)
    if any(metrics['loss_history']):
        plt.figure(figsize=(10, 6))
        plt.plot(metrics['loss_history'], alpha=0.7, color='purple')
        plt.xlabel('Episode'); plt.ylabel('MSE Loss (avg per episode)')
        plt.title('Training Loss over Episodes')
        plt.grid(True, alpha=0.3)
        plt.savefig(f'{path}/training_loss.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("Loss plot saved")


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
        f.write(f"batch_size={cfg.batch_size}\n")
        f.write(f"buffer_size={cfg.buffer_size}\n")
        f.write(f"nn_nodes={cfg.nn_nodes}\n")
        f.write(f"eval_episodes={cfg.eval_episodes}\n")
        f.write(f"device={device}\n")
        f.write(f"\nTime to finish: {elapsed:.4f} s\n")

        f.write("\n=== Training (last 100 episodes) ===\n")
        f.write(f"Average reward: {np.mean(metrics['episode_rewards'][-100:]):.2f}\n")
        f.write(f"Success rate: {np.mean(metrics['successful_episode'][-100:]) * 100:.1f}%\n")
        f.write(f"Average loss: {np.mean(metrics['loss_history'][-100:]):.4f}\n")

        f.write(f"\n=== Evaluation (greedy policy, {cfg.eval_episodes} episodes) ===\n")
        f.write(f"Average reward: {eval_stats['avg_reward']:.2f}\n")
        f.write(f"Success rate: {eval_stats['success_rate']:.1f}%\n")

    print(f"Results saved to {path}/results.txt")


def append_summary(cfg: Config, metrics, eval_stats, elapsed, path, summary_path="DQN/summary.csv"):
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
        'batch_size': cfg.batch_size,
        'buffer_size': cfg.buffer_size,
        'nn_nodes': cfg.nn_nodes,
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
    cfg =Config(
    learning_rate=0.001, epsilon_decay=1.5, starting_e=1.0, discount_factor=0.99,
    episodes=2500, max_episode_steps=300, is_slippery=True, eval_episodes=100,
    batch_size=256, buffer_size=8192, nn_nodes=32, exp_name="dqn_stochastic_baseline"

    )

    now = datetime.now().strftime("%Y-%m-%d__%H-%M-%S")
    path = f"DQN/{cfg.exp_name}__{now}"
    os.makedirs(path, exist_ok=True)

    start = time.perf_counter()
    q_network, metrics = train(cfg)
    elapsed = time.perf_counter() - start

    eval_stats = evaluate(q_network, cfg)

    plot_metrics(metrics, cfg, path)
    save_results(cfg, metrics, eval_stats, elapsed, path)
    append_summary(cfg, metrics, eval_stats, elapsed, path)

    torch.save(q_network.state_dict(), f"{path}/dqn_model.pt")

    print(f"\nDone. Results saved in {path}")
    print(f"Eval -> avg reward: {eval_stats['avg_reward']:.2f}, success rate: {eval_stats['success_rate']:.1f}%")