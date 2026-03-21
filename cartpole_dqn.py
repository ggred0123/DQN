import os
os.environ["SDL_AUDIODRIVER"] = "dummy"  # suppress ALSA errors on headless server

import gymnasium as gym
from gymnasium.wrappers import RecordVideo
import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np
from collections import deque
import matplotlib.pyplot as plt
import shutil
import glob


SEED = 0
DEVICE = torch.device("cpu")


def set_seed(seed):
    """Set seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"


class DQN(nn.Module):
    """
    Q-network: takes state (4-dim) as input and outputs Q-values for each action (2).
    Architecture: 3-layer MLP with ReLU activations.
    """
    def __init__(self, state_dim, action_dim):
        super(DQN, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim)
        )

    def forward(self, x):
        return self.net(x)


class ReplayMemory:
    """Replay memory buffer with fixed capacity N."""
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size, device):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            torch.stack(states).to(device),
            torch.tensor(actions, dtype=torch.long, device=device),
            torch.tensor(rewards, dtype=torch.float32, device=device),
            torch.stack(next_states).to(device),
            torch.tensor(dones, dtype=torch.float32, device=device)
        )

    def __len__(self):
        return len(self.buffer)


def select_action(state, policy_net, epsilon, action_dim, device):
    """Epsilon-greedy action selection."""
    if random.random() < epsilon:
        return random.randrange(action_dim)
    else:
        with torch.no_grad():
            q_values = policy_net(state.unsqueeze(0).to(device))
            return q_values.argmax(dim=1).item()


def running_DQL(env, M, T, gamma=0.99, lr=1e-4, batch_size=128,
                memory_capacity=10000, eps_start=1.0, eps_end=0.01,
                eps_decay=0.995, target_update_freq=10, warmup_steps=1000,
                record_video=True, save_graph=True, verbose=True):
    """
    Deep Q-Learning with Experience Replay.

    Core components:
    - Epsilon-greedy policy with gradual decay
    - Gradient descent with replay memory (experience replay)
    - Replay memory warm-up
    - Fixed Q-targets (target network updated at regular intervals)

    Args:
        record_video: If True, record all episodes and save the best one as video.
        save_graph:   If True, save the training performance graph.
        verbose:      If True, print training progress.

    Returns:
        episode_durations: list of step counts per episode.
    """
    device = DEVICE
    if verbose:
        print(f"Using device: {device}")

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    # Initialize action-value function Q (policy_net) with random weights
    policy_net = DQN(state_dim, action_dim).to(device)
    # Initialize target network (fixed Q-targets) as a copy of policy_net
    target_net = DQN(state_dim, action_dim).to(device)
    target_net.load_state_dict(policy_net.state_dict())

    optimizer = optim.Adam(policy_net.parameters(), lr=lr)
    # Initialize replay memory D with capacity N
    memory = ReplayMemory(memory_capacity)

    episode_durations = []
    max_duration = 0
    best_episode = 0
    epsilon = eps_start
    total_steps = 0

    # --- Optionally wrap env with RecordVideo ---
    if record_video:
        all_videos_folder = "all_videos"
        if os.path.exists(all_videos_folder):
            shutil.rmtree(all_videos_folder)
        env.close()
        env = gym.make('CartPole-v1', max_episode_steps=2000, render_mode="rgb_array")
        env.reset(seed=SEED)  # keep seed
        env = RecordVideo(env, video_folder=all_videos_folder,
                          episode_trigger=lambda x: True, disable_logger=True)

    for episode in range(1, M + 1):
        # Reset environment and get initial state: s1 = {x1}, phi1 = phi(s1)
        state, _ = env.reset()
        state = torch.tensor(state, dtype=torch.float32)

        for t in range(1, T + 1):
            # With probability epsilon select a random action
            # Otherwise select a = argmax_a Q(phi(s), a; theta)
            action = select_action(state, policy_net, epsilon, action_dim, device)

            # Execute action in emulator and observe reward r and image x_{t+1}
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            next_state = torch.tensor(next_state, dtype=torch.float32)

            # Store transition (phi_t, a_t, r_t, phi_{t+1}) in D
            # (stored on CPU, moved to device when sampled)
            memory.push(state, action, reward, next_state, done)

            state = next_state
            total_steps += 1

            # Replay memory warm-up: only update after enough transitions
            if len(memory) >= warmup_steps:
                # Sample random minibatch of transitions from D
                s_batch, a_batch, r_batch, ns_batch, d_batch = memory.sample(batch_size, device)

                # Compute Q(phi_j, a_j; theta) for the taken actions
                q_values = policy_net(s_batch).gather(1, a_batch.unsqueeze(1)).squeeze(1)

                # Set y_j (using fixed target network):
                #   y_j = r_j                                        for terminal phi_{j+1}
                #   y_j = r_j + gamma * max_{a'} Q(phi_{j+1}, a'; theta^-)  for non-terminal
                with torch.no_grad():
                    max_next_q = target_net(ns_batch).max(dim=1).values
                    targets = r_batch + gamma * max_next_q * (1.0 - d_batch)

                # Perform gradient descent step on (y_j - Q(phi_j, a_j; theta))^2
                loss = nn.functional.smooth_l1_loss(q_values, targets)
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(policy_net.parameters(), max_norm=100)
                optimizer.step()

            if done:
                break

        # Record episode duration
        duration = t
        episode_durations.append(duration)

        # Track the best episode
        if duration > max_duration:
            max_duration = duration
            best_episode = episode
            if verbose:
                print(f"  >> New best! Episode {episode}: {duration} steps")

        # Gradually decay epsilon
        epsilon = max(eps_end, epsilon * eps_decay)

        # Fixed Q-targets: update target network at regular intervals
        if episode % target_update_freq == 0:
            target_net.load_state_dict(policy_net.state_dict())

        if verbose and episode % 20 == 0:
            avg = np.mean(episode_durations[-20:])
            print(f"Episode {episode}/{M} | Duration: {duration} | "
                  f"Avg(20): {avg:.1f} | Max: {max_duration} (ep {best_episode}) | Eps: {epsilon:.4f}")

    env.close()

    # --- Extract the best episode video ---
    if record_video:
        best_video_index = best_episode - 1
        best_video_file = os.path.join(all_videos_folder, f"rl-video-episode-{best_video_index}.mp4")

        final_folder = "best_video"
        if os.path.exists(final_folder):
            shutil.rmtree(final_folder)
        os.makedirs(final_folder)

        if os.path.exists(best_video_file):
            shutil.copy2(best_video_file, os.path.join(final_folder, "best_episode.mp4"))
            print(f"\nBest episode {best_episode} ({max_duration} steps) video saved to '{final_folder}/best_episode.mp4'")
        else:
            print(f"\nWarning: Could not find video file for episode {best_episode}")
            available = sorted(glob.glob(os.path.join(all_videos_folder, "*.mp4")))
            print(f"Available videos: {len(available)} files")
            if available:
                shutil.copy2(available[-1], os.path.join(final_folder, "best_episode.mp4"))
                print(f"Copied fallback: {available[-1]}")

        shutil.rmtree(all_videos_folder)

    # --- Save performance graph ---
    if save_graph:
        plt.figure(figsize=(10, 5))
        plt.plot(episode_durations)
        plt.title("DQN CartPole Training")
        plt.xlabel("Episode")
        plt.ylabel("Step count")
        plt.savefig("training_graph.png", dpi=150, bbox_inches='tight')
        plt.close()
        print("Performance graph saved as training_graph.png")

    return episode_durations


def main():
    # Create CartPole environment
    # Action space: env.action_space.n (2: push cart to the left, push cart to the right)
    # State space: env.observation_space.shape (4: cart pos, cart vel, pole angle, pole ang vel)
    set_seed(SEED)
    env = gym.make('CartPole-v1', max_episode_steps=2000)
    env.reset(seed=SEED)

    M = 400   # number of episodes
    T = 2000   # max time steps per episode

    running_DQL(env, M, T,
                gamma=0.99,
                lr=1e-4,
                batch_size=128,
                memory_capacity=10000,
                eps_start=1.0,
                eps_end=0.01,
                eps_decay=0.995,
                target_update_freq=10,
                warmup_steps=1000,
                record_video=True,
                save_graph=True)


if __name__ == "__main__":
    main()
