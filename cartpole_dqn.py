import gymnasium as gym
from gymnasium.wrappers import RecordVideo
import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np
from collections import deque
import matplotlib.pyplot as plt


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

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            torch.stack(states),
            torch.tensor(actions, dtype=torch.long),
            torch.tensor(rewards, dtype=torch.float32),
            torch.stack(next_states),
            torch.tensor(dones, dtype=torch.float32)
        )

    def __len__(self):
        return len(self.buffer)


def select_action(state, policy_net, epsilon, action_dim):
    """Epsilon-greedy action selection."""
    if random.random() < epsilon:
        return random.randrange(action_dim)
    else:
        with torch.no_grad():
            q_values = policy_net(state.unsqueeze(0))
            return q_values.argmax(dim=1).item()


def running_DQL(env, M, T, gamma=0.99, lr=1e-4, batch_size=128,
                memory_capacity=10000, eps_start=1.0, eps_end=0.01,
                eps_decay=0.995, target_update_freq=10, warmup_steps=1000):
    """
    Deep Q-Learning with Experience Replay.

    Core components:
    - Epsilon-greedy policy with gradual decay
    - Gradient descent with replay memory (experience replay)
    - Replay memory warm-up
    - Fixed Q-targets (target network updated at regular intervals)
    """
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    # Initialize action-value function Q (policy_net) with random weights
    policy_net = DQN(state_dim, action_dim)
    # Initialize target network (fixed Q-targets) as a copy of policy_net
    target_net = DQN(state_dim, action_dim)
    target_net.load_state_dict(policy_net.state_dict())

    optimizer = optim.Adam(policy_net.parameters(), lr=lr)
    # Initialize replay memory D with capacity N
    memory = ReplayMemory(memory_capacity)

    episode_durations = []
    max_duration = 0
    best_episode = 0
    epsilon = eps_start
    total_steps = 0

    for episode in range(1, M + 1):
        # Reset environment and get initial state: s1 = {x1}, phi1 = phi(s1)
        state, _ = env.reset()
        state = torch.tensor(state, dtype=torch.float32)

        for t in range(1, T + 1):
            # With probability epsilon select a random action
            # Otherwise select a = argmax_a Q(phi(s), a; theta)
            action = select_action(state, policy_net, epsilon, action_dim)

            # Execute action in emulator and observe reward r and image x_{t+1}
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            next_state = torch.tensor(next_state, dtype=torch.float32)

            # Store transition (phi_t, a_t, r_t, phi_{t+1}) in D
            memory.push(state, action, reward, next_state, done)

            state = next_state
            total_steps += 1

            # Replay memory warm-up: only update after enough transitions
            if len(memory) >= warmup_steps:
                # Sample random minibatch of transitions from D
                s_batch, a_batch, r_batch, ns_batch, d_batch = memory.sample(batch_size)

                # Compute Q(phi_j, a_j; theta) for the taken actions
                q_values = policy_net(s_batch).gather(1, a_batch.unsqueeze(1)).squeeze(1)

                # Set y_j (using fixed target network):
                #   y_j = r_j                                        for terminal phi_{j+1}
                #   y_j = r_j + gamma * max_{a'} Q(phi_{j+1}, a'; theta^-)  for non-terminal
                with torch.no_grad():
                    max_next_q = target_net(ns_batch).max(dim=1).values
                    targets = r_batch + gamma * max_next_q * (1.0 - d_batch)

                # Perform gradient descent step on (y_j - Q(phi_j, a_j; theta))^2
                loss = nn.functional.mse_loss(q_values, targets)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            if done:
                break

        # Record episode duration
        duration = t
        episode_durations.append(duration)

        # Track the best episode (longest survival)
        if duration > max_duration:
            max_duration = duration
            best_episode = episode

        # Gradually decay epsilon
        epsilon = max(eps_end, epsilon * eps_decay)

        # Fixed Q-targets: update target network at regular intervals
        if episode % target_update_freq == 0:
            target_net.load_state_dict(policy_net.state_dict())

        if episode % 20 == 0:
            avg = np.mean(episode_durations[-20:])
            print(f"Episode {episode}/{M} | Duration: {duration} | "
                  f"Avg(20): {avg:.1f} | Max: {max_duration} | Eps: {epsilon:.4f}")

    env.close()

    # --- Record video of the best-performing episode using the trained agent ---
    print(f"\nTraining complete. Best episode: {best_episode} with {max_duration} steps.")
    print("Recording video of trained agent...")

    video_env = gym.make('CartPole-v1', max_episode_steps=500, render_mode="rgb_array")
    video_env = RecordVideo(video_env, video_folder="best_video",
                            episode_trigger=lambda x: True, disable_logger=True)

    state, _ = video_env.reset()
    state = torch.tensor(state, dtype=torch.float32)
    for t in range(1, T + 1):
        action = select_action(state, policy_net, 0.0, action_dim)  # greedy (no exploration)
        next_state, reward, terminated, truncated, _ = video_env.step(action)
        done = terminated or truncated
        state = torch.tensor(next_state, dtype=torch.float32)
        if done:
            print(f"Video episode lasted {t} steps.")
            break
    video_env.close()

    # --- Save performance graph ---
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
    env = gym.make('CartPole-v1', max_episode_steps=500)

    M = 400   # number of episodes
    T = 500   # max time steps per episode

    running_DQL(env, M, T,
                gamma=0.99,
                lr=1e-4,
                batch_size=128,
                memory_capacity=10000,
                eps_start=1.0,
                eps_end=0.01,
                eps_decay=0.995,
                target_update_freq=10,
                warmup_steps=1000)


if __name__ == "__main__":
    main()
