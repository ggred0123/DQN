import gymnasium as gym
from gymnasium.wrappers import RecordVideo
import torch
import torch.nn as nn

"""
TODO:
Train an agent that can survive over 400 time steps (~8 seconds).

Also, save:
- A video of the episode in which the agent survived for the longest number of time steps.
- A performance graph (the number of time steps survived in each episode) that shows the learning progress of your agent.
similar to the provided examples.

You may refer to the lecture notes if needed.
"""


class DQN(nn.Module):
    """
    TODO:
    Design your own Q-network.
    """



def running_DQL(env, M, T, ...):
    """
    TODO:
    Implement the main loop of Deep Q-Learning.

    You are free to design the details, but your implementation should include:
    - Epsilon-greedy policy
    - Network update via gradient descent with replay memory
    - Replay memory warm-up
    - Fixed Q-targets
    """

    episode_durations = []
    max_duration = 0
    for episode in range(1, M):
        # Record video only for the last episode (you may modify this)
        if episode == M - 1:
            env.close()
            env = gym.make('CartPole-v1', max_episode_steps=500, render_mode="rgb_array")
            env = RecordVideo(env, video_folder="PATH", episode_trigger=lambda x: True, disable_logger=True)
        
        # Reset environment and get initial state
        state, _ = env.reset()
        state = torch.tensor(state, dtype=torch.float32)

        for t in range(1, T):
            # You can take a step for the next state and get next_state, reward, terminated_flg, and truncated_flg
            next_state, reward, terminated_flg, truncated_flg, _ = env.step(action)

    env.close()


def main():
    # Create CartPole environment
    # Action space: env.action_space.n (2: push cart to the left, push cart to the right)
    # State space: env.observation_space.shape (4: cart pos, cart vel, pole angle, pole ang vel)
    env = gym.make('CartPole-v1', max_episode_steps=500 ,render_mode='human')

    M = 
    T = 
    
    running_DQL(env, M, T, ...)


if __name__ == "__main__":
    main()