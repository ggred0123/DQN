"""
Experiment 1: Compare training results across different seeds
- Same hyperparameters, only seed varies
- Track best step, episode reached, and average performance per seed
- Overlay all results on a single graph
"""
import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
from cartpole_dqn import set_seed, running_DQL


SEEDS = [0, 42, 100, 200, 300, 500]
M = 400
T = 2000


if __name__ == "__main__":
    results = {}
    print("=" * 60)
    print(f"Experiment 1: Seed variation (num episodes = {M})")
    print("=" * 60)

    plt.figure(figsize=(14, 6))

    for seed in SEEDS:
        print(f"\n--- Seed {seed} training start ---")
        set_seed(seed)
        env = gym.make('CartPole-v1', max_episode_steps=2000)
        env.reset(seed=seed)

        durations = running_DQL(env, M, T,
                                record_video=False,
                                save_graph=False,
                                verbose=False)

        best = max(durations)
        best_ep = durations.index(best) + 1
        avg_last50 = np.mean(durations[-50:])

        results[seed] = {
            'best': best,
            'best_episode': best_ep,
            'avg_last50': avg_last50,
        }
        print(f"  Seed {seed}: Best={best} (ep {best_ep}), Avg(last50)={avg_last50:.1f}")
        plt.plot(durations, alpha=0.7, label=f"Seed {seed} (best={best})")

    plt.title("DQN CartPole - Seed Comparison", fontsize=14)
    plt.xlabel("Episode")
    plt.ylabel("Step count")
    plt.legend(loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("experiment_seed_comparison.png", dpi=150)
    plt.close()

    # Summary table
    print("\n" + "=" * 60)
    print(f"{'Seed':>6} | {'Best':>5} | {'Best Ep':>8} | {'Avg(last50)':>12}")
    print("-" * 60)
    for seed in SEEDS:
        r = results[seed]
        print(f"{seed:>6} | {r['best']:>5} | {r['best_episode']:>8} | {r['avg_last50']:>12.1f}")
    print("=" * 60)
    print("\nGraph saved: experiment_seed_comparison.png")
