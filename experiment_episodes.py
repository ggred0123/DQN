"""
Experiment 2: Compare training trends as episode count (M) increases from 400 to 800
- Same seed (42), only episode count varies
- Observe how much performance improves with longer training
- Track best step, 400+ achievement rate, and average performance per M
"""
import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
from cartpole_dqn import set_seed, running_DQL


SEED = 42
M_VALUES = [400, 500, 600, 700, 800]
T = 500


if __name__ == "__main__":
    results = {}
    print("=" * 70)
    print(f"Experiment 2: Episode count variation (seed = {SEED})")
    print("=" * 70)

    fig, axes = plt.subplots(len(M_VALUES), 1, figsize=(14, 4 * len(M_VALUES)), sharex=False)

    for idx, M in enumerate(M_VALUES):
        print(f"\n--- M = {M} training start ---")
        set_seed(SEED)
        env = gym.make('CartPole-v1', max_episode_steps=500)
        env.reset(seed=SEED)

        durations = running_DQL(env, M, T,
                                record_video=False,
                                save_graph=False,
                                verbose=False)

        best = max(durations)
        best_ep = durations.index(best) + 1
        over_400 = sum(1 for d in durations if d >= 400)
        pct_400 = over_400 / len(durations) * 100
        avg_all = np.mean(durations)
        avg_last50 = np.mean(durations[-50:])

        # Moving average (window=20)
        window = 20
        moving_avg = [np.mean(durations[max(0, i - window):i + 1]) for i in range(len(durations))]

        results[M] = {
            'best': best,
            'best_episode': best_ep,
            'avg_all': avg_all,
            'avg_last50': avg_last50,
            'over_400_count': over_400,
            'over_400_pct': pct_400,
            'durations': durations,
        }
        print(f"  M={M}: Best={best} (ep {best_ep}), "
              f"Avg(all)={avg_all:.1f}, Avg(last50)={avg_last50:.1f}, "
              f"400+: {over_400}/{M} ({pct_400:.1f}%)")

        # Subplot
        ax = axes[idx]
        ax.plot(durations, alpha=0.4, color='steelblue', linewidth=0.8)
        ax.plot(moving_avg, color='darkblue', linewidth=1.5, label=f'Moving avg (w={window})')
        ax.axhline(y=400, color='red', linestyle='--', alpha=0.5, label='Target (400)')
        ax.set_title(f"M = {M}  |  Best = {best} (ep {best_ep})  |  400+: {pct_400:.1f}%", fontsize=11)
        ax.set_ylabel("Step count")
        ax.legend(loc='upper left', fontsize=8)
        ax.grid(True, alpha=0.3)

    axes[-1].set_xlabel("Episode")
    plt.tight_layout()
    plt.savefig("experiment_episodes_comparison.png", dpi=150)
    plt.close()

    # Overlay graph
    plt.figure(figsize=(14, 6))
    for M in M_VALUES:
        durations = results[M]['durations']
        window = 20
        moving_avg = [np.mean(durations[max(0, i - window):i + 1]) for i in range(len(durations))]
        plt.plot(moving_avg, label=f"M={M} (best={results[M]['best']})", linewidth=1.5)

    plt.axhline(y=400, color='red', linestyle='--', alpha=0.5, label='Target (400)')
    plt.title("DQN CartPole - Episode Count Comparison (Moving Avg)", fontsize=14)
    plt.xlabel("Episode")
    plt.ylabel("Step count (moving avg, w=20)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("experiment_episodes_overlay.png", dpi=150)
    plt.close()

    # Summary table
    print("\n" + "=" * 70)
    print(f"{'M':>5} | {'Best':>5} | {'Best Ep':>8} | {'Avg(all)':>9} | {'Avg(last50)':>12} | {'400+ (%)':>10}")
    print("-" * 70)
    for M in M_VALUES:
        r = results[M]
        print(f"{M:>5} | {r['best']:>5} | {r['best_episode']:>8} | {r['avg_all']:>9.1f} | "
              f"{r['avg_last50']:>12.1f} | {r['over_400_pct']:>9.1f}%")
    print("=" * 70)
    print("\nGraphs saved:")
    print("  - experiment_episodes_comparison.png (individual subplots)")
    print("  - experiment_episodes_overlay.png    (moving avg overlay)")
