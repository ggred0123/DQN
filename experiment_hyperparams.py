"""
Experiment: Hyperparameter Ablation Study
- Fixed: seed=42, M=400
- Baseline: lr=1e-4, eps_decay=0.995, target_update_freq=10
- Change one hyperparameter at a time to analyze its effect

Experiment A: Learning Rate (lr)
Experiment B: Epsilon Decay Rate (eps_decay)
Experiment C: Target Network Update Frequency (target_update_freq)
"""
import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
from cartpole_dqn import set_seed, running_DQL


SEED = 42
M = 400
T = 2000

# Baseline hyperparameters
BASE = {
    'gamma': 0.99,
    'lr': 1e-4,
    'batch_size': 128,
    'memory_capacity': 10000,
    'eps_start': 1.0,
    'eps_end': 0.01,
    'eps_decay': 0.995,
    'target_update_freq': 10,
    'warmup_steps': 1000,
}

# Values to sweep
EXPERIMENTS = {
    'lr': {
        'values': [1e-3, 5e-4, 1e-4, 5e-5, 1e-5],
        'label': 'Learning Rate',
        'fmt': lambda v: f"{v:.0e}",
    },
    'eps_decay': {
        'values': [0.99, 0.993, 0.995, 0.997, 0.999],
        'label': 'Epsilon Decay Rate',
        'fmt': lambda v: f"{v}",
    },
    'target_update_freq': {
        'values': [5, 10, 20, 50, 100],
        'label': 'Target Update Frequency',
        'fmt': lambda v: f"{v}",
    },
}


def run_experiment(param_name, param_values, param_label, param_fmt):
    """Run experiment for a single hyperparameter."""
    results = {}

    for val in param_values:
        tag = param_fmt(val)
        print(f"\n  {param_label} = {tag} training start...")

        # Copy baseline and override the target parameter
        kwargs = BASE.copy()
        kwargs[param_name] = val

        set_seed(SEED)
        env = gym.make('CartPole-v1', max_episode_steps=2000)
        env.reset(seed=SEED)

        durations = running_DQL(env, M, T,
                                record_video=False,
                                save_graph=False,
                                verbose=False,
                                **kwargs)

        best = max(durations)
        best_ep = durations.index(best) + 1
        avg_all = np.mean(durations)
        avg_last50 = np.mean(durations[-50:])
        over_400 = sum(1 for d in durations if d >= 400)

        results[val] = {
            'tag': tag,
            'best': best,
            'best_episode': best_ep,
            'avg_all': avg_all,
            'avg_last50': avg_last50,
            'over_400': over_400,
            'durations': durations,
        }
        print(f"    Best={best} (ep {best_ep}), Avg(last50)={avg_last50:.1f}, 400+: {over_400}")

    return results


def plot_experiment(param_name, param_values, param_label, param_fmt, results):
    """Generate experiment plots: training curves + summary bar chart."""
    fig, axes = plt.subplots(1, 2, figsize=(18, 6))

    window = 20
    y_max = max(r['best'] for r in results.values())
    shared_ylim = (0, y_max * 1.1)

    # (1) Moving average training curves
    ax1 = axes[0]
    for val in param_values:
        r = results[val]
        durations = r['durations']
        moving_avg = [np.mean(durations[max(0, i - window):i + 1]) for i in range(len(durations))]
        ax1.plot(moving_avg, linewidth=1.5,
                 label=f"{param_fmt(val)} (best={r['best']})")

    ax1.axhline(y=400, color='red', linestyle='--', alpha=0.5, label='Target (400)')
    ax1.set_title(f"{param_label} - Training Curves (Moving Avg, w={window})", fontsize=12)
    ax1.set_xlabel("Episode")
    ax1.set_ylabel("Step count")
    ax1.set_ylim(*shared_ylim)
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3)

    # (2) Best (left y-axis, same scale as left graph) / 400+ count (right y-axis)
    ax2 = axes[1]
    ax2r = ax2.twinx()
    tags = [param_fmt(v) for v in param_values]
    bests = [results[v]['best'] for v in param_values]
    over_400s = [results[v]['over_400'] for v in param_values]

    x = np.arange(len(tags))
    w = 0.35
    bars1 = ax2.bar(x - w/2, bests, w, label='Best (steps)', color='steelblue')
    bars2 = ax2r.bar(x + w/2, over_400s, w, label='400+ count', color='coral')
    ax2.set_title(f"{param_label} - Performance Summary", fontsize=12)
    ax2.set_xlabel(param_label)
    ax2.set_ylabel("Best (steps)", color='steelblue')
    ax2r.set_ylabel("400+ count", color='coral')
    ax2.set_ylim(*shared_ylim)
    ax2.set_xticks(x)
    ax2.set_xticklabels(tags)
    lines1, labels1 = ax2.get_legend_handles_labels()
    lines2, labels2 = ax2r.get_legend_handles_labels()
    ax2.legend(lines1 + lines2, labels1 + labels2)
    ax2.grid(True, alpha=0.3, axis='y')

    # Label values on top of bars
    for bar in bars1:
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                 f'{int(bar.get_height())}', ha='center', va='bottom', fontsize=8)
    for bar in bars2:
        ax2r.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                  f'{int(bar.get_height())}', ha='center', va='bottom', fontsize=8)

    plt.tight_layout()
    filename = f"experiment_{param_name}.png"
    plt.savefig(filename, dpi=150)
    plt.close()
    print(f"  Graph saved: {filename}")


def print_table(param_name, param_values, param_label, param_fmt, results):
    """Print summary table."""
    print(f"\n{'Value':>12} | {'Best':>5} | {'Best Ep':>8} | {'Avg(all)':>9} | {'Avg(last50)':>12} | {'400+':>5}")
    print("-" * 65)
    for val in param_values:
        r = results[val]
        print(f"{r['tag']:>12} | {r['best']:>5} | {r['best_episode']:>8} | "
              f"{r['avg_all']:>9.1f} | {r['avg_last50']:>12.1f} | {r['over_400']:>5}")


if __name__ == "__main__":
    all_results = {}

    print("=" * 65)
    print("Hyperparameter Ablation Study")
    print(f"Fixed: Seed={SEED}, M={M}, T={T}")
    print(f"Baseline: lr={BASE['lr']}, eps_decay={BASE['eps_decay']}, "
          f"target_update_freq={BASE['target_update_freq']}")
    print("=" * 65)

    for param_name, config in EXPERIMENTS.items():
        values = config['values']
        label = config['label']
        fmt = config['fmt']

        print(f"\n{'='*65}")
        print(f"Experiment: {label}")
        print(f"{'='*65}")

        results = run_experiment(param_name, values, label, fmt)
        all_results[param_name] = results

        print_table(param_name, values, label, fmt, results)
        plot_experiment(param_name, values, label, fmt, results)

    print(f"\n{'='*65}")
    print("All experiments complete!")
    print("Generated graphs:")
    print("  - experiment_lr.png")
    print("  - experiment_eps_decay.png")
    print("  - experiment_target_update_freq.png")
    print(f"{'='*65}")
