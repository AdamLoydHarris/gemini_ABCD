"""
Analysis of gemini_agent experiment logs.

Usage:
    python analysis.py [path/to/experiment_log.jsonl]

Produces:
    - Per-session reward summary
    - Goal visit efficiency (actual steps vs BFS optimal)
    - D→A transition analysis (transitive inference test)
    - Trajectory plots saved to logs/plots/
"""

import json
import sys
from collections import deque
from pathlib import Path

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

# ---------------------------------------------------------------------------
# Grid / BFS utilities
# ---------------------------------------------------------------------------

NODE_TO_RC = {
    1: (0, 0), 2: (0, 1), 3: (0, 2),
    4: (1, 0), 5: (1, 1), 6: (1, 2),
    7: (2, 0), 8: (2, 1), 9: (2, 2),
}
RC_TO_NODE = {v: k for k, v in NODE_TO_RC.items()}
DIRECTIONS = [(-1, 0), (1, 0), (0, -1), (0, 1)]


def bfs_distance(start: int, end: int) -> int:
    if start == end:
        return 0
    visited = {start}
    queue = deque([(start, 0)])
    while queue:
        node, dist = queue.popleft()
        r, c = NODE_TO_RC[node]
        for dr, dc in DIRECTIONS:
            nr, nc = r + dr, c + dc
            if 0 <= nr <= 2 and 0 <= nc <= 2:
                neighbor = RC_TO_NODE[(nr, nc)]
                if neighbor == end:
                    return dist + 1
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append((neighbor, dist + 1))
    return -1  # unreachable (shouldn't happen on this grid)


# ---------------------------------------------------------------------------
# Log loading
# ---------------------------------------------------------------------------

def load_log(log_path: Path) -> tuple[list[dict], list[dict]]:
    steps, reflections = [], []
    with log_path.open() as f:
        for line in f:
            record = json.loads(line)
            if record['type'] == 'step':
                steps.append(record)
            elif record['type'] == 'reflection':
                reflections.append(record)
    return steps, reflections


# ---------------------------------------------------------------------------
# Analysis functions
# ---------------------------------------------------------------------------

def session_reward_summary(steps: list[dict]) -> dict[int, dict]:
    sessions = {}
    for rec in steps:
        s = rec['session']
        if s not in sessions:
            sessions[s] = {'total': 0, 'rewards_at': [], 'cumulative': []}
        sessions[s]['total'] += rec['reward']
        if rec['reward'] == 1:
            sessions[s]['rewards_at'].append(rec['step'])
        sessions[s]['cumulative'].append(sessions[s]['total'])
    return sessions


def goal_visit_efficiency(steps: list[dict]) -> list[dict]:
    """
    For each goal visit, compute how many steps were taken from the previous
    reward (or session start) to reach this goal, vs BFS optimal from the
    position where the previous reward was collected.
    """
    visits = []
    # Group by session
    by_session = {}
    for rec in steps:
        by_session.setdefault(rec['session'], []).append(rec)

    for session, records in sorted(by_session.items()):
        prev_reward_node = records[0]['agent_node']  # start node approximation
        prev_reward_step = 0

        for rec in records:
            if rec['reward'] == 1:
                steps_taken = rec['step'] - prev_reward_step
                optimal = bfs_distance(prev_reward_node, rec['agent_node'])
                visits.append({
                    'session': session,
                    'step': rec['step'],
                    'goal': rec['current_goal'],
                    'from_node': prev_reward_node,
                    'to_node': rec['agent_node'],
                    'steps_taken': steps_taken,
                    'optimal': optimal,
                    'efficiency': optimal / steps_taken if steps_taken > 0 else 1.0,
                })
                prev_reward_node = rec['agent_node']
                prev_reward_step = rec['step']

    return visits


def da_transition_analysis(visits: list[dict]) -> list[dict]:
    """
    Extract visits where goal=='A' and the previous visit was goal=='D'.
    These are the D→A transitions — the transitive inference test.
    """
    da = []
    by_session = {}
    for v in visits:
        by_session.setdefault(v['session'], []).append(v)

    for session, session_visits in sorted(by_session.items()):
        for i, v in enumerate(session_visits):
            if v['goal'] == 'A' and i > 0 and session_visits[i - 1]['goal'] == 'D':
                v = dict(v, da_trial=i)  # which D→A trial within session
                da.append(v)

    return da


# ---------------------------------------------------------------------------
# Printing
# ---------------------------------------------------------------------------

def print_summary(sessions: dict, visits: list[dict], da: list[dict],
                  reflections: list[dict]):
    print("\n" + "=" * 60)
    print("REWARD SUMMARY (per session)")
    print("=" * 60)
    print(f"{'Session':>8}  {'Total':>6}  {'Rewards':>7}  {'First reward at step':>21}")
    for s, data in sorted(sessions.items()):
        first = data['rewards_at'][0] if data['rewards_at'] else '-'
        print(f"{s:>8}  {data['total']:>6}  {len(data['rewards_at']):>7}  {str(first):>21}")

    print("\n" + "=" * 60)
    print("GOAL VISIT EFFICIENCY  (optimal / actual steps)")
    print("=" * 60)
    goal_stats = {}
    for v in visits:
        g = v['goal']
        goal_stats.setdefault(g, []).append(v['efficiency'])
    for g in ['A', 'B', 'C', 'D']:
        if g in goal_stats:
            effs = goal_stats[g]
            print(f"  Goal {g}: mean efficiency {np.mean(effs):.2f}  "
                  f"(n={len(effs)}, optimal/actual, 1.0=perfect)")

    print("\n" + "=" * 60)
    print("D→A TRANSITIONS  (transitive inference test)")
    print("=" * 60)
    if not da:
        print("  No D→A transitions found in log.")
    else:
        print(f"  {'Session':>8}  {'Step':>5}  {'From':>5}  {'To':>4}  "
              f"{'Actual':>7}  {'Optimal':>8}  {'Efficiency':>10}")
        for v in da:
            print(f"  {v['session']:>8}  {v['step']:>5}  "
                  f"{v['from_node']:>5}  {v['to_node']:>4}  "
                  f"{v['steps_taken']:>7}  {v['optimal']:>8}  "
                  f"{v['efficiency']:>10.2f}")
        effs = [v['efficiency'] for v in da]
        print(f"\n  Mean D→A efficiency: {np.mean(effs):.2f}  "
              f"(all goals mean: {np.mean([v['efficiency'] for v in visits]):.2f})")

    print("\n" + "=" * 60)
    print("POST-SESSION REFLECTIONS")
    print("=" * 60)
    for r in reflections:
        print(f"\n--- Session {r['session']} (total={r['session_total']}) ---")
        print(r['reflection'])


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

GOAL_COLORS = {'A': 'green', 'B': 'blue', 'C': 'orange', 'D': 'red'}


def plot_cumulative_rewards(sessions: dict, out_dir: Path):
    fig, ax = plt.subplots(figsize=(9, 4))
    for s, data in sorted(sessions.items()):
        ax.plot(range(1, len(data['cumulative']) + 1), data['cumulative'],
                label=f"Session {s}")
    ax.set_xlabel("Step")
    ax.set_ylabel("Cumulative reward")
    ax.set_title("Cumulative reward per session")
    ax.legend()
    fig.tight_layout()
    path = out_dir / "cumulative_rewards.png"
    fig.savefig(path, dpi=120)
    plt.close(fig)
    print(f"  Saved: {path}")


def plot_efficiency_by_session(visits: list[dict], out_dir: Path):
    by_session = {}
    for v in visits:
        by_session.setdefault(v['session'], []).append(v)

    fig, ax = plt.subplots(figsize=(9, 4))
    sessions_sorted = sorted(by_session.keys())
    means = [np.mean([v['efficiency'] for v in by_session[s]]) for s in sessions_sorted]
    ax.bar(sessions_sorted, means, color='steelblue')
    ax.axhline(1.0, color='red', linestyle='--', label='Perfect (optimal)')
    ax.set_xlabel("Session")
    ax.set_ylabel("Mean efficiency (optimal / actual)")
    ax.set_title("Navigation efficiency per session")
    ax.set_xticks(sessions_sorted)
    ax.legend()
    fig.tight_layout()
    path = out_dir / "efficiency_by_session.png"
    fig.savefig(path, dpi=120)
    plt.close(fig)
    print(f"  Saved: {path}")


def plot_da_efficiency(da: list[dict], visits: list[dict], out_dir: Path):
    if not da:
        return
    sessions = sorted({v['session'] for v in visits})
    all_mean = [np.mean([v['efficiency'] for v in visits if v['session'] == s])
                for s in sessions]
    da_sessions = [v['session'] for v in da]
    da_effs = [v['efficiency'] for v in da]

    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(sessions, all_mean, 'o-', color='steelblue', label='All goals (mean)')
    ax.scatter(da_sessions, da_effs, color='red', zorder=5, s=80,
               label='D→A transitions')
    ax.axhline(1.0, color='gray', linestyle='--', alpha=0.5)
    ax.set_xlabel("Session")
    ax.set_ylabel("Efficiency (optimal / actual)")
    ax.set_title("D→A efficiency vs all-goal mean (transitive inference test)")
    ax.legend()
    fig.tight_layout()
    path = out_dir / "da_efficiency.png"
    fig.savefig(path, dpi=120)
    plt.close(fig)
    print(f"  Saved: {path}")


def plot_trajectories(steps: list[dict], out_dir: Path):
    by_session = {}
    for rec in steps:
        by_session.setdefault(rec['session'], []).append(rec)

    for session, records in sorted(by_session.items()):
        task = records[0]['task']
        goal_nodes = {label: node for label, node in task.items()}

        fig, ax = plt.subplots(figsize=(6, 6))
        ax.set_xlim(-0.5, 2.5)
        ax.set_ylim(-0.5, 2.5)
        ax.set_xticks([0, 1, 2])
        ax.set_yticks([0, 1, 2])
        ax.set_yticklabels([0, 1, 2][::-1])  # row 0 at top
        ax.grid(True, alpha=0.3)
        ax.set_title(f"Session {session} trajectory  (task: {task})")
        ax.set_xlabel("col")
        ax.set_ylabel("row")

        # Draw goal locations
        for label, node in goal_nodes.items():
            r, c = NODE_TO_RC[node]
            ax.add_patch(mpatches.FancyBboxPatch(
                (c - 0.45, (2 - r) - 0.45), 0.9, 0.9,
                boxstyle="round,pad=0.05",
                facecolor=GOAL_COLORS[label], alpha=0.25, zorder=1
            ))
            ax.text(c, 2 - r, label, ha='center', va='center',
                    fontsize=14, fontweight='bold',
                    color=GOAL_COLORS[label], zorder=2)

        # Draw trajectory
        xs = [NODE_TO_RC[rec['agent_node']][1] for rec in records]
        ys = [2 - NODE_TO_RC[rec['agent_node']][0] for rec in records]
        ax.plot(xs, ys, '-', color='black', alpha=0.3, linewidth=0.8, zorder=3)

        # Mark reward steps
        for rec in records:
            if rec['reward'] == 1:
                r, c = NODE_TO_RC[rec['agent_node']]
                ax.plot(c, 2 - r, '*', markersize=14,
                        color=GOAL_COLORS[rec['current_goal']], zorder=5)

        # Mark start
        r0, c0 = NODE_TO_RC[records[0]['agent_node']]
        ax.plot(c0, 2 - r0, 's', markersize=10, color='black',
                label='Start', zorder=6)

        legend_patches = [
            mpatches.Patch(color=GOAL_COLORS[g], label=f"Goal {g} (node {goal_nodes[g]})")
            for g in ['A', 'B', 'C', 'D']
        ]
        legend_patches.append(mpatches.Patch(color='black', label='Start'))
        ax.legend(handles=legend_patches, loc='upper right', fontsize=8)

        fig.tight_layout()
        path = out_dir / f"trajectory_session_{session}.png"
        fig.savefig(path, dpi=120)
        plt.close(fig)
        print(f"  Saved: {path}")


def animate_sessions(steps: list[dict], out_dir: Path, fps: int = 4):
    by_session = {}
    for rec in steps:
        by_session.setdefault(rec['session'], []).append(rec)

    for session, records in sorted(by_session.items()):
        task = records[0]['task']
        goal_nodes = {label: node for label, node in task.items()}

        fig, (ax_grid, ax_reward) = plt.subplots(
            1, 2, figsize=(10, 5),
            gridspec_kw={'width_ratios': [1, 1.4]}
        )
        fig.suptitle(f"Session {session}", fontsize=13)

        # ---- grid axes setup ----
        ax_grid.set_xlim(-0.5, 2.5)
        ax_grid.set_ylim(-0.5, 2.5)
        ax_grid.set_xticks([0, 1, 2])
        ax_grid.set_yticks([0, 1, 2])
        ax_grid.set_yticklabels([0, 1, 2][::-1])
        ax_grid.grid(True, alpha=0.3)
        ax_grid.set_aspect('equal')

        for label, node in goal_nodes.items():
            r, c = NODE_TO_RC[node]
            ax_grid.add_patch(mpatches.FancyBboxPatch(
                (c - 0.45, (2 - r) - 0.45), 0.9, 0.9,
                boxstyle="round,pad=0.05",
                facecolor=GOAL_COLORS[label], alpha=0.2, zorder=1
            ))
            ax_grid.text(c, 2 - r, label, ha='center', va='center',
                         fontsize=16, fontweight='bold',
                         color=GOAL_COLORS[label], zorder=2)

        trail_line, = ax_grid.plot([], [], '-', color='black', alpha=0.25,
                                   linewidth=1.0, zorder=3)
        agent_dot, = ax_grid.plot([], [], 'o', markersize=18, color='black',
                                  zorder=5)
        reward_flash, = ax_grid.plot([], [], '*', markersize=22, zorder=6)
        step_text = ax_grid.text(0.02, 0.98, '', transform=ax_grid.transAxes,
                                 va='top', fontsize=10)

        # ---- reward axes setup ----
        ax_reward.set_xlim(0, len(records) + 1)
        ax_reward.set_ylim(-0.5, len(records) + 1)
        ax_reward.set_xlabel("Step")
        ax_reward.set_ylabel("Cumulative reward")
        ax_reward.set_title("Reward over time")
        cumulative = np.cumsum([r['reward'] for r in records])
        ax_reward.plot(range(1, len(records) + 1), cumulative,
                       color='lightgrey', linewidth=1, zorder=1)
        reward_line, = ax_reward.plot([], [], color='steelblue',
                                      linewidth=2, zorder=2)
        reward_dot, = ax_reward.plot([], [], 'o', color='steelblue',
                                     markersize=6, zorder=3)

        def init():
            trail_line.set_data([], [])
            agent_dot.set_data([], [])
            reward_flash.set_data([], [])
            reward_line.set_data([], [])
            reward_dot.set_data([], [])
            step_text.set_text('')
            return trail_line, agent_dot, reward_flash, reward_line, reward_dot, step_text

        def update(frame):
            rec = records[frame]
            r, c = NODE_TO_RC[rec['agent_node']]
            ax_c, ax_r = c, 2 - r

            xs = [NODE_TO_RC[records[i]['agent_node']][1] for i in range(frame + 1)]
            ys = [2 - NODE_TO_RC[records[i]['agent_node']][0] for i in range(frame + 1)]
            trail_line.set_data(xs, ys)
            agent_dot.set_data([ax_c], [ax_r])

            if rec['reward'] == 1:
                reward_flash.set_data([ax_c], [ax_r])
                reward_flash.set_color(GOAL_COLORS[rec['current_goal']])
            else:
                reward_flash.set_data([], [])

            steps_so_far = range(1, frame + 2)
            reward_line.set_data(list(steps_so_far), cumulative[:frame + 1])
            reward_dot.set_data([frame + 1], [cumulative[frame]])

            step_text.set_text(
                f"Step {rec['step']}/{len(records)}  "
                f"reward: +{rec['reward']}  total: {rec['session_total']}"
            )
            return trail_line, agent_dot, reward_flash, reward_line, reward_dot, step_text

        ani = animation.FuncAnimation(
            fig, update, frames=len(records),
            init_func=init, blit=True, interval=1000 // fps
        )

        fig.tight_layout()
        path = out_dir / f"session_{session}.gif"
        ani.save(path, writer='pillow', fps=fps)
        plt.close(fig)
        print(f"  Saved: {path}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    if len(sys.argv) > 1:
        log_path = Path(sys.argv[1])
        if log_path.is_dir():
            log_path = log_path / "experiment_log.jsonl"
    else:
        # Default: most recent timestamped run folder
        logs_dir = Path(__file__).parent.parent / "logs"
        run_dirs = sorted(p for p in logs_dir.iterdir() if p.is_dir())
        log_path = run_dirs[-1] / "experiment_log.jsonl" if run_dirs else logs_dir / "experiment_log.jsonl"

    if not log_path.exists():
        print(f"Log not found: {log_path}")
        print("Run gemini_agent.py first to generate a log.")
        sys.exit(1)

    out_dir = log_path.parent / "plots"
    out_dir.mkdir(exist_ok=True)

    print(f"Loading {log_path} ...")
    steps, reflections = load_log(log_path)
    print(f"  {len(steps)} step records, {len(reflections)} reflections")

    sessions = session_reward_summary(steps)
    visits = goal_visit_efficiency(steps)
    da = da_transition_analysis(visits)

    print_summary(sessions, visits, da, reflections)

    print("\nGenerating plots ...")
    plot_cumulative_rewards(sessions, out_dir)
    plot_efficiency_by_session(visits, out_dir)
    plot_da_efficiency(da, visits, out_dir)
    plot_trajectories(steps, out_dir)
    animate_sessions(steps, out_dir)


if __name__ == "__main__":
    main()
