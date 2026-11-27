#!/usr/bin/env python3
import argparse
import csv
import json
import os

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

MOVE_COUNTS = 3
COUNT = 2 * np.pi / 16384


def load(prefix):
    with open(prefix + '_state.csv') as f:
        rows = list(csv.DictReader(f))
    state = {k: np.array([float(r[k]) for r in rows]) for k in ('t', 'qpos1', 'qpos2', 'qvel1', 'qvel2')}
    with open(prefix + '_cmd.csv') as f:
        cmds = [dict(t=float(r['t']), q=(float(r['q1des']), float(r['q2des'])), label=r['label'])
                for r in csv.DictReader(f)]
    with open(prefix + '_meta.json') as f:
        meta = json.load(f)
    return state, cmds, meta


def crossing(t, y, level):
    idx = np.where(y >= level)[0]
    if not idx.size:
        return np.nan
    i = idx[0]
    if i == 0 or y[i] == y[i - 1]:
        return t[i]
    return t[i - 1] + (level - y[i - 1]) / (y[i] - y[i - 1]) * (t[i] - t[i - 1])


def fit_fopdt(t, y):
    best = (np.inf, np.nan, np.nan)
    for dead in np.arange(0.0, 0.6, 0.01):
        for tau in np.arange(0.05, 1.5, 0.01):
            model = np.where(t > dead, 1.0 - np.exp(-(t - dead) / tau), 0.0)
            err = np.sum((model - y) ** 2)
            if err < best[0]:
                best = (err, dead, tau)
    return best[1], best[2]


def analyse(state, cmds):
    results = []
    for k in range(1, len(cmds) - 1):
        prev, cmd, nxt = cmds[k - 1], cmds[k], cmds[k + 1]
        joint = 0 if (cmd['q'][0] != prev['q'][0]) else 1
        q0, qf = prev['q'][joint], cmd['q'][joint]
        delta = qf - q0
        if abs(delta) < 1e-9:
            continue
        m = (state['t'] >= cmd['t'] - 0.3) & (state['t'] < nxt['t'])
        t = state['t'][m] - cmd['t']
        q = state['qpos%d' % (joint + 1)][m]
        other = state['qpos%d' % (2 - joint)][m]
        qv = state['qvel%d' % (joint + 1)][m]
        before = t < 0
        after = ~before
        if not before.any() or not after.any():
            continue
        q_start = q[before][-1]
        other_start = other[before][-1]
        span = qf - q_start
        y = (q - q_start) / span
        moved = np.abs(q - q_start) > max(MOVE_COUNTS * COUNT, 0.05 * abs(span))
        idx = np.where(after & moved)[0]
        t_move = t[idx[0]] if idx.size else np.nan
        t10 = crossing(t[after], y[after], 0.1)
        t90 = crossing(t[after], y[after], 0.9)
        band = max(0.05 * abs(span), 0.005)
        outside = np.where(after & (np.abs(q - qf) > band))[0]
        t_settle = t[outside[-1] + 1] if outside.size and outside[-1] + 1 < len(t) else (0.0 if not outside.size else np.nan)
        tail = after & (t >= t[-1] - 0.5)
        dead, tau = fit_fopdt(t[after], y[after])
        results.append(dict(
            label=cmd['label'], joint=joint + 1, delta=delta,
            t_move=t_move, t10=t10, t90=t90, rise=t90 - t10, settle=t_settle,
            overshoot=max(0.0, (y[after].max() - 1.0) * 100.0),
            shortfall=float(np.mean((qf - q[tail]) * np.sign(span))) if tail.any() else np.nan,
            vpeak=float(np.max(np.abs(qv[after]))),
            coupling=float(np.max(np.abs(other[after] - other_start))),
            dead=dead, tau=tau,
            t=t, y=y))
    return results


def summarise(results):
    groups = {}
    for r in results:
        groups.setdefault((r['joint'], round(abs(r['delta']), 4)), []).append(r)
    lines = ['| joint | step (rad) | n | first move (ms) | 10-90 rise (ms) | settle 5% (ms) | overshoot (%) '
             '| shortfall (mrad) | peak vel (rad/s) | other joint dev (mrad) | fit dead time (ms) | fit tau (ms) |',
             '|---|---|---|---|---|---|---|---|---|---|---|---|']
    for (joint, step), rs in sorted(groups.items()):
        def ms(key):
            v = np.array([r[key] for r in rs]) * 1e3
            return f'{np.nanmean(v):.0f} ± {np.nanstd(v):.0f}'
        def val(key, scale=1.0, fmt='{:.1f}'):
            v = np.array([r[key] for r in rs]) * scale
            return (fmt + ' ± ' + fmt).format(np.nanmean(v), np.nanstd(v))
        lines.append(f'| q{joint} | {step:.3f} | {len(rs)} | {ms("t_move")} | {ms("rise")} | {ms("settle")} | '
                     f'{val("overshoot")} | {val("shortfall", 1e3)} | {val("vpeak", 1.0, "{:.2f}")} | {val("coupling", 1e3)} | '
                     f'{ms("dead")} | {ms("tau")} |')
    return '\n'.join(lines)


def plot(state, cmds, results, prefix):
    fig, axes = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
    for j, ax in enumerate(axes):
        tc = [c['t'] for c in cmds]
        qc = [c['q'][j] for c in cmds]
        ax.step(tc, qc, where='post', color='0.6', label='command')
        ax.plot(state['t'], state['qpos%d' % (j + 1)], '.-', ms=3, lw=1, label='measured')
        ax.set_ylabel(f'q{j + 1} (rad)')
        ax.grid(alpha=0.3)
        ax.legend(loc='upper right')
    axes[-1].set_xlabel('time (s)')
    fig.tight_layout()
    fig.savefig(prefix + '_timeseries.png', dpi=130)
    plt.close(fig)

    steps = sorted({round(abs(r['delta']), 4) for r in results})
    fig, axes = plt.subplots(2, len(steps), figsize=(4 * len(steps), 6), sharey=True, squeeze=False)
    for j in range(2):
        for s, step in enumerate(steps):
            ax = axes[j][s]
            for r in results:
                if r['joint'] == j + 1 and round(abs(r['delta']), 4) == step:
                    ax.plot(r['t'], r['y'], '.-', ms=3, lw=1, alpha=0.7)
            ax.axhline(1.0, color='k', lw=0.8)
            ax.axvline(0.0, color='k', lw=0.8)
            ax.set_xlim(-0.3, 2.0)
            ax.set_title(f'q{j + 1} step {step:.3f} rad')
            ax.grid(alpha=0.3)
            if s == 0:
                ax.set_ylabel('normalised response')
            if j == 1:
                ax.set_xlabel('time since command (s)')
    fig.tight_layout()
    fig.savefig(prefix + '_steps.png', dpi=130)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('prefix', help='path prefix, e.g. data/step_20261127_120000')
    args = parser.parse_args()
    state, cmds, meta = load(args.prefix)
    dt = np.diff(state['t'])
    results = analyse(state, cmds)
    table = summarise(results)
    report = [f'# Step test {os.path.basename(args.prefix)}', '',
              f'status: {meta["status"]}; base q = ({meta["base"][0]:.4f}, {meta["base"][1]:.4f}); '
              f'{len(state["t"])} samples, {1 / np.median(dt):.1f} Hz median, max gap {dt.max() * 1e3:.0f} ms', '',
              'Times are from the command publish to the /robot_state receive time; they include the serial_node '
              'timer (0-100 ms), the 9600 baud link, the firmware and the feedback period (about +-50 ms). '
              'Response normalised from the measured start; shortfall > 0 = stopped short of the target; '
              'fit = first order plus dead time, y = 1 - exp(-(t - dead) / tau).', '',
              table]
    with open(args.prefix + '_report.md', 'w') as f:
        f.write('\n'.join(report) + '\n')
    plot(state, cmds, results, args.prefix)
    print('\n'.join(report))
    print(f'\nSaved {args.prefix}_report.md, _timeseries.png, _steps.png')


if __name__ == '__main__':
    main()
