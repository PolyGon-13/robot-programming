#!/usr/bin/env python3
import argparse
import csv
import json
import os
import sys
import time

import rclpy
from rclpy.node import Node
from twolink_msgs.msg import CMD, XYpos

Q_LIMITS = ((-3.0, 3.0), (-3.0, 3.0))
PUBLISH_PERIOD = 0.02
STATE_TIMEOUT = 1.0
TRACK_TIMEOUT = 3.0


def build_plan(base, steps, repeats, hold, settle):
    plan = [(0.0, base[0], base[1], 'start')]
    t = settle
    for joint in (0, 1):
        for _ in range(repeats):
            for step in steps:
                for sign in (1.0, -1.0):
                    target = list(base)
                    target[joint] += sign * step
                    plan.append((t, target[0], target[1], f'q{joint + 1} {sign * step:+.3f}'))
                    t += hold
                    plan.append((t, base[0], base[1], f'q{joint + 1} back'))
                    t += hold
    plan.append((t, base[0], base[1], 'end'))
    return plan


class StepTest(Node):
    def __init__(self):
        super().__init__('step_test')
        self.pub = self.create_publisher(CMD, 'des_value', 10)
        self.create_subscription(XYpos, 'robot_state', self.on_state, 50)
        self.state = None
        self.state_time = None
        self.rows = []
        self.target = None
        self.label = ''
        self.t0 = None

    def on_state(self, msg):
        now = time.monotonic()
        self.state = msg
        self.state_time = now
        if self.t0 is not None:
            self.rows.append([
                f'{now - self.t0:.4f}', msg.qpos1, msg.qpos2, msg.qvel1, msg.qvel2,
                self.target[0], self.target[1], self.label])

    def publish(self, q1, q2):
        msg = CMD()
        msg.q1des = float(q1)
        msg.q2des = float(q2)
        self.pub.publish(msg)


def spin_for(node, seconds):
    end = time.monotonic() + seconds
    while time.monotonic() < end:
        rclpy.spin_once(node, timeout_sec=0.005)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--steps', type=float, nargs='+', default=[0.05, 0.1, 0.2])
    parser.add_argument('--repeats', type=int, default=2)
    parser.add_argument('--hold', type=float, default=3.0)
    parser.add_argument('--settle', type=float, default=2.0)
    parser.add_argument('--out-dir', default='.')
    parser.add_argument('--yes', action='store_true')
    args = parser.parse_args()

    rclpy.init()
    node = StepTest()
    print('Waiting for /robot_state ...')
    end = time.monotonic() + 5.0
    while node.state is None and time.monotonic() < end:
        rclpy.spin_once(node, timeout_sec=0.05)
    if node.state is None:
        print('No /robot_state within 5 s. Is serial_node running with the same ROS_DOMAIN_ID / ROS_LOCALHOST_ONLY?')
        return 1
    if node.count_publishers('des_value') > 1:
        print('Another node publishes /des_value (keyboard command_node?). Stop it first.')
        return 1

    base = (node.state.qpos1, node.state.qpos2)
    plan = build_plan(base, args.steps, args.repeats, args.hold, args.settle)
    for _, q1, q2, label in plan:
        for value, (lo, hi), name in ((q1, Q_LIMITS[0], 'q1'), (q2, Q_LIMITS[1], 'q2')):
            if not lo <= value <= hi:
                print(f'Target {name}={value:.3f} ({label}) is outside [{lo}, {hi}]. Aborting.')
                return 1

    duration = plan[-1][0]
    print(f'Base q1={base[0]:.4f} q2={base[1]:.4f} rad')
    print(f'Steps {args.steps} rad, +/- each, q1 then q2, {args.repeats} repeats, hold {args.hold} s')
    print(f'{len(plan) - 2} commands, {duration:.0f} s total. Keep the e-stop in reach.')
    if not args.yes and input('Start? [y/N] ').strip().lower() != 'y':
        print('Cancelled.')
        return 0
    spin_for(node, 0.5)
    if time.monotonic() - node.state_time > STATE_TIMEOUT:
        print('/robot_state stopped while waiting for confirmation. Aborting.')
        return 1
    drift = max(abs(node.state.qpos1 - base[0]), abs(node.state.qpos2 - base[1]))
    if drift > 0.01:
        print(f'Robot moved {drift:.3f} rad since the plan was made. Run again.')
        return 1

    stamp = time.strftime('%Y%m%d_%H%M%S')
    os.makedirs(args.out_dir, exist_ok=True)
    prefix = os.path.join(args.out_dir, f'step_{stamp}')
    events = []
    node.target = base
    node.label = 'start'
    node.t0 = time.monotonic()
    index = 0
    last_pub = 0.0
    off_track_since = None
    status = 'completed'
    try:
        while True:
            now = time.monotonic() - node.t0
            while index < len(plan) and plan[index][0] <= now:
                _, q1, q2, label = plan[index]
                node.target = (q1, q2)
                node.label = label
                node.publish(q1, q2)
                last_pub = now
                events.append([f'{now:.4f}', q1, q2, label])
                if label != 'end':
                    print(f'{now:7.2f} s  {label:12s} q1des={q1:+.4f} q2des={q2:+.4f}')
                index += 1
            if index >= len(plan):
                spin_for(node, 1.0)
                break
            if now - last_pub >= PUBLISH_PERIOD:
                node.publish(*node.target)
                last_pub = now
            if time.monotonic() - node.state_time > STATE_TIMEOUT:
                status = 'aborted: /robot_state stopped'
                break
            error = max(abs(node.state.qpos1 - node.target[0]), abs(node.state.qpos2 - node.target[1]))
            limit = 0.05 + 2.0 * max(args.steps)
            if error > limit:
                off_track_since = off_track_since or time.monotonic()
                if time.monotonic() - off_track_since > TRACK_TIMEOUT:
                    status = f'aborted: tracking error {error:.3f} rad > {limit:.3f} for {TRACK_TIMEOUT} s'
                    break
            else:
                off_track_since = None
            rclpy.spin_once(node, timeout_sec=0.002)
    except KeyboardInterrupt:
        status = 'aborted: Ctrl+C'

    if status != 'completed':
        hold = (node.state.qpos1, node.state.qpos2) if 'tracking' in status else base
        for _ in range(10):
            node.publish(*hold)
            spin_for(node, 0.02)
        print(f'{status}. Holding q1={hold[0]:.4f} q2={hold[1]:.4f}.')

    with open(prefix + '_state.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['t', 'qpos1', 'qpos2', 'qvel1', 'qvel2', 'q1des', 'q2des', 'label'])
        writer.writerows(node.rows)
    with open(prefix + '_cmd.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['t', 'q1des', 'q2des', 'label'])
        writer.writerows(events)
    with open(prefix + '_meta.json', 'w') as f:
        json.dump({'base': base, 'steps': args.steps, 'repeats': args.repeats, 'hold': args.hold,
                   'settle': args.settle, 'status': status, 'start': stamp}, f, indent=1)
    print(f'{status}. Saved {prefix}_state.csv, _cmd.csv, _meta.json ({len(node.rows)} state samples)')
    node.destroy_node()
    rclpy.shutdown()
    return 0


if __name__ == '__main__':
    sys.exit(main())
