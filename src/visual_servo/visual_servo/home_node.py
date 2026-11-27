import math
import sys
import time

import numpy as np
import rclpy
from rclpy.node import Node
from twolink_msgs.msg import CMD, XYpos

from visual_servo import servo_core as sc


class HomeNode(Node):
    def __init__(self):
        super().__init__('home')
        p = self.declare_parameters('', [
            ('home', [0.0, math.pi / 2]), ('max_vel', 0.5), ('rate', 20.0), ('start_delay', 2.0),
            ('tolerance', 0.02), ('settle_timeout', 4.0), ('state_timeout', 5.0), ('max_error', 0.4)])
        self.v = {x.name: x.value for x in p}
        self.state = None
        self.state_time = None
        self.pub = self.create_publisher(CMD, 'des_value', 10)
        self.create_subscription(XYpos, 'robot_state', self.on_state, 50)

    def on_state(self, msg):
        self.state = np.array([msg.qpos1, msg.qpos2])
        self.state_time = time.monotonic()

    def publish(self, q):
        msg = CMD()
        msg.q1des, msg.q2des = float(q[0]), float(q[1])
        self.pub.publish(msg)

    def spin_for(self, seconds):
        end = time.monotonic() + seconds
        while time.monotonic() < end:
            rclpy.spin_once(self, timeout_sec=0.005)

    def state_ok(self):
        return self.state_time is not None and time.monotonic() - self.state_time < 1.0

    def run(self):
        log = self.get_logger()
        home = np.clip(np.array(self.v['home'], dtype=float), sc.Q_MIN, sc.Q_MAX)
        end = time.monotonic() + self.v['state_timeout']
        while self.state is None and time.monotonic() < end:
            rclpy.spin_once(self, timeout_sec=0.05)
        if self.state is None:
            log.error('No /robot_state: is serial_node running and the robot started? Not homing.')
            return 1
        if self.count_publishers('des_value') > 1:
            log.error('Another node publishes /des_value (keyboard?). Not homing.')
            return 1
        start = self.state.copy()
        delta = home - start
        if np.max(np.abs(delta)) < self.v['tolerance']:
            log.info(f'Already at home ({start[0]:+.3f}, {start[1]:+.3f}).')
            return 0
        duration = max(1.0, 1.5 * float(np.max(np.abs(delta))) / self.v['max_vel'])
        log.warning(f'Moving to home ({home[0]:+.3f}, {home[1]:+.3f}) from ({start[0]:+.3f}, {start[1]:+.3f}) '
                    f'in {duration:.1f} s, starting in {self.v["start_delay"]:.0f} s. Keep the e-stop in reach.')
        self.spin_for(self.v['start_delay'])
        t0 = time.monotonic()
        period = 1.0 / self.v['rate']
        while True:
            s = min(1.0, (time.monotonic() - t0) / duration)
            target = start + delta * (3 * s * s - 2 * s * s * s)
            if not self.state_ok():
                log.error('/robot_state stopped during homing: stopping here.')
                return 1
            if np.max(np.abs(self.state - target)) > self.v['max_error']:
                self.publish(self.state)
                log.error(f'Robot does not follow (error > {self.v["max_error"]} rad): holding the current pose.')
                return 1
            self.publish(target)
            if s >= 1.0:
                break
            self.spin_for(period)
        end = time.monotonic() + self.v['settle_timeout']
        while time.monotonic() < end:
            self.publish(home)
            self.spin_for(period)
            if np.max(np.abs(self.state - home)) < self.v['tolerance']:
                log.info(f'Home reached ({self.state[0]:+.3f}, {self.state[1]:+.3f}).')
                return 0
        log.warning(f'Home not reached within {self.v["tolerance"]} rad: now ({self.state[0]:+.3f}, '
                    f'{self.state[1]:+.3f}). The robot keeps the home target.')
        return 2


def main(args=None):
    rclpy.init(args=args)
    node = HomeNode()
    code = 1
    try:
        code = node.run()
    except KeyboardInterrupt:
        try:
            if node.state is not None:
                node.publish(node.state)
        except Exception:
            pass
        code = 1
    node.destroy_node()
    if rclpy.ok():
        rclpy.shutdown()
    sys.exit(code)


if __name__ == '__main__':
    main()
