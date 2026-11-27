import bisect
import math
import time

import numpy as np
import rclpy
from geometry_msgs.msg import Pose
from rclpy.node import Node
from std_msgs.msg import Float64MultiArray
from twolink_msgs.msg import CMD, XYpos

from visual_servo import servo_core as sc

MODES = ['track', 'distance_limited', 'joint_limited', 'hold']


class ServoNode(Node):
    def __init__(self):
        super().__init__('visual_servo')
        p = self.declare_parameters('', [
            ('d_goal', sc.D_GOAL), ('v_max', sc.V_MAX), ('a_max', sc.A_MAX), ('omega', sc.OMEGA),
            ('lag_tau', sc.LAG_TAU), ('lead_time', sc.LEAD_TIME), ('coast_time', sc.COAST_TIME),
            ('deadband', sc.DEADBAND), ('track_tau', sc.TRACK_TAU), ('range_tau', sc.RANGE_TAU),
            ('q_min', list(sc.Q_MIN)), ('q_max', list(sc.Q_MAX)),
            ('cam_center_y', sc.CAM_CENTER_Y), ('cam_bottom_z', sc.CAM_BOTTOM_Z),
            ('rate', sc.VisualServo.CONTROL_RATE), ('image_delay', 0.0),
            ('marker_topic', '/aruco_detector/pose')])
        v = {x.name: x.value for x in p}
        self.servo = sc.VisualServo(
            d_goal=v['d_goal'], v_max=v['v_max'], a_max=v['a_max'], omega=v['omega'], lag_tau=v['lag_tau'],
            lead_time=v['lead_time'], coast_time=v['coast_time'], deadband=v['deadband'],
            q_min=v['q_min'], q_max=v['q_max'],
            mount=sc.mount_transform(v['cam_center_y'], v['cam_bottom_z']))
        self.servo.tracker.tau = v['track_tau']
        self.servo.tracker.range_tau = v['range_tau']
        self.ring = sc.reach_ring(self.servo.l2, self.servo.alpha, 1.0, self.servo.q_min, self.servo.q_max)
        self.image_delay = v['image_delay']
        self.dt = 1.0 / v['rate']
        self.state_t = []
        self.state_q = []
        self.markers = []
        self.started = False
        self.last_mode = None
        self.last_lost = None
        self.cmd_pub = self.create_publisher(CMD, 'des_value', 10)
        self.debug_pub = self.create_publisher(Float64MultiArray, 'visual_servo/debug', 10)
        self.create_subscription(XYpos, 'robot_state', self.on_state, 50)
        self.create_subscription(Pose, v['marker_topic'], self.on_marker, 10)
        self.create_timer(self.dt, self.on_timer)
        self.get_logger().info(
            f"visual_servo: d_goal {v['d_goal']} m, rate {v['rate']:.0f} Hz, v_max {v['v_max']} rad/s, a_max "
            f"{v['a_max']} rad/s2, omega {v['omega']}, lag_tau {v['lag_tau']} s, lead {v['lead_time']} s, coast "
            f"{v['coast_time']} s, reach ring {self.ring[0]:.3f}-{self.ring[1]:.3f} m. "
            'Waiting for /robot_state and the first marker.')

    def on_state(self, msg):
        now = time.monotonic()
        self.state_t.append(now)
        self.state_q.append((msg.qpos1, msg.qpos2))
        cutoff = bisect.bisect_left(self.state_t, now - 2.0)
        if cutoff:
            del self.state_t[:cutoff], self.state_q[:cutoff]

    def on_marker(self, msg):
        self.markers.append((time.monotonic(),
                             (msg.position.x, msg.position.y, msg.position.z),
                             (msg.orientation.x, msg.orientation.y, msg.orientation.z, msg.orientation.w)))

    def q_at(self, t):
        i = bisect.bisect_left(self.state_t, t)
        if i <= 0:
            return np.array(self.state_q[0])
        if i >= len(self.state_t):
            return np.array(self.state_q[-1])
        t0, t1 = self.state_t[i - 1], self.state_t[i]
        w = (t - t0) / (t1 - t0) if t1 > t0 else 1.0
        return (1 - w) * np.array(self.state_q[i - 1]) + w * np.array(self.state_q[i])

    def on_timer(self):
        if not self.state_t:
            self.markers.clear()
            return
        now = time.monotonic()
        if now - self.state_t[-1] > 1.0:
            self.get_logger().warning('/robot_state stale for more than 1 s; not commanding', throttle_duration_sec=2.0)
            self.markers.clear()
            return
        marker = self.markers[-1] if self.markers else None
        self.markers.clear()
        if not self.started:
            if marker is None:
                return
            if self.count_publishers('des_value') > 1:
                self.get_logger().error('Another node publishes /des_value (keyboard?). Not starting.',
                                        throttle_duration_sec=2.0)
                return
            self.servo.reset(np.array(self.state_q[-1]))
            self.started = True
            self.get_logger().info('First marker received: servo started.')
        if marker is not None:
            t_marker, p_cam, quat = marker
            q_cmd, info = self.servo.step(p_cam, self.q_at(t_marker - self.image_delay), quat, dt=self.dt)
        else:
            q_cmd, info = self.servo.step(None, np.array(self.state_q[-1]), dt=self.dt)
        msg = CMD()
        msg.q1des, msg.q2des = float(q_cmd[0]), float(q_cmd[1])
        self.cmd_pub.publish(msg)
        m = info['marker_base'] if info['marker_base'] is not None else [math.nan] * 3
        q_ik = info['q_ik'] if info['q_ik'] is not None else [math.nan] * 2
        target = info['target'] if info['target'] is not None else [math.nan] * 2
        q_now = self.state_q[-1]
        self.debug_pub.publish(Float64MultiArray(data=[
            info['e_theta'], info['e_d'], info['e_psi'], m[0], m[1], q_ik[0], q_ik[1],
            q_cmd[0], q_cmd[1], q_now[0], q_now[1], float(info['held']),
            float(bool(info['reachable'])), float(bool(info['inside_limits'])),
            float(MODES.index(info['mode'])), info['d_target'],
            float(info['seen']), float(info['lost']), target[0], target[1],
            info['q_ref'][0], info['q_ref'][1]]))
        if info['mode'] != self.last_mode or info['lost'] != self.last_lost:
            text = {'track': 'Tracking.',
                    'distance_limited': 'Distance goal not reachable: centring only, holding the nearest distance.',
                    'joint_limited': 'Joint 1 at its limit: centring with joint 2 only.',
                    'hold': 'No usable target: holding.'}[info['mode']]
            if info['lost'] and info['mode'] != 'hold':
                text = 'Marker lost: staying on its last predicted position.'
            self.get_logger().info(text)
            self.last_mode, self.last_lost = info['mode'], info['lost']


def main(args=None):
    rclpy.init(args=args)
    node = ServoNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    if rclpy.ok():
        rclpy.shutdown()


if __name__ == '__main__':
    main()
