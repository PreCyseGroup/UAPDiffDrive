from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import rclpy
from ament_index_python.packages import get_package_share_directory
from geometry_msgs.msg import PoseStamped, Twist
from nav_msgs.msg import Odometry, Path as PathMsg
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from std_msgs.msg import Float32MultiArray

from .policy import RobotParams, TrainedWheelPolicy, wheel_to_unicycle
from .trajectory import Trajectory, generate_lemniscate, tracking_features


def yaw_from_quaternion(x: float, y: float, z: float, w: float) -> float:
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    return math.atan2(siny_cosp, cosy_cosp)


def quaternion_from_yaw(yaw: float) -> tuple[float, float, float, float]:
    half = 0.5 * yaw
    return 0.0, 0.0, math.sin(half), math.cos(half)


class LemniscateNNTracker(Node):
    def __init__(self) -> None:
        super().__init__("lemniscate_nn_tracker")

        share_dir = Path(get_package_share_directory("nn_trajectory_tracker"))
        default_model = str(share_dir / "models" / "il_policy.pt")
        default_stats = str(share_dir / "models" / "norm_stats.npz")

        self.declare_parameter("odom_topic", "/odemetry/filtered")
        self.declare_parameter("cmd_vel_topic", "/cmd_vel")
        self.declare_parameter("frame_id", "vicon/world")
        self.declare_parameter("model_path", default_model)
        self.declare_parameter("norm_stats_path", default_stats)
        self.declare_parameter("device", "cpu")
        self.declare_parameter("control_rate_hz", 60.0)
        self.declare_parameter("lookahead_time", 1.0 / 60.0)
        self.declare_parameter("odom_timeout_sec", 0.25)
        self.declare_parameter("publish_zero_before_first_odom", True)
        self.declare_parameter("anchor_trajectory_to_first_odom", True)
        self.declare_parameter("repeat_trajectory", True)
        self.declare_parameter("eta", 1.0)
        self.declare_parameter("alpha", 4.0)
        self.declare_parameter("phase", 0.0)
        self.declare_parameter("x_offset", 0.0)
        self.declare_parameter("y_offset", 0.0)
        self.declare_parameter("wheel_radius", 0.04445)
        self.declare_parameter("wheelbase", 0.393)
        self.declare_parameter("wheel_speed_limit", 10.0)
        self.declare_parameter("max_linear_velocity", 0.0)
        self.declare_parameter("max_angular_velocity", 0.0)
        self.declare_parameter("odom_qos_reliable", True)

        self.odom_topic = str(self.get_parameter("odom_topic").value)
        self.cmd_vel_topic = str(self.get_parameter("cmd_vel_topic").value)
        self.frame_id = str(self.get_parameter("frame_id").value)
        self.control_dt = 1.0 / float(self.get_parameter("control_rate_hz").value)
        self.lookahead_steps = max(0, int(round(float(self.get_parameter("lookahead_time").value) / self.control_dt)))
        self.odom_timeout_sec = float(self.get_parameter("odom_timeout_sec").value)
        self.publish_zero_before_first_odom = bool(self.get_parameter("publish_zero_before_first_odom").value)
        self.anchor_trajectory_to_first_odom = bool(self.get_parameter("anchor_trajectory_to_first_odom").value)
        self.repeat_trajectory = bool(self.get_parameter("repeat_trajectory").value)
        self.max_linear_velocity = float(self.get_parameter("max_linear_velocity").value)
        self.max_angular_velocity = float(self.get_parameter("max_angular_velocity").value)

        self.robot = RobotParams(
            wheel_radius=float(self.get_parameter("wheel_radius").value),
            wheelbase=float(self.get_parameter("wheelbase").value),
            wheel_speed_limit=float(self.get_parameter("wheel_speed_limit").value),
        )
        model_path = str(self.get_parameter("model_path").value)
        stats_path = str(self.get_parameter("norm_stats_path").value)
        self.policy = TrainedWheelPolicy(
            model_path if model_path else default_model,
            stats_path if stats_path else default_stats,
            device=str(self.get_parameter("device").value),
        )

        self.trajectory = self._make_trajectory(
            float(self.get_parameter("x_offset").value),
            float(self.get_parameter("y_offset").value),
        )
        self.start_time = None
        self.last_odom_time = None
        self.last_pose: tuple[float, float, float] | None = None
        self.path_published = False

        reliable = bool(self.get_parameter("odom_qos_reliable").value)
        odom_qos = QoSProfile(depth=10)
        odom_qos.reliability = ReliabilityPolicy.RELIABLE if reliable else ReliabilityPolicy.BEST_EFFORT

        path_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )
        self.cmd_pub = self.create_publisher(Twist, self.cmd_vel_topic, 10)
        self.ref_pose_pub = self.create_publisher(PoseStamped, "nn_tracker/reference_pose", 10)
        self.path_pub = self.create_publisher(PathMsg, "nn_tracker/reference_path", path_qos)
        self.feature_pub = self.create_publisher(Float32MultiArray, "nn_tracker/policy_features", 10)
        self.wheel_pub = self.create_publisher(Float32MultiArray, "nn_tracker/wheel_speeds", 10)
        self.create_subscription(Odometry, self.odom_topic, self._odom_callback, odom_qos)
        self.timer = self.create_timer(self.control_dt, self._control_callback)

        self.get_logger().info(
            f"Tracking lemniscate from {self.odom_topic} to {self.cmd_vel_topic}; frame_id={self.frame_id}"
        )

    def _make_trajectory(self, x_offset: float, y_offset: float) -> Trajectory:
        return generate_lemniscate(
            dt=self.control_dt,
            eta=float(self.get_parameter("eta").value),
            alpha=float(self.get_parameter("alpha").value),
            phase=float(self.get_parameter("phase").value),
            x_offset=x_offset,
            y_offset=y_offset,
        )

    def _odom_callback(self, msg: Odometry) -> None:
        p = msg.pose.pose.position
        q = msg.pose.pose.orientation
        yaw = yaw_from_quaternion(q.x, q.y, q.z, q.w)
        self.last_pose = (float(p.x), float(p.y), float(yaw))
        self.last_odom_time = self.get_clock().now()

        if self.start_time is None:
            self.start_time = self.last_odom_time
            if self.anchor_trajectory_to_first_odom:
                x_offset = float(p.x) - float(self.trajectory.x[0])
                y_offset = float(p.y) - float(self.trajectory.y[0])
                self.trajectory = self._make_trajectory(x_offset, y_offset)
                self.path_published = False
            self.get_logger().info(
                f"Started trajectory at x={p.x:.3f}, y={p.y:.3f}, yaw={yaw:.3f}; "
                f"duration={self.trajectory.t[-1]:.2f}s"
            )

    def _control_callback(self) -> None:
        now = self.get_clock().now()
        if self.last_pose is None or self.start_time is None:
            if self.publish_zero_before_first_odom:
                self._publish_zero()
            return

        if self.last_odom_time is None:
            self._publish_zero()
            return
        odom_age = (now - self.last_odom_time).nanoseconds * 1e-9
        if odom_age > self.odom_timeout_sec:
            self._publish_zero()
            self.get_logger().warn(f"Odometry stale for {odom_age:.3f}s; publishing zero cmd_vel.", throttle_duration_sec=1.0)
            return

        if not self.path_published:
            self._publish_path(now)
            self.path_published = True

        elapsed = (now - self.start_time).nanoseconds * 1e-9
        traj_duration = float(self.trajectory.t[-1])
        traj_time = elapsed % traj_duration if self.repeat_trajectory else min(elapsed, traj_duration)
        k = int(np.searchsorted(self.trajectory.t, traj_time, side="left"))
        k = int(np.clip(k, 0, len(self.trajectory.t) - 1))

        x, y, yaw = self.last_pose
        feature = tracking_features(x, y, yaw, self.trajectory, k, self.lookahead_steps)
        wr, wl = self.policy.wheel_speeds(feature, self.robot)
        linear, angular = wheel_to_unicycle(wr, wl, self.robot)
        if self.max_linear_velocity > 0.0:
            linear = float(np.clip(linear, -self.max_linear_velocity, self.max_linear_velocity))
        if self.max_angular_velocity > 0.0:
            angular = float(np.clip(angular, -self.max_angular_velocity, self.max_angular_velocity))

        cmd = Twist()
        cmd.linear.x = linear
        cmd.angular.z = angular
        self.cmd_pub.publish(cmd)

        self._publish_reference_pose(now, k)
        self.feature_pub.publish(Float32MultiArray(data=feature.astype(np.float32).tolist()))
        self.wheel_pub.publish(Float32MultiArray(data=[float(wr), float(wl)]))

    def _publish_zero(self) -> None:
        self.cmd_pub.publish(Twist())

    def _publish_reference_pose(self, stamp, k: int) -> None:
        msg = PoseStamped()
        msg.header.stamp = stamp.to_msg()
        msg.header.frame_id = self.frame_id
        msg.pose.position.x = float(self.trajectory.x[k])
        msg.pose.position.y = float(self.trajectory.y[k])
        qx, qy, qz, qw = quaternion_from_yaw(float(self.trajectory.theta[k]))
        msg.pose.orientation.x = qx
        msg.pose.orientation.y = qy
        msg.pose.orientation.z = qz
        msg.pose.orientation.w = qw
        self.ref_pose_pub.publish(msg)

    def _publish_path(self, stamp) -> None:
        path = PathMsg()
        path.header.stamp = stamp.to_msg()
        path.header.frame_id = self.frame_id
        stride = max(1, len(self.trajectory.t) // 1000)
        for k in range(0, len(self.trajectory.t), stride):
            pose = PoseStamped()
            pose.header = path.header
            pose.pose.position.x = float(self.trajectory.x[k])
            pose.pose.position.y = float(self.trajectory.y[k])
            qx, qy, qz, qw = quaternion_from_yaw(float(self.trajectory.theta[k]))
            pose.pose.orientation.x = qx
            pose.pose.orientation.y = qy
            pose.pose.orientation.z = qz
            pose.pose.orientation.w = qw
            path.poses.append(pose)
        self.path_pub.publish(path)


def main(args: list[str] | None = None) -> None:
    rclpy.init(args=args)
    node = LemniscateNNTracker()
    try:
        rclpy.spin(node)
    finally:
        node._publish_zero()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
