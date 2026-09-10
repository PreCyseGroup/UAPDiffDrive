from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    pkg_share = FindPackageShare("nn_trajectory_tracker")
    config_file = PathJoinSubstitution([pkg_share, "config", "lemniscate_tracker.yaml"])
    model_file = PathJoinSubstitution([pkg_share, "models", "il_policy.pt"])
    stats_file = PathJoinSubstitution([pkg_share, "models", "norm_stats.npz"])

    odom_topic = LaunchConfiguration("odom_topic")
    cmd_vel_topic = LaunchConfiguration("cmd_vel_topic")
    frame_id = LaunchConfiguration("frame_id")
    device = LaunchConfiguration("device")
    eta = LaunchConfiguration("eta")
    alpha = LaunchConfiguration("alpha")
    max_linear_velocity = LaunchConfiguration("max_linear_velocity")
    max_angular_velocity = LaunchConfiguration("max_angular_velocity")

    return LaunchDescription(
        [
            DeclareLaunchArgument("odom_topic", default_value="/odemetry/filtered"),
            DeclareLaunchArgument("cmd_vel_topic", default_value="/cmd_vel"),
            DeclareLaunchArgument("frame_id", default_value="vicon/world"),
            DeclareLaunchArgument("device", default_value="cpu"),
            DeclareLaunchArgument("model_path", default_value=model_file),
            DeclareLaunchArgument("norm_stats_path", default_value=stats_file),
            DeclareLaunchArgument("eta", default_value="1.0"),
            DeclareLaunchArgument("alpha", default_value="4.0"),
            DeclareLaunchArgument("max_linear_velocity", default_value="0.0"),
            DeclareLaunchArgument("max_angular_velocity", default_value="0.0"),
            Node(
                package="nn_trajectory_tracker",
                executable="lemniscate_nn_tracker",
                name="lemniscate_nn_tracker",
                output="screen",
                parameters=[
                    config_file,
                    {
                        "odom_topic": odom_topic,
                        "cmd_vel_topic": cmd_vel_topic,
                        "frame_id": frame_id,
                        "device": device,
                        "eta": eta,
                        "alpha": alpha,
                        "max_linear_velocity": max_linear_velocity,
                        "max_angular_velocity": max_angular_velocity,
                        "model_path": LaunchConfiguration("model_path"),
                        "norm_stats_path": LaunchConfiguration("norm_stats_path"),
                    },
                ],
            ),
        ]
    )
