source ~/PX4_RL_WS/PX4-Autopilot/Tools/setup_gazebo.bash ~/PX4_RL_WS/PX4-Autopilot/ ~/PX4_RL_WS/PX4-Autopilot/build/px4_sitl_default
export ROS_PACKAGE_PATH=$ROS_PACKAGE_PATH:~/PX4_RL_WS/PX4-Autopilot
export ROS_PACKAGE_PATH=$ROS_PACKAGE_PATH:~/PX4_RL_WS/PX4-Autopilot/Tools/sitl_gazebo

source  ~/PX4_RL_WS/catkin_ws/devel/setup.bash
