# robot-programming

## 빌드

```bash
cd ~/robot-programming
source /opt/ros/humble/setup.bash
rosdep install --from-paths src --ignore-src -r -y
sudo apt install -y ros-humble-realsense2-camera
colcon build --symlink-install
```

## 터미널마다 먼저 실행

```bash
cd ~/robot-programming
source /opt/ros/humble/setup.bash
export ROS_DOMAIN_ID=51
source install/setup.bash
```

## RViz로 로봇 모델 확인 (로봇 없이)

```bash
# 터미널 1
ros2 launch twolink_gazebo gazebo.launch.py use_gazebo:=false

# 터미널 2
ros2 run joint_state_publisher_gui joint_state_publisher_gui
```

## Gazebo 시각화 (로봇 없이)

```bash
# 터미널 1
ros2 launch twolink_gazebo gazebo.launch.py

# 터미널 2
ros2 run twolink_gazebo dummy_data_publisher.py

# 터미널 3
ros2 run twolink_gazebo joint_publisher.py
```

## Gazebo 시각화 (실제 로봇 연결)

```bash
# 터미널 1
ros2 launch twolink_gazebo gazebo.launch.py

# 터미널 2
ros2 run twolink_control serial_node

# 터미널 3
ros2 run twolink_gazebo joint_publisher.py
```

## 키보드로 관절 움직이기 (실제 로봇 연결)

```bash
# 터미널 1
ros2 run twolink_control serial_node

# 터미널 2
ros2 run twolink_control command_node
```

## 마커 인식

```bash
# 터미널 1
ros2 launch realsense2_camera rs_launch.py

# 터미널 2
ros2 run marker_detection detection_realsense
```
