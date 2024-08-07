# ROS

## Ros-humble

### Tools

#### ros-humble-web-video-server

网页查看相机图像数据

- Install

```bash

apt update
apt install ros-humble-async-web-server-cpp ros-humble-cv-bridge ros-humble-image-transport
cd ~/web_ws/src
git clone --branch ros2 https://github.com/RobotWebTools/web_video_server
cd ~/web_ws
colcon build --packages-select web_video_server

```

- Usage

```bash

source ~/web_ws/install/setup.bash
ros2 run web_video_server web_video_server

```

网页端输入 IP:8080即可（默认端口）