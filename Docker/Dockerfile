FROM px4io/px4-dev-ros-melodic:latest

# nvidia-container-runtime
ENV NVIDIA_VISIBLE_DEVICES \
    ${NVIDIA_VISIBLE_DEVICES:-all}
ENV NVIDIA_DRIVER_CAPABILITIES \
    ${NVIDIA_DRIVER_CAPABILITIES:+$NVIDIA_DRIVER_CAPABILITIES,}graphics

## create a non-root user
ARG USER_ID=1000
RUN useradd -m --no-log-init --system  --uid ${USER_ID} ncrl -g sudo
RUN echo '%sudo ALL=(ALL) NOPASSWD:ALL' >> /etc/sudoers
USER ncrl
WORKDIR /home/ncrl
RUN sudo apt-get -o Acquire::ForceIPv4=true update && sudo apt-get -o Acquire::ForceIPv4=true install -q -y \
    vim
RUN mkdir  PX4_RL_WS

RUN echo "source ~/PX4_RL_WS/PX4-Autopilot/Tools/setup_gazebo.bash ~/PX4_RL_WS/PX4-Autopilot/ ~/PX4_RL_WS/PX4-Autopilot/build/px4_sitl_default" >> ~/.bashrc
RUN echo "export ROS_PACKAGE_PATH=$ROS_PACKAGE_PATH:~/PX4_RL_WS/PX4-Autopilot" >> ~/.bashrc
RUN echo "export ROS_PACKAGE_PATH=$ROS_PACKAGE_PATH:~/PX4_RL_WS/PX4-Autopilot/Tools/sitl_gazebo" >> ~/.bashrc

