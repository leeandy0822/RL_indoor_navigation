# Indoor Navigation

![](https://user-images.githubusercontent.com/40656204/155317580-c9717d80-c594-4ad7-b250-aac24fb23abd.gif)

![](https://user-images.githubusercontent.com/40656204/139214160-2353717e-466e-46fb-a1d6-18caa950e6bd.gif)



## Architecture <a name="architecture"></a>
![](https://user-images.githubusercontent.com/40656204/174959017-14469587-adcb-401f-ab1e-76797b25b34e.png)

---

## Development Platform <a name="platform"></a>
* Ubuntu 18.04
* ROS Melodic
* Gazebo 9.19.0
* Python 3.7.0
---

## Simulation Environment Setup<a name="simulationinstall"></a>

-  [PX4-AUTOPILOT Download](https://drive.google.com/file/d/1UOc7kZXKpTuCZGh5KfG4-G2deOsJlv65/view)
-  [Model Download](https://drive.google.com/drive/folders/15oR-j1Nj4WMCyUofwuvOF4qQ-nsbXPid?usp=sharing)
-  [Replay Buffer Download](https://drive.google.com/drive/folders/1weF114nljVHWGdSbjrCx1i-awduWp4zq?usp=sharing)

### Docker Environment<a name="simpx4install"></a>
1. `install docker for ubuntu`
2. `git clone repo`

### PX4 Environment <a name="simpx4install"></a>
1. `Download PX4_Autopilot`
2. `$ mv PX4_Autopilot ~/indoor_navigation/`
3. `$ cd indoor_navigation`
4. `$ source docker_start.sh`
5. `$ cd PX4_RL_WS/PX4-Autopilot`
6.  `$ make px4_sitl gazebo`
7.  `Run these commands in terminal before using PX4 function`
    ```shell
    source ~/PX4-Autopilot/Tools/setup_gazebo.bash ~/PX4-Autopilot/ ~/PX4-Autopilot/build/px4_sitl_default
    export ROS_PACKAGE_PATH=$ROS_PACKAGE_PATH:~/PX4-Autopilot
    export ROS_PACKAGE_PATH=$ROS_PACKAGE_PATH:~/PX4-Autopilot/Tools/sitl_gazebo
    ```

## Experiment Environment Setup <a name="experimentinstall"></a>
### RPLiDAR Driver <a name="exprplidarinstall"></a>
1. `$ cd ~/RL_indoor_navigation/catkin_ws/src`
2. `$ git clone https://github.com/zonghan0904/rplidar_python.git`
3. `$ cd ~/RL_indoor_navigation/catkin_ws`
4. `$ catkin_make`
5. `$ roscd rplidar_python/lib_for_install`
6. unpack construct-2.5.2.zip and pyserial-3.0.1.tar.gz
7. go to unpacked folders and run `$ sudo python setup.py install`

### MAVROS <a name="expmavrosinstall"></a>
```
sudo apt-get install ros-melodic-mavros ros-melodic-mavros-extras screen openssh-server git

# 如果出現：GeographicLib exception: File not readable...
sudo /opt/ros/melodic/lib/mavros/install_geographiclib_datasets.sh

# about serial port problem
sudo usermod -a -G tty "USER_name"
ex: sudo usermod -a -G tty ncrl
sudo usermod -a -G dialout "USER_name"

# 更改為fcu_url:="/dev/ttyACM0:921600"
sudo gedit /opt/ros/melodic/share/mavros/launch/px4.launch
# 配合Qground的parameter裡的baud rate也需要改

# 如果出現：FCU: DeviceError:serial:open: Permission denied
chmod 777 /dev/ttyACM0

# test
roslaunch mavros px4.launch
# 確認是否回傳 IMU detected 和 /mavros/imu/data_raw
# 如果沒有回傳就斷開 pixhawk 的 type C 並重新接上 (需在蜂鳴器響之前完成)
```

### VRPN <a name="expvrpninstall"></a>
```
sudo apt-get install ros-melodic-vrpn-client-ros

#更改server_IP為裝有optitrack軟體的電腦的IP
sudo gedit /opt/ros/melodic/share/vrpn_client_ros/launch/sample.launch

#test
roslaunch vrpn_client_ros sample.launch
# 確認是否回傳 Creating new tracker rigidbody7
# 如果沒有就參考以下的 optitrack setting
```

### OptiTrack <a name="expoptitrackinstall"></a>
1. 框出 rigidbody 並命名為 rigidbody7
2. 設定 data streaming 的本端 IP 為開啟 motive 本身的電腦 IP

### Calibration of UAV (QGround) <a name="expcalibrationinstall"></a>
1. 調整 parameter 把 EKF2_AID_MASK 設成 use GPS 、 EKF2_HGT_MODE 設成 GPS
2. reboot pixhawk to set parameters
3. 校正 compass -> gyro -> accelerometer -> level meter
4. 使用 analysis 工具觀察 highes_imu 的 acc 值，數值變化不可超過0.2，否則須重新校正
6. 調整 EKF2_AID_MASK 成 vision pose fusion 跟 vision yaw fusion 、 EKF2_HGT_MODE 成 vision
7. reboot pixhawk to set parameters

---

## How to Run ? <a name="howtorun"></a>
在launch gazebo world之前,可以調整indoor_navigation/config裡的yaml檔來調整一些parameter

### Simulation <a name="runsimulation"></a>
- Docker
```
source docker_start.sh
roslaunch indoor_navigation px4_sim.launch world_name:=<name of world in indoor_navigation/worlds>
```
- PC
```
python train.py --save_buffer --save_model --constraint full
python test.py --load_model <path_to_weights> --constraint full --visualize
```

### Experiment <a name="runexperiment"></a>
```
ssh -X -C ncrl@192.168.2.210 # 連進去 TX2

roslaunch mavros px4.launch fcu_url:=/dev/ttyACM0:921600 gcs_url:=udp://@192.168.2.178  # gcs = 筆電ip
roslaunch vrpn_client_ros sample.launch
rostopic pub /vrpn_client_node/goal/pose geometry_msgs/PoseStamped <Tab>    # 輸入目標的位置資訊（或是直接放OptiTrack marker）
roslaunch indoor_navigation experiment_settings.launch
rosbag record /drone/desired_action /drone/constrained_action -O DRLCBF_experiment1_with_CBF.bag
rosparam set /px4/drone/safety_distance 0.8     # CBF 安全距離
rosparam set /px4/goal/reach_threshold 0.3      # 判別是否到達目標點的距離
python3 exp_test.py --load_model pretrained_weights/CBFDRL_empty_room_obstacle_ep_3500.ckpt --episode_num 1 --constraint {none, full} # 不加CBF或加CBF
```

---

## Unit Test <a name="unittest"></a>
`$ cd ~/catkin_ws/src/indoor_navigation/src` (must go to this directory)

### PX4 Gym-Like Environment <a name="px4gym"></a>
* `$ python -m indoor_navigation.envs.px4_env` (simulation)
* `$ python -m indoor_navigation.envs.exp_px4_env` (experiment)

### Constraint Generator <a name="constraintgenerator"></a>
* `$ python -m indoor_navigation.obstacle_avoidance.cbf` (simulation)
* `$ python -m indoor_navigation.obstacle_avoidance.exp_cbf` (experiment)

### Local Map Generator <a name="localgenerator"></a>
* `$ python -m indoor_navigation.motion_planning.local_map_generator` (simulation)
* `$ python -m indoor_navigation.motion_planning.exp_local_map_generator` (experiment)

### Forward Spanning Tree <a name="fst"></a>
* `$ python -m indoor_navigation.motion_planning.fst` (simulation)
* `$ python -m indoor_navigation.motion_planning.exp_fst` (experiment)
