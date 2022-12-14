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

# ???????????????GeographicLib exception: File not readable...
sudo /opt/ros/melodic/lib/mavros/install_geographiclib_datasets.sh

# about serial port problem
sudo usermod -a -G tty "USER_name"
ex: sudo usermod -a -G tty ncrl
sudo usermod -a -G dialout "USER_name"

# ?????????fcu_url:="/dev/ttyACM0:921600"
sudo gedit /opt/ros/melodic/share/mavros/launch/px4.launch
# ??????Qground???parameter??????baud rate????????????

# ???????????????FCU: DeviceError:serial:open: Permission denied
chmod 777 /dev/ttyACM0

# test
roslaunch mavros px4.launch
# ?????????????????? IMU detected ??? /mavros/imu/data_raw
# ??????????????????????????? pixhawk ??? type C ??????????????? (??????????????????????????????)
```

### VRPN <a name="expvrpninstall"></a>
```
sudo apt-get install ros-melodic-vrpn-client-ros

#??????server_IP?????????optitrack??????????????????IP
sudo gedit /opt/ros/melodic/share/vrpn_client_ros/launch/sample.launch

#test
roslaunch vrpn_client_ros sample.launch
# ?????????????????? Creating new tracker rigidbody7
# ?????????????????????????????? optitrack setting
```

### OptiTrack <a name="expoptitrackinstall"></a>
1. ?????? rigidbody ???????????? rigidbody7
2. ?????? data streaming ????????? IP ????????? motive ??????????????? IP

### Calibration of UAV (QGround) <a name="expcalibrationinstall"></a>
1. ?????? parameter ??? EKF2_AID_MASK ?????? use GPS ??? EKF2_HGT_MODE ?????? GPS
2. reboot pixhawk to set parameters
3. ?????? compass -> gyro -> accelerometer -> level meter
4. ?????? analysis ???????????? highes_imu ??? acc ??????????????????????????????0.2????????????????????????
6. ?????? EKF2_AID_MASK ??? vision pose fusion ??? vision yaw fusion ??? EKF2_HGT_MODE ??? vision
7. reboot pixhawk to set parameters

---

## How to Run ? <a name="howtorun"></a>
???launch gazebo world??????,????????????indoor_navigation/config??????yaml??????????????????parameter

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
ssh -X -C ncrl@192.168.2.210 # ????????? TX2

roslaunch mavros px4.launch fcu_url:=/dev/ttyACM0:921600 gcs_url:=udp://@192.168.2.178  # gcs = ??????ip
roslaunch vrpn_client_ros sample.launch
rostopic pub /vrpn_client_node/goal/pose geometry_msgs/PoseStamped <Tab>    # ?????????????????????????????????????????????OptiTrack marker???
roslaunch indoor_navigation experiment_settings.launch
rosbag record /drone/desired_action /drone/constrained_action -O DRLCBF_experiment1_with_CBF.bag
rosparam set /px4/drone/safety_distance 0.8     # CBF ????????????
rosparam set /px4/goal/reach_threshold 0.3      # ????????????????????????????????????
python3 exp_test.py --load_model pretrained_weights/CBFDRL_empty_room_obstacle_ep_3500.ckpt --episode_num 1 --constraint {none, full} # ??????CBF??????CBF
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
