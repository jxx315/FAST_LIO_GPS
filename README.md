# FAST_LIO_GPS

## About FAST_LIO_GPS
this is a  modified version of  [FAST-LIO2](https://github.com/hku-mars/FAST_LIO) and [SC-PGO](https://github.com/gisbi-kim/SC-A-LOAM),  With the help of gps, We can  obtain map with UTM coordinate. So, using global map , Vehicle navigation can run in the world coordinate system



![avatar](./docs/live_map.jpg)

![avatar](./docs/map.png)

![avatar](./docs/submap4.png)

## News

   

## Features
-  directly mapping in UTM Coordinate system
-  need gps-->lidar extrinsic_T and extrinsic_R

## Prerequisites

- Ubuntu and ROS
Ubuntu >= 18.04
ROS >= Melodic. ROS Installation

- build

```
    cd ~/$A_ROS_DIR$/src
    git clone https://github.com/jxx315/FAST_LIO_GPS.git
    cd ..
    catkin_make
    source devel/setup.bash
```


- how to use

```
    source devel/setup.bash
    roslaunch fast_lio mapping_rs32.launch
```

```
    source devel/setup.bash
    roslaunch aloam_velodyne fastlio_rs32_gps.launch
```

- save pcd 

```
rosservice call /opt/save_map "utm: false
resolution: 0.0
destination: ''" 


```

## TODO

- clean up the code
- using low cost Single antenna gps&RTK for global localization

    





