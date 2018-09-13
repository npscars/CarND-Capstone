# CarND CAPSTONE

## Project Overview
This is the final project for the Self Driving Car Engineer Nano degree that brings together everything covered over the course of this program. Here we get to explore a real life example that combines computer vision, sensor fusion, and path planning to allow a vehicle to navigate through its environment while obeying traffic lights and obstacles. We borrow concepts learned from previous projects to publish waypoints ahead of the vehicle to indicate the desired path to follow and use neural networks to classify images and predict the state of a traffic light for our path planning controller. The initial project is built and tested on the Unity simulator provided by Udacity which gets uploaded to Carla once approved!


## Getting Started
View the project outline and how to download and bringup the source code from the repository [here](https://github.com/djiglesias/CarND-Capstone/blob/devel/INSTALL.md). Once downloaded, the following commands can be run to compile the application.

    $ git clone https://github.com/djiglesias/CarND-Capstone.git
    $ cd ros
    $ catkin_make
    $ source devel/setup.sh
    $ roslaunch launch/styx.launch
 
 Open the term 3 simulator, which can be downloaded [here](https://github.com/udacity/CarND-Capstone/releases), then disable the manual driving icon and enable the camera for traffic light updates to be sent to the path planning controller!

<p align="center">
 <img src="./res/traffic_light.gif" width=550>
</p>

## Building the ROS Nodes

### Waypoint Updated (Partial)

The first step of the project is to get the ROS controller to publish a sample of waypoints ahead of the car to display the desired trajectory to follow. When the simulator first launches it publishes all the waypoints related to the respective track to `/base_waypoints`, this topic operates as a latch so it is only published once to reduce the amount of processing required during runtime (there are approximately 11,000 waypoints). Additionally, there are eight traffic lights on the simulator track where the location of the stopline is hardcoded via the `sim_traffic_light.config` yaml file loaded upon launch. A helpful walk through for this section is [HERE](https://www.youtube.com/watch?v=6GIFyUzhaQo).

The track is relatively simple as shown below by the blue line with the traffic lights shown as red dots. However, this section only displays the leading N waypoints ahead of the car since there is no data related to the traffic lights. The output of this node publishes the list of leading waypoinys ahead of the car to `/final_waypoints` which is used by the next section for controlling the car via DBW.

<p align="center">
 <img src="./res/track.png" width=550>
</p>

### Drive-By-Wire Controller
Once the Waypoint Updater node is publishing a list N leading waypoints to `/final_waypoints` the drive-by-wire node can be built to adjust the pose of the car with repsect to the next waypoint in the list. Following the project section walkthrough [HERE](https://www.youtube.com/watch?v=kdfXo6atphY) explains the implementation of a PID controller with the Yaw Controller for generating smooth movements between frames that adhere to the jerk restraints imposed on this project. Providing the controllers with the parameters of the vehicle in the simulator allows for the computation of an appropriate braking force per wheel since torque is just a perpendicual force applied at a distance from an origin... or the radius of a wheel multiplied by the force of the car (mass and desired acceleration).

The output of the yaw controller are commands for controlling three import aspects of the vehicle: steering, throttle, and brake. These are published to `/vehicle/steering_cmd`, `/vehicle/throttle_cmd`, and `/vehicle/brake_cmd` respectively that are handled by a third party controller.

### Traffic Light Detector
The Waypoint Updater (partial) and DBW nodes allow the vehicle to successfully drive around the track, but there is no regard to track lights... red lights are merely bad suggestions in this case. The project walkthrough [HERE](https://www.youtube.com/watch?v=oTfArPhstQU) demonstrates how to subscribe to `/vehicle/traffic_lights` to read data regarding updated traffic light states. When running on the simulator the states of the lights can simply be read from the simulator directly, however this will not be the case in real life. For development purposes, the states of the lights were used.

This node subscribes to `/current_pose` which provides the current position (x,y) of the vehicle on the track which is used to search for the closest leading traffic light on the track. Once the traffic light is found the index of that light is published to `/traffic_waypoint` which indicates the next traffic light needed by the Waypoint Updater node. 

<p align="center">
 <img src="./res/green.png" width=280>
 <img src="./res/yellow.png" width=280>
 <img src="./res/red.png" width=280>
</p>


### Waypoint Updater (Full)
... [HERE](https://www.youtube.com/watch?v=2tDrj8KjIL4)

### Run Simulator
...

- respawn on nodes that fail


## Traffic Light Classifier
...

## Track Test
...