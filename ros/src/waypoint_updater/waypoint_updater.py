#!/usr/bin/env python

import rospy
from geometry_msgs.msg import PoseStamped
from styx_msgs.msg import Lane, Waypoint
from std_msgs.msg import Int32
from scipy.spatial import KDTree
import numpy as np
import math

'''
This node will publish waypoints from the car's current position to some `x` distance ahead.

As mentioned in the doc, you should ideally first implement a version which does not care
about traffic lights or obstacles.

Once you have created dbw_node, you will update this node to use the status of traffic lights too.

Please note that our simulator also provides the exact location of traffic lights and their
current status in `/vehicle/traffic_lights` message. You can use this message to build this node
as well as to verify your TL classifier.

TODO (for Yousuf and Aaron): Stopline location for each traffic light.
'''

LOOKAHEAD_WPS = 50 # Number of waypoints we will publish. You can change this number
MAX_DECEL = 0.5


class WaypointUpdater(object):
    def __init__(self):

        rospy.init_node('waypoint_updater',log_level=rospy.DEBUG)

        # TODO: Add other member variables you need below
        self.pose = None
        self.waypoints_2d = None
        #self.base_waypoints = None #-- only for partial implementations
        self.waypoint_tree = None

        #only for full waypoint update code
        self.base_lane = None
        self.stopline_wp_idx = -1


        rospy.Subscriber('/current_pose', PoseStamped, self.pose_cb)
        rospy.Subscriber('/base_waypoints', Lane, self.waypoints_cb)

        # TODO: Add a subscriber for /traffic_waypoint and /obstacle_waypoint below
        rospy.Subscriber('/traffic_waypoint', Int32, self.traffic_cb)
        #rospy.Subscriber('/obstacle_waypoint', ) # needs to publish /obstacle_waypoint first


        self.final_waypoints_pub = rospy.Publisher('final_waypoints', Lane, queue_size=1)

        self.loop()
    '''  #--used in partial waypoint updater 
    def loop(self):
        rate = rospy.Rate(50) # can be educed to around 30Hz due to the frequency of around 30Hz as drive by wire runs around that frequency
        while not rospy.is_shutdown():
            if self.pose and self.base_waypoints:
                # Get closest waypoint
                closet_waypoint_idx = self.get_closest_waypoint_idx()
                self.publish_waypoints(closet_waypoint_idx)
            rate.sleep()
    '''
    def loop(self):
        rate = rospy.Rate(50)
        while not rospy.is_shutdown():
            if self.pose and self.base_lane:
                #rospy.logwarn("i should be able to publish waypoints")
                self.publish_waypoints()
            rate.sleep()

    def publish_waypoints(self):
        final_lane = self.generate_lane()
        self.final_waypoints_pub.publish(final_lane)

    def generate_lane(self):
        lane = Lane()
        lane.header = self.base_lane.header
        closest_idx = self.get_closest_waypoint_idx()
        #rospy.logwarn("The closest idx is {0}".format(closest_idx))
        farthest_idx = closest_idx + LOOKAHEAD_WPS
        base_waypoints = self.base_lane.waypoints[closest_idx:farthest_idx]

        if self.stopline_wp_idx == -1 or (self.stopline_wp_idx >= farthest_idx):
            lane.waypoints = base_waypoints
        else:#potentially detected a red traffic light
            rospy.loginfo("Red Traffic light detected")
            lane.waypoints = self.decelerate_waypoints(base_waypoints, closest_idx)

        return lane

    def get_closest_waypoint_idx(self):
        x = self.pose.pose.position.x
        y = self.pose.pose.position.y
        closest_idx = self.waypoint_tree.query([x, y], 1)[1]

        # check if closed point is ahead or behind of vehicle using dot product and hyperplane logic
        closest_coord = self.waypoints_2d[closest_idx]
        prev_coord = self.waypoints_2d[closest_idx-1]
        cl_vec = np.array(closest_coord)
        prev_vec = np.array(prev_coord)
        pose_vec = np.array([x, y])

        val = np.dot(cl_vec - prev_vec, pose_vec - cl_vec) #+ve is same direction as cos(<90) is >0 and vice versa
        if val > 0:
            closest_idx = (closest_idx + 1) % len(self.waypoints_2d) # nearest pose is behind the current pose, please correct it
        return closest_idx


    def decelerate_waypoints(self, waypoints, closest_idx):
        temp = []
        for i, wp in enumerate(waypoints):
            p = Waypoint()
            p.pose = wp.pose

            stop_idx = max(self.stopline_wp_idx - closest_idx - 2, 0) #two waypoints back from line so front of car stops before the stop line
            dist = self.distance(waypoints, i, stop_idx)
            vel = math.sqrt(2*MAX_DECEL*dist)
            if vel < 1.:
                vel = 0.

            p.twist.twist.linear.x = min(vel, wp.twist.twist.linear.x)
            temp.append(p)

        return temp
    ''' 
        # Below function only used when waypoint_updatedr implemented only partially
        def publish_waypoints(self, closet_idx):
            lane = Lane()
            lane.header = self.base_waypoints.header
            lane.waypoints = self.base_waypoints.waypoints[closet_idx:closet_idx + LOOKAHEAD_WPS]
            #rospy.logwarn("\n ### {0} number of Lookahead waypoint to publish ### \n".format(LOOKAHEAD_WPS))
            self.final_waypoints_pub.publish(lane)
    '''
    def pose_cb(self, msg):
        # TODO: Implement
        self.pose = msg

    def waypoints_cb(self, waypoints):
        # TODO: Implement
        #self.base_waypoints = waypoints # only used in partial implementation
        self.base_lane = waypoints
        if not self.waypoints_2d: # to make sure that self.waypoints_2d is initialized before subscriber is initialized 
            self.waypoints_2d = [[waypoint.pose.pose.position.x, waypoint.pose.pose.position.y] for waypoint in waypoints.waypoints]
            self.waypoint_tree = KDTree(self.waypoints_2d)
            rospy.logwarn("A kdtree is made")

    def traffic_cb(self, msg):
        # TODO: Callback for /traffic_waypoint message. Implement
        self.stopline_wp_idx = msg.data

    def obstacle_cb(self, msg):
        # TODO: Callback for /obstacle_waypoint message. We will implement it later
        pass

    def get_waypoint_velocity(self, waypoint):
        return waypoint.twist.twist.linear.x

    def set_waypoint_velocity(self, waypoints, waypoint, velocity):
        waypoints[waypoint].twist.twist.linear.x = velocity

    def distance(self, waypoints, wp1, wp2):
        dist = 0
        dl = lambda a, b: math.sqrt((a.x-b.x)**2 + (a.y-b.y)**2  + (a.z-b.z)**2)
        for i in range(wp1, wp2+1):
            dist += dl(waypoints[wp1].pose.pose.position, waypoints[i].pose.pose.position)
            wp1 = i
        return dist


if __name__ == '__main__':
    try:
        WaypointUpdater()
    except rospy.ROSInterruptException:
        rospy.logerr('Could not start waypoint updater node.')
