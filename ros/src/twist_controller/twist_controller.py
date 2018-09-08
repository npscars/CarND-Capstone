import curses

import rospy
from lowpass import LowPassFilter
from pid import PID
from yaw_controller import YawController


GAS_DENSITY = 2.858
ONE_MPH = 0.44704


class Controller(object):
    def __init__(self, *args, **kwargs):
        # create attributes
        self.vehicle_mass = None
        self.fuel_capacity = None
        self.brake_deadband = None
        self.decel_limit = None
        self.accel_limit = None
        self.wheel_radius = None
        self.wheel_base = None
        self.steer_ratio = None
        self.max_lat_accel = None
        self.max_steer_angle = None
        # fill in attributes from key-value parameters
        for attribute in kwargs.keys():
            setattr(self, attribute, kwargs[attribute])
        # create PID controller
        kp = 0.3
        ki = 0.1
        kd = 0.0
        mn = 0.0  # minimum throttle value
        mx = 0.2  # maximum throttle value
        self.throttle_controller = PID(kp, ki, kd, mn, mx)
        # create low pass filter
        tau = 0.5  # 1/(2*pi*tau) -> cutoff frequency
        ts = 0.02  # sample time
        self.vel_lpf = LowPassFilter(tau, ts)
        # create yaw controller
        self.yaw_controller = YawController(wheel_base=self.wheel_base,
                                            steer_ratio=self.steer_ratio,
                                            min_speed=0.1,
                                            max_lat_accel=self.max_lat_accel,
                                            max_steer_angle=self.max_steer_angle)

        self.last_time = rospy.get_time()
        self.last_velocity = 0.

    def control(self, current_velocity, dbw_enabled, linear_velocity, angular_velocity):
        # Return throttle, brake, steer
        if not dbw_enabled:
            self.throttle_controller.reset()
            return 0., 0., 0.

        current_velocity = self.vel_lpf.filt(current_velocity)

        steering = self.yaw_controller.get_steering(linear_velocity=linear_velocity,
                                                    angular_velocity=angular_velocity,
                                                    current_velocity=current_velocity)

        velocity_error = linear_velocity - current_velocity
        self.last_velocity = current_velocity
        #
        current_time = rospy.get_time()
        sample_time = current_time - self.last_time
        self.last_time = current_time

        throttle = self.throttle_controller.step(velocity_error, sample_time)

        brake = 0

        if linear_velocity == 0 and current_velocity < 0.1:
            throttle = 0
            brake = 700
        elif throttle < 0.1 and velocity_error < 0:
            throttle = 0
            decel = max(velocity_error, self.decel_limit)
            brake = abs(decel) * self.vehicle_mass * self.wheel_radius
        return throttle, brake, steering
