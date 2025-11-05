import numpy as np
import numpy as np
from numpy import pi,cos,sin
from collections import deque
import heapq
import time as t
from scipy.ndimage import binary_dilation
import copy
from shapely.geometry import Polygon, Point
from shapely.ops import unary_union
from shapely import contains_xy
import numpy as np
import heapq
from scipy.spatial import cKDTree
from matplotlib.path import Path

class Firetruck:
    def __init__(self, pos: tuple[float, float], angle: float, scale: int):
        # Units in meters and seconds
        self.WIDTH = 1.8
        self.LENGTH = 5.2
        self.WHEELBASE = 2.8
        
        self.y = pos[0]
        self.x = pos[1]
        self.ang = angle
        self.scale = scale
        
        # Physical limits
        self.MAX_ACC = 4.0  # m/s^2
        self.MAX_VEL = 10.0  # m/s
        self.MAX_STEERING_ANGLE = np.pi / 4  # 45 degrees (typical for vehicles)
        self.MIN_TURN_RAD = 13.0
        
        # Current state
        self.velocity = 0.0  # forward velocity (m/s)
        self.steering_angle = 0.0  # steering angle (radians)
        self.acceleration = 0.0  # m/s^2
        
        # Geometry offsets
        self._wheelbase_offset = (self.LENGTH - self.WHEELBASE) / 2
        self.boundary_offsets = [
            (self.WHEELBASE + self._wheelbase_offset, self.WIDTH/2),   # FR
            (self.WHEELBASE + self._wheelbase_offset, -self.WIDTH/2),  # FL
            (-self._wheelbase_offset, self.WIDTH/2),                   # RR
            (-self._wheelbase_offset, -self.WIDTH/2)                   # RL
        ]
        self.wheel_points = [
            (self.WHEELBASE, self.WIDTH/2),   # FR
            (self.WHEELBASE, -self.WIDTH/2),  # FL
            (0, self.WIDTH/2),                # RR
            (0, -self.WIDTH/2)                # RL
        ]

    def calc_boundary(self, pos, buffer: float = 0.0, scale=1):
        """Calculate the 4 corners of the vehicle at given position"""
        y, x, ang = pos
        y = y * scale
        x = x * scale
        
        corners = []
        for x_off, y_off in self.boundary_offsets:
            x_off *= (1 + buffer) * scale
            y_off *= (1 + buffer) * scale
            corner_x = x + (x_off * np.cos(ang)) - (y_off * np.sin(ang))
            corner_y = y + (x_off * np.sin(ang)) + (y_off * np.cos(ang))
            corners.append((corner_x, corner_y))
        
        return corners  # [FR, FL, RR, RL]

    def calc_wheel_pts(self, pos, buffer: float = 0.0, scale=1):
        """Calculate wheel positions at given pose"""
        y, x, ang = pos
        y = y * scale
        x = x * scale
        
        wheels = []
        for x_off, y_off in self.wheel_points:
            x_off *= (1 + buffer) * scale
            y_off *= (1 + buffer) * scale
            wheel_x = x + (x_off * np.cos(ang)) - (y_off * np.sin(ang))
            wheel_y = y + (x_off * np.sin(ang)) + (y_off * np.cos(ang))
            wheels.append((wheel_x, wheel_y))
        
        return wheels  # [FR, FL, RR, RL]

    def get_current_pos(self):
        """Return current pose (y, x, angle)"""
        return (self.y, self.x, self.ang)

    def wrap_to_pi(self, angle):
        """Wrap angle to [-pi, pi)"""
        return (angle + np.pi) % (2.0 * np.pi) - np.pi

    def update_pos(self, pos, drive_vel, steering_angle, dt=0.05):
        """
        Update vehicle pose using kinematic bicycle (Ackermann) model.
        Reference point: midpoint between rear wheels.

        Args:
            pos: (y, x, theta) current pose
            drive_vel: linear velocity at rear axle (m/s)
            steering_angle: front wheel steering angle (radians)
            dt: timestep (s)

        Returns:
            np.array([y_new, x_new, theta_new])
        """
        y, x, theta = float(pos[0]), float(pos[1]), float(pos[2])
        L = float(self.WHEELBASE)

        eps = 1e-8

        if abs(steering_angle) < eps:
            # Straight motion
            dx = drive_vel * np.cos(theta) * dt
            dy = drive_vel * np.sin(theta) * dt
            x_new = x + dx
            y_new = y + dy
            theta_new = theta
        else:
            # Turning motion (ICC-based integration)
            R = L / np.tan(steering_angle)  # turning radius
            omega = drive_vel / R           # yaw rate
            dtheta = omega * dt

            # ICC update
            x_new = x + R * (np.sin(theta + dtheta) - np.sin(theta))
            y_new = y - R * (np.cos(theta + dtheta) - np.cos(theta))
            theta_new = theta + dtheta

        theta_new = self.wrap_to_pi(theta_new)
        return np.array([y_new, x_new, theta_new])

    def set_control(self, velocity, steering_angle):
        """
        Set control inputs (for path planning / execution)
        
        Args:
            velocity: desired forward velocity (m/s)
            steering_angle: desired steering angle (radians)
        """
        # Clamp to physical limits
        self.velocity = np.clip(velocity, -self.MAX_VEL, self.MAX_VEL)
        self.steering_angle = np.clip(steering_angle, 
                                     -self.MAX_STEERING_ANGLE, 
                                     self.MAX_STEERING_ANGLE)

    def timestep(self, dt=0.05):
        """
        Advance simulation by one timestep using current control inputs
        
        Args:
            dt: timestep duration (s)
        """
        # Update position based on current velocity and steering
        new_pos = self.update_pos(
            (self.y, self.x, self.ang), 
            self.velocity, 
            self.steering_angle, 
            dt
        )
        
        self.y = new_pos[0]
        self.x = new_pos[1]
        self.ang = new_pos[2]

    def get_turning_radius(self):
        """Get current turning radius (inf if straight)"""
        if abs(self.steering_angle) < 1e-8:
            return np.inf
        return self.WHEELBASE / np.tan(self.steering_angle)
    
    def generate_motion_primitives_for_firetruck(truck, step_distance=1.0):
        """
        Generate motion primitives for your firetruck
        
        Returns: list of (velocity, steering_angle, duration)
        """
        primitives = []
        
        # Steering angles to sample
        steering_angles = np.linspace(
            -truck.MAX_STEERING_ANGLE, 
            truck.MAX_STEERING_ANGLE, 
            5
        )
        
        velocity = 2.0  # constant forward speed for planning
        
        for steering in steering_angles:
            # Calculate how long to drive to cover step_distance
            if abs(steering) < 1e-6:
                duration = step_distance / velocity
            else:
                R = truck.WHEELBASE / np.tan(steering)
                arc_angle = step_distance / abs(R)
                duration = arc_angle * abs(R) / velocity
            
            primitives.append((velocity, steering, duration))
        
        return primitives
    
    def get_min_turning_radius(self):
        """
        Calculate minimum turning radius for firetruck
        """
        return self.WHEELBASE / np.tan(self.MAX_STEERING_ANGLE)