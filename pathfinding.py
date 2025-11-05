import random

import numpy as np
from scipy.ndimage import distance_transform_edt, binary_dilation
import  matplotlib.pyplot as plt
from firehouse import Firetruck

class DubinsPath:
    def __init__(self, turning_radius):
        self.rho = turning_radius
    
    def mod2pi(self, theta):
        """Normalize angle to [0, 2π)"""
        return theta - 2.0 * np.pi * np.floor(theta / (2.0 * np.pi))
    
    def LSL(self, alpha, beta, d):
        """Left-Straight-Left path"""
        sa = np.sin(alpha)
        ca = np.cos(alpha)
        sb = np.sin(beta)
        cb = np.cos(beta)
        
        tmp = 2.0 + d**2 - 2.0 * (ca * cb + sa * sb - d * (sa - sb))
        
        if tmp < 0:
            return None
        
        t = self.mod2pi(-alpha + np.arctan2(cb - ca, d + sa - sb))
        p = np.sqrt(max(tmp, 0))
        q = self.mod2pi(beta - np.arctan2(cb - ca, d + sa - sb))
        
        return t, p, q
    
    def RSR(self, alpha, beta, d):
        """Right-Straight-Right path"""
        sa = np.sin(alpha)
        ca = np.cos(alpha)
        sb = np.sin(beta)
        cb = np.cos(beta)
        
        tmp = 2.0 + d**2 - 2.0 * (ca * cb + sa * sb - d * (sb - sa))
        
        if tmp < 0:
            return None
        
        t = self.mod2pi(alpha - np.arctan2(ca - cb, d - sa + sb))
        p = np.sqrt(max(tmp, 0))
        q = self.mod2pi(-beta + np.arctan2(ca - cb, d - sa + sb))
        
        return t, p, q
    
    def LSR(self, alpha, beta, d):
        """Left-Straight-Right path"""
        sa = np.sin(alpha)
        ca = np.cos(alpha)
        sb = np.sin(beta)
        cb = np.cos(beta)
        
        tmp = -2.0 + d**2 + 2.0 * (ca * cb + sa * sb + d * (sa + sb))
        
        if tmp < 0:
            return None
        
        p = np.sqrt(max(tmp, 0))
        theta = np.arctan2(-ca - cb, d + sa + sb) - np.arctan2(-2.0, p)
        t = self.mod2pi(-alpha + theta)
        q = self.mod2pi(-beta + theta)
        
        return t, p, q
    
    def RSL(self, alpha, beta, d):
        """Right-Straight-Left path"""
        sa = np.sin(alpha)
        ca = np.cos(alpha)
        sb = np.sin(beta)
        cb = np.cos(beta)
        
        tmp = -2.0 + d**2 + 2.0 * (ca * cb + sa * sb - d * (sa + sb))
        
        if tmp < 0:
            return None
        
        p = np.sqrt(max(tmp, 0))
        theta = np.arctan2(ca + cb, d - sa - sb) - np.arctan2(2.0, p)
        t = self.mod2pi(alpha - theta)
        q = self.mod2pi(beta - theta)
        
        return t, p, q
    
    def RLR(self, alpha, beta, d):
        """Right-Left-Right path"""
        sa = np.sin(alpha)
        ca = np.cos(alpha)
        sb = np.sin(beta)
        cb = np.cos(beta)
        
        tmp = 0.125 * (6.0 - d**2 + 2.0 * (ca * cb + sa * sb + d * (sa - sb)))
        
        if abs(tmp) > 1.0:
            return None
        
        p = self.mod2pi(2.0 * np.pi - np.arccos(tmp))
        t = self.mod2pi(alpha - np.arctan2(ca - cb, d - sa + sb) + p / 2.0)
        q = self.mod2pi(alpha - beta - t + p)
        
        return t, p, q
    
    def LRL(self, alpha, beta, d):
        """Left-Right-Left path"""
        sa = np.sin(alpha)
        ca = np.cos(alpha)
        sb = np.sin(beta)
        cb = np.cos(beta)
        
        tmp = 0.125 * (6.0 - d**2 + 2.0 * (ca * cb + sa * sb - d * (sa - sb)))
        
        if abs(tmp) > 1.0:
            return None
        
        p = self.mod2pi(2.0 * np.pi - np.arccos(tmp))
        t = self.mod2pi(-alpha + np.arctan2(cb - ca, d + sa - sb) + p / 2.0)
        q = self.mod2pi(beta - alpha - t + p)
        
        return t, p, q
    
    def plan(self, start, goal):
        """
        Find shortest Dubins path
        
        Args:
            start: (x, y, theta)
            goal: (x, y, theta)
        
        Returns:
            length: shortest path length in meters
        """
        x1, y1, theta1 = start
        x2, y2, theta2 = goal
        
        # Transform to canonical form
        dx = x2 - x1
        dy = y2 - y1
        d = np.sqrt(dx**2 + dy**2) / self.rho
        theta = self.mod2pi(np.arctan2(dy, dx))
        alpha = self.mod2pi(theta1 - theta)
        beta = self.mod2pi(theta2 - theta)
        
        # Try all 6 path types
        planners = [self.LSL, self.RSR, self.LSR, self.RSL, self.RLR, self.LRL]
        best_length = float('inf')
        
        for planner in planners:
            result = planner(alpha, beta, d)
            if result is not None:
                t, p, q = result
                length = (t + p + q) * self.rho
                if length < best_length:
                    best_length = length
        
        return best_length if best_length != float('inf') else None

def dubins_heuristic(start_state, goal_state, turning_radius):
    """
    Non-holonomic heuristic using Dubins paths
    
    Args:
        start_state: (y, x, theta) - YOUR CONVENTION
        goal_state: (y, x, theta)
        turning_radius: minimum turning radius
    
    Returns:
        Shortest path length (admissible heuristic)
    """
    y_s, x_s, theta_s = start_state
    y_g, x_g, theta_g = goal_state
    
    dubins = DubinsPath(turning_radius)
    length = dubins.plan(
        (x_s, y_s, theta_s),
        (x_g, y_g, theta_g)
    )
    
    return length if length is not None else np.sqrt((x_g - x_s)**2 + (y_g - y_s)**2)

def heuristic_NHO(current_state, goal_state, turning_radius):
    """
    Non-holonomic-without-obstacles heuristic
    This is ADMISSIBLE (never overestimates)
    """
    return dubins_heuristic(
        (current_state.y, current_state.x, current_state.theta),
        (goal_state.y, goal_state.x, goal_state.theta),
        turning_radius
    )

def hybrid_astar_with_NHO(start, goal, valid_grid, cost_grid, truck:Firetruck):
    """
    Hybrid A* with Dubins heuristic
    """
    import heapq
    
    open_set = []
    closed_set = set()
    
    turning_radius = truck.MIN_TURN_RAD
    
    start.g = 0
    start.h = heuristic_NHO(start, goal, turning_radius)
    heapq.heappush(open_set, (start.g + start.h, id(start), start))
    
    while open_set:
        _, _, current = heapq.heappop(open_set)
        
        # Goal check
        goal_dist = np.sqrt((current.y - goal.y)**2 + (current.x - goal.x)**2)
        goal_angle_diff = abs(current.theta - goal.theta)
        
        if goal_dist < 1.0 and goal_angle_diff < np.pi/6:  # 30 degrees
            return reconstruct_path(current)
        
        # Discretize for closed set
        disc_state = discretize_state(current.y, current.x, current.theta)
        if disc_state in closed_set:
            continue
        closed_set.add(disc_state)
        
        # Expand using motion primitives
        primitives = truck.generate_motion_primitives_for_firetruck()
        
        for velocity, steering, duration in primitives:
            # Simulate motion
            new_pos = truck.update_pos(
                (current.y, current.x, current.theta),
                velocity,
                steering,
                duration
            )
            
            new_state = State(new_pos[0], new_pos[1], new_pos[2])
            
            # Collision check
            if not is_state_valid(new_state, valid_grid, truck):
                continue
            
            # Cost calculation
            distance = np.sqrt((new_state.y - current.y)**2 + 
                                (new_state.x - current.x)**2)
            
            # Sample cost from cost_grid
            grid_y = int(round(new_state.y))
            grid_x = int(round(new_state.x))
            clearance_cost = cost_grid[grid_y, grid_x]
            
            edge_cost = distance * (1.0 + 10.0 * clearance_cost)
            
            new_state.g = current.g + edge_cost
            new_state.h = heuristic_NHO(new_state, goal, turning_radius)
            new_state.parent = current
            
            heapq.heappush(open_set, 
                            (new_state.g + new_state.h, id(new_state), new_state))
    
    return None  # No path found

def is_state_valid(state, valid_grid, truck):
    """
    Check if state is collision-free
    """
    # Get vehicle corners at this state
    corners = truck.calc_boundary((state.y, state.x, state.theta), scale=1)
    
    # Check all corners and interpolate along edges
    for corner in corners:
        grid_x = int(round(corner[0]))
        grid_y = int(round(corner[1]))
        
        # Bounds check
        if (grid_x < 0 or grid_x >= valid_grid.shape[1] or
            grid_y < 0 or grid_y >= valid_grid.shape[0]):
            return False
        
        # Collision check (valid_grid: 0=free, 1=obstacle)
        if valid_grid[grid_y, grid_x] == 1:
            return False
    
    return True

class State:
    def __init__(self, y, x, theta):
        self.y = y
        self.x = x
        self.theta = theta
        self.g = 0
        self.h = 0
        self.parent = None

def discretize_state(y, x, theta, xy_resolution=0.5, theta_resolution=np.pi/8):
    """Discretize state for duplicate detection"""
    disc_y = int(round(y / xy_resolution))
    disc_x = int(round(x / xy_resolution))
    disc_theta = int(round(theta / theta_resolution)) % int(2*np.pi/theta_resolution)
    return (disc_y, disc_x, disc_theta)

def reconstruct_path(state):
    """Backtrack from goal to start"""
    path = []
    while state:
        path.append((state.y, state.x, state.theta))
        state = state.parent
    return path[::-1]

