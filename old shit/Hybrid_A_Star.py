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

def hybrid_astar_with_NHO(start, goal, valid_grid, cost_grid, truck:Firetruck, 
                          minor_grid_size=0.2, timeout=60.0, track_search=True):
    """
    Hybrid A* with Dubins heuristic
    
    Args:
        minor_grid_size: meters per grid cell (for coordinate conversion)
        timeout: max search time in seconds
        track_search: if True, return searched nodes for visualization
    """
    import heapq
    import time
    
    start_time = time.time()
    open_set = []
    closed_set = set()
    nodes_expanded = 0
    
    # Track all expanded nodes for visualization
    expanded_nodes = [] if track_search else None
    
    turning_radius = truck.MIN_TURN_RAD
    
    start.g = 0
    start.h = heuristic_NHO(start, goal, turning_radius)
    heapq.heappush(open_set, (start.g + start.h, id(start), start))
    
    print(f"Initial heuristic: {start.h:.2f}m")
    
    while open_set:
        # Timeout check
        if time.time() - start_time > timeout:
            print(f"✗ Timeout after {timeout}s ({nodes_expanded} nodes expanded)")
            if track_search:
                return None, expanded_nodes
            return None
        
        _, _, current = heapq.heappop(open_set)
        
        # Progress printing
        nodes_expanded += 1
        if nodes_expanded % 500 == 0:
            elapsed = time.time() - start_time
            goal_dist = np.sqrt((current.y - goal.y)**2 + (current.x - goal.x)**2)
            print(f"  Nodes: {nodes_expanded}, Open: {len(open_set)}, "
                  f"Dist to goal: {goal_dist:.1f}m, Time: {elapsed:.1f}s")
        
        # Goal check
        goal_dist = np.sqrt((current.y - goal.y)**2 + (current.x - goal.x)**2)
        goal_angle_diff = abs(current.theta - goal.theta)
        
        if goal_dist < 1.0 and goal_angle_diff < np.pi/6:
            print(f"✓ Goal reached! ({nodes_expanded} nodes expanded)")
            if track_search:
                return reconstruct_path(current), expanded_nodes
            return reconstruct_path(current)
        
        # Discretize for closed set
        disc_state = discretize_state(current.y, current.x, current.theta)
        if disc_state in closed_set:
            continue
        closed_set.add(disc_state)
        
        # Track this expanded node
        if track_search:
            expanded_nodes.append((current.y, current.x, current.theta, current.g, current.h))
        
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
            if not is_state_valid(new_state, valid_grid, truck, minor_grid_size):
                continue
            
            # Cost calculation
            distance = np.sqrt((new_state.y - current.y)**2 + 
                             (new_state.x - current.x)**2)
            
            # Convert meters to grid cells for cost lookup
            grid_y = int(round(new_state.y / minor_grid_size))
            grid_x = int(round(new_state.x / minor_grid_size))
            
            # Bounds check
            if (grid_y < 0 or grid_y >= cost_grid.shape[0] or
                grid_x < 0 or grid_x >= cost_grid.shape[1]):
                continue
            
            clearance_cost = cost_grid[grid_y, grid_x]
            
            edge_cost = distance * (1.0 + 5.0 * clearance_cost)
            
            new_state.g = current.g + edge_cost
            new_state.h = heuristic_NHO(new_state, goal, turning_radius)
            new_state.parent = current
            
            heapq.heappush(open_set, 
                          (new_state.g + new_state.h, id(new_state), new_state))
    
    print(f"✗ No path found ({nodes_expanded} nodes expanded)")
    if track_search:
        return None, expanded_nodes
    return None

def is_state_valid(state, valid_grid, truck, minor_grid_size=0.2):
    """
    Check if state is collision-free
    
    Args:
        state: State in meters
        valid_grid: binary grid (0=free, 1=obstacle)
        truck: Firetruck instance
        minor_grid_size: meters per grid cell
    """
    # Get vehicle corners at this state (returns meters)
    corners = truck.calc_boundary((state.y, state.x, state.theta), scale=1)
    
    # Check corners - corners are (x, y) in meters
    for corner in corners:
        corner_x_m, corner_y_m = corner
        
        # Convert meters to grid cells
        grid_x = int(round(corner_x_m / minor_grid_size))
        grid_y = int(round(corner_y_m / minor_grid_size))
        
        # Bounds check
        if (grid_x < 0 or grid_x >= valid_grid.shape[1] or
            grid_y < 0 or grid_y >= valid_grid.shape[0]):
            return False
        
        # Collision check
        if valid_grid[grid_y, grid_x] == 1:
            return False
    
    # Check edges (3 points per edge is enough)
    for i in range(4):
        c1_x_m, c1_y_m = corners[i]
        c2_x_m, c2_y_m = corners[(i + 1) % 4]
        
        for t in [0.33, 0.67]:  # Check 2 points per edge instead of 5
            edge_x_m = c1_x_m + t * (c2_x_m - c1_x_m)
            edge_y_m = c1_y_m + t * (c2_y_m - c1_y_m)
            
            grid_x = int(round(edge_x_m / minor_grid_size))
            grid_y = int(round(edge_y_m / minor_grid_size))
            
            if (grid_x < 0 or grid_x >= valid_grid.shape[1] or
                grid_y < 0 or grid_y >= valid_grid.shape[0]):
                return False
            
            if valid_grid[grid_y, grid_x] == 1:
                return False
    
    return True

def visualize_search_tree(start_state, goal_state, expanded_nodes, binary_grid, 
                          dist_cost_grid, minor_grid_size=0.2):
    """
    Visualize which nodes A* searched through
    
    Args:
        start_state: starting State
        goal_state: goal State
        expanded_nodes: list of (y, x, theta, g, h) tuples
        binary_grid: obstacle grid
        dist_cost_grid: cost grid
        minor_grid_size: meters per grid cell
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8))
    
    # Left plot: Search tree over cost grid
    ax1.imshow(dist_cost_grid, cmap='plasma', origin='lower', vmin=0, vmax=1, alpha=0.5)
    ax1.contour(binary_grid, levels=[0.5], colors='cyan', linewidths=1, alpha=0.5)
    
    # Plot expanded nodes
    if len(expanded_nodes) > 0:
        node_y = [n[0] / minor_grid_size for n in expanded_nodes]
        node_x = [n[1] / minor_grid_size for n in expanded_nodes]
        
        # Color by expansion order (blue=early, red=late)
        colors = np.arange(len(node_x))
        scatter = ax1.scatter(node_x, node_y, c=colors, s=10, cmap='coolwarm', 
                            alpha=0.6, zorder=5)
        plt.colorbar(scatter, ax=ax1, label='Expansion order')
    
    # Plot start and goal
    start_x = start_state.x / minor_grid_size
    start_y = start_state.y / minor_grid_size
    goal_x = goal_state.x / minor_grid_size
    goal_y = goal_state.y / minor_grid_size
    
    ax1.plot(start_x, start_y, 'go', markersize=12, label='Start', zorder=10)
    ax1.plot(goal_x, goal_y, 'r*', markersize=20, label='Goal', zorder=10)
    
    ax1.set_title(f'Search Tree ({len(expanded_nodes)} nodes expanded)')
    ax1.set_xlabel('X (grid cells)')
    ax1.set_ylabel('Y (grid cells)')
    ax1.legend()
    ax1.invert_yaxis()
    ax1.grid(True, alpha=0.3)
    
    # Right plot: Heatmap of search density
    if len(expanded_nodes) > 0:
        # Create density map
        density_grid = np.zeros_like(binary_grid, dtype=float)
        for y, x, theta, g, h in expanded_nodes:
            grid_y = int(round(y / minor_grid_size))
            grid_x = int(round(x / minor_grid_size))
            if 0 <= grid_y < density_grid.shape[0] and 0 <= grid_x < density_grid.shape[1]:
                density_grid[grid_y, grid_x] += 1
        
        im = ax2.imshow(density_grid, cmap='hot', origin='lower', interpolation='bilinear')
        ax2.contour(binary_grid, levels=[0.5], colors='cyan', linewidths=1, alpha=0.7)
        plt.colorbar(im, ax=ax2, label='Visit count')
        
        ax2.plot(start_x, start_y, 'go', markersize=12, label='Start')
        ax2.plot(goal_x, goal_y, 'r*', markersize=20, label='Goal')
        
        ax2.set_title('Search Density (where A* looked)')
        ax2.set_xlabel('X (grid cells)')
        ax2.set_ylabel('Y (grid cells)')
        ax2.legend()
        ax2.invert_yaxis()
    
    plt.tight_layout()
    plt.savefig('search_tree_visualization.png', dpi=150)
    print("Saved search_tree_visualization.png")
    # plt.show()

def visualize_search_progress(expanded_nodes, start_state, goal_state, 
                              binary_grid, minor_grid_size=0.2, num_frames=10):
    """
    Show how the search progressed over time (animated via multiple plots)
    
    Args:
        num_frames: number of snapshots to show
    """
    if len(expanded_nodes) == 0:
        print("No nodes to visualize")
        return
    
    fig, axes = plt.subplots(2, 5, figsize=(20, 8))
    axes = axes.flatten()
    
    # Show search at different stages
    stages = np.linspace(0, len(expanded_nodes), num_frames + 1, dtype=int)[1:]
    
    for idx, stage in enumerate(stages):
        ax = axes[idx]
        
        # Show obstacles
        ax.imshow(binary_grid, cmap='binary', origin='lower', alpha=0.3)
        
        # Show nodes expanded up to this stage
        nodes_so_far = expanded_nodes[:stage]
        if len(nodes_so_far) > 0:
            node_y = [n[0] / minor_grid_size for n in nodes_so_far]
            node_x = [n[1] / minor_grid_size for n in nodes_so_far]
            ax.scatter(node_x, node_y, c='blue', s=5, alpha=0.5)
        
        # Show start and goal
        ax.plot(start_state.x / minor_grid_size, start_state.y / minor_grid_size, 
               'go', markersize=8)
        ax.plot(goal_state.x / minor_grid_size, goal_state.y / minor_grid_size, 
               'r*', markersize=12)
        
        ax.set_title(f'After {stage} nodes')
        ax.invert_yaxis()
        ax.set_xticks([])
        ax.set_yticks([])
    
    plt.tight_layout()
    plt.savefig('search_progress.png', dpi=150)
    print("Saved search_progress.png")
    # plt.show()



class State:
    def __init__(self, y, x, theta):
        self.y = y
        self.x = x
        self.theta = theta
        self.g = 0
        self.h = 0
        self.parent = None

def discretize_state(y, x, theta, xy_resolution=1.0, theta_resolution=np.pi/6):
    """
    Discretize state for duplicate detection
    
    Args:
        xy_resolution: in METERS (1.0m = 5 grid cells at 0.2m/cell)
        theta_resolution: angular resolution (π/6 = 30 degrees)
    """
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