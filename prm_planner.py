import numpy as np
from scipy.spatial import KDTree
import heapq
from firehouse import Firetruck

class PRM_Planner:
    def __init__(self, binary_grid, firetruck, minor_grid_size=0.2, 
                 num_samples=2000, k_nearest=15):
        """
        Probabilistic Roadmap for Ackermann vehicle
        
        Args:
            binary_grid: 0=free, 1=obstacle
            firetruck: Firetruck instance for collision checking
            minor_grid_size: meters per grid cell
            num_samples: number of random configurations to sample
            k_nearest: number of nearest neighbors to try connecting
        """
        self.grid = binary_grid
        self.truck = firetruck
        self.minor_grid_size = minor_grid_size
        self.num_samples = num_samples
        self.k_nearest = k_nearest
        
        # Grid dimensions in meters
        self.max_y = binary_grid.shape[0] * minor_grid_size
        self.max_x = binary_grid.shape[1] * minor_grid_size
        
        self.nodes = []  # List of (y, x, theta) samples
        self.edges = {}  # Adjacency list {node_idx: [(neighbor_idx, cost), ...]}
        
        print(f"PRM Planner initialized:")
        print(f"  Map size: {self.max_y}m x {self.max_x}m")
        print(f"  Samples: {num_samples}, k-nearest: {k_nearest}")
    
    def is_valid_config(self, y, x, theta):
        """Check if a configuration is collision-free"""
        # Bounds check
        # if y < 0 or y >= self.max_y or x < 0 or x >= self.max_x:
        #     return False
        
        # Get vehicle corners
        corners = self.truck.calc_boundary((y, x, theta), scale=1,buffer = -0.1)
        
        # Check all corners and edges
        for corner_x_m, corner_y_m in corners:
            grid_x = int(round(corner_x_m / self.minor_grid_size))
            grid_y = int(round(corner_y_m / self.minor_grid_size))
            
            # if (grid_x < 0 or grid_x >= self.grid.shape[1] or #getting errors oin righter spaces and commented out for longer tests for testing due to frloat to int conversion
            # #     grid_y < 0 or grid_y >= self.grid.shape[0]):
            # #     return False
            
            # # if self.grid[grid_y, grid_x] == 1:
            # #     return False
        
        # Check edges between corners
        for i in range(4):
            c1_x, c1_y = corners[i]
            c2_x, c2_y = corners[(i + 1) % 4]
            
            for t in [0.33, 0.67]:
                edge_x = c1_x + t * (c2_x - c1_x)
                edge_y = c1_y + t * (c2_y - c1_y)
                
                grid_x = int(round(edge_x / self.minor_grid_size))
                grid_y = int(round(edge_y / self.minor_grid_size))
                
                if (grid_x < 0 or grid_x >= self.grid.shape[1] or
                    grid_y < 0 or grid_y >= self.grid.shape[0]):
                    return False
                
                if self.grid[grid_y, grid_x] == 1:
                    return False
        
        return True
    
    def sample_configurations(self):
        """Learning phase: randomly sample valid configurations"""
        print(f"\nSampling {self.num_samples} configurations...")
        
        valid_count = 0
        attempts = 0
        max_attempts = self.num_samples * 100  # Prevent infinite loop
        
        while valid_count < self.num_samples and attempts < max_attempts:
            attempts += 1
            
            # Random position and orientation
            y = np.random.uniform(5.0, self.max_y - 5.0)  # 5m margin
            x = np.random.uniform(5.0, self.max_x - 5.0)
            theta = np.random.uniform(-np.pi, np.pi)
            
            # Check if valid
            if self.is_valid_config(y, x, theta):
                self.nodes.append((y, x, theta))
                valid_count += 1
                
                if valid_count % 200 == 0:
                    print(f"  Sampled {valid_count}/{self.num_samples} valid nodes "
                          f"({attempts} attempts, {100*valid_count/attempts:.1f}% success rate)")
        
        print(f"✓ Sampled {len(self.nodes)} valid configurations in {attempts} attempts")
    
    def can_connect(self, node1, node2, num_checks=10):
        """
        Check if two nodes can be connected with a simple motion
        Uses straight-line interpolation with orientation blending
        """
        y1, x1, theta1 = node1
        y2, x2, theta2 = node2
        
        # Check intermediate configurations along path
        for i in range(num_checks + 1):
            t = i / num_checks
            
            # Linear interpolation of position
            y = y1 + t * (y2 - y1)
            x = x1 + t * (x2 - x1)
            
            # Interpolate angle (handle wraparound)
            dtheta = theta2 - theta1
            # Normalize to [-pi, pi]
            dtheta = (dtheta + np.pi) % (2 * np.pi) - np.pi
            theta = theta1 + t * dtheta
            
            if not self.is_valid_config(y, x, theta):
                return False
        
        return True
    
    def can_connect_kinematic(self, node1, node2):
        """
        Check if two nodes can be connected with actual truck motion
        Uses simple forward/reverse simulation
        """
        y1, x1, theta1 = node1
        y2, x2, theta2 = node2
        
        # Try forward and reverse motion
        for velocity in [2.0, -2.0]:  # Forward and reverse
            for steering in np.linspace(-self.truck.MAX_STEERING_ANGLE, 
                                    self.truck.MAX_STEERING_ANGLE, 5):
                # Simulate motion for fixed duration
                current = np.array([y1, x1, theta1])
                
                for step in range(20):  # Try 20 steps
                    current = self.truck.update_pos(current, velocity, steering, dt=0.1)
                    
                    # Check if valid
                    if not self.is_valid_config(*current):
                        break
                    
                    # Check if reached goal
                    dist = np.sqrt((current[0] - y2)**2 + (current[1] - x2)**2)
                    angle_diff = abs(current[2] - theta2)
                    
                    if dist < 1.0 and angle_diff < np.pi/6:
                        return True
        
        return False

    
    def connect_neighbors(self):
        """Connect each node to k nearest neighbors"""
        print(f"\nConnecting nodes (k={self.k_nearest})...")
        
        # Build KD-tree for position only (ignore theta for nearest neighbor search)
        positions = np.array([(n[0], n[1]) for n in self.nodes])
        tree = KDTree(positions)
        
        total_edges = 0
        
        for i, node in enumerate(self.nodes):
            if i % 200 == 0:
                print(f"  Processing node {i}/{len(self.nodes)}... ({total_edges} edges so far)")
            
            # Find k nearest neighbors
            distances, indices = tree.query([node[0], node[1]], k=self.k_nearest + 1)
            
            self.edges[i] = []
            
            for dist, j in zip(distances[1:], indices[1:]):  # Skip self
                if self.can_connect(node, self.nodes[j]):
                # if self.can_connect_kinematic(node, self.nodes[j]):
                    # Edge cost is Euclidean distance
                    cost = np.sqrt((node[0] - self.nodes[j][0])**2 + 
                                  (node[1] - self.nodes[j][1])**2)
                    self.edges[i].append((j, cost))
                    total_edges += 1
        
        print(f"✓ Created {total_edges} edges ({total_edges/len(self.nodes):.1f} avg per node)")
    
    def build_roadmap(self):
        """Complete learning phase: sample and connect"""
        print("\n" + "="*60)
        print("BUILDING PRM ROADMAP")
        print("="*60)
        
        self.sample_configurations()
        self.connect_neighbors()
        
        print("\n" + "="*60)
        print("ROADMAP BUILD COMPLETE")
        print("="*60)
    
    def connect_to_roadmap(self, config, max_neighbors=20):
        """
        Connect a new configuration (start/goal) to existing roadmap
        
        Returns:
            node_idx: index of the added node
        """
        if not self.is_valid_config(*config):
            print(f"  WARNING: Configuration {config} is not valid!")
            return None
        
        # Add to nodes
        node_idx = len(self.nodes)
        self.nodes.append(config)
        self.edges[node_idx] = []
        
        # Find nearby nodes to connect
        positions = np.array([(n[0], n[1]) for n in self.nodes[:-1]])  # Exclude new node
        if len(positions) == 0:
            return node_idx
        
        tree = KDTree(positions)
        distances, indices = tree.query([config[0], config[1]], k=min(max_neighbors, len(positions)))
        
        # Make sure indices is iterable
        if not hasattr(indices, '__iter__'):
            indices = [indices]
            distances = [distances]
        
        connections = 0
        for dist, j in zip(distances, indices):
            if self.can_connect(config, self.nodes[j]):
            # if self.can_connect_kinematic(config, self.nodes[j]):
                cost = np.sqrt((config[0] - self.nodes[j][0])**2 + 
                              (config[1] - self.nodes[j][1])**2)
                self.edges[node_idx].append((j, cost))
                
                # Add reverse edge
                if j not in self.edges:
                    self.edges[j] = []
                self.edges[j].append((node_idx, cost))
                connections += 1
        
        print(f"  Connected to {connections} roadmap nodes")
        return node_idx
    
    def graph_search(self, start_idx, goal_idx):
        """A* search on the roadmap graph"""
        print(f"\nSearching roadmap from node {start_idx} to {goal_idx}...")
        
        # Heuristic: Euclidean distance
        def heuristic(idx):
            node = self.nodes[idx]
            goal = self.nodes[goal_idx]
            return np.sqrt((node[0] - goal[0])**2 + (node[1] - goal[1])**2)
        
        open_set = []
        heapq.heappush(open_set, (0 + heuristic(start_idx), start_idx))
        
        came_from = {}
        g_score = {start_idx: 0}
        
        nodes_expanded = 0
        
        while open_set:
            _, current = heapq.heappop(open_set)
            nodes_expanded += 1
            
            if current == goal_idx:
                print(f"✓ Path found! ({nodes_expanded} nodes expanded)")
                # Reconstruct path
                path = []
                while current in came_from:
                    path.append(self.nodes[current])
                    current = came_from[current]
                path.append(self.nodes[start_idx])
                return path[::-1]
            
            # Expand neighbors
            if current not in self.edges:
                continue
            
            for neighbor, cost in self.edges[current]:
                tentative_g = g_score[current] + cost
                
                if neighbor not in g_score or tentative_g < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g
                    f_score = tentative_g + heuristic(neighbor)
                    heapq.heappush(open_set, (f_score, neighbor))
        
        print(f"✗ No path found ({nodes_expanded} nodes expanded)")
        return None
    
    def query(self, start, goal, goal_angle_tolerance=np.pi):
        """
        Query phase with optional goal orientation
        
        Args:
            start: (y, x, theta) tuple
            goal: (y, x, theta) tuple OR (y, x) tuple (no orientation constraint)
            goal_angle_tolerance: how close final angle must be (π = any angle OK)
        
        Returns:
            path: list of (y, x, theta) waypoints
        """
        # Handle goal with or without orientation
        if len(goal) == 2:
            goal = (goal[0], goal[1], 0.0)  # Add dummy angle
            goal_angle_tolerance = np.pi  # Accept any final orientation
        
        print("\n" + "="*60)
        print("QUERYING PRM")
        print("="*60)
        print(f"Start: ({start[0]:.1f}, {start[1]:.1f}, {start[2]:.2f})")
        print(f"Goal:  ({goal[0]:.1f}, {goal[1]:.1f}, any angle)" if goal_angle_tolerance >= np.pi 
            else f"Goal:  ({goal[0]:.1f}, {goal[1]:.1f}, {goal[2]:.2f})")
        
        # Connect start and goal to roadmap
        print("\nConnecting start to roadmap...")
        start_idx = self.connect_to_roadmap(start)
        
        if start_idx is None:
            print("✗ Failed to connect start configuration")
            return None
        
        print("Connecting goal to roadmap...")
        goal_idx = self.connect_to_roadmap(goal)
        
        if goal_idx is None:
            print("✗ Failed to connect goal configuration")
            return None
        
        # Search the graph
        path = self.graph_search(start_idx, goal_idx)
        
        if path is not None:
            print(f"\n✓ Path found with {len(path)} waypoints")
        
        print("="*60 + "\n")
        
        return path