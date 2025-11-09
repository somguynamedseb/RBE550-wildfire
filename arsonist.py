import heapq
import numpy as np

class wumpi:
    def __init__(self,pos=[0,0]):
        self.y = pos[0]
        self.x = pos[1]
        self.y_tar = None
        self.x_tar = None
        self.path = []
        self.score = 0
        
    def update_path(self,target,obstacle_grid):
        self.y_tar = target[0]
        self.x_tar = target[1]
        self.path = astar_grid((self.y,self.x),(self.y_tar,self.x_tar),obstacle_grid) 
    
    def timestep(self,obstacle_grid,time):
        y,x = self.path[0]
        self.y = y
        self.x = x
        self.path = self.path[1:]
        burn_grid,burning_points = self.arson(obstacle_grid,time)
        return burn_grid,burning_points
        
        
    def arson(self, burn_grid, time, radius=40):
        """
        Mark cells within a radius as burning
        
        Args:
            burn_grid: grid to modify (3=burned out, 2=on fire, 1=obstacle, 0=free, -1=burned free)
            time: current time
            radius: radius in grid cells (1 = immediate neighbors, 2 = 2 cells away, etc.)
        
        Returns:
            burn_grid: modified grid
            burning_points: list of (y, x, time) tuples for newly burning cells
        """
        rows, cols = burn_grid.shape
        burning_points = []
        
        # Generate all points within radius
        for dy in range(-radius, radius + 1):
            for dx in range(-radius, radius + 1):
                # Skip center point (self)
                if dy == 0 and dx == 0:
                    continue
                
                # Optional: Use circular radius instead of square
                # if dy**2 + dx**2 > radius**2:
                #     continue
                
                y = self.y + dy
                x = self.x + dx
                
                # Bounds check
                if 0 <= y < rows and 0 <= x < cols:
                    if burn_grid[y][x] == 1:  # Not on fire or previously burned
                        burn_grid[y][x] = 2  # On fire
                        burning_points.append((y, x, time))
        # if len(burning_points)>0:
        #     print("AAAAAAAAAAAAAAAAAAAAAAAAAAH")
        self.score += len(burning_points)
        return burn_grid, burning_points
    def get_pos(self):
        return (self.y,self.x)
    
    def get_target(self):
        return (self.y_tar,self.x_tar)
    
import numpy as np
import heapq

def astar_grid(start, goal, obstacle_grid):
    """
    Simple A* on grid with obstacle avoidance
    
    Args:
        start: (y, x) tuple in grid coordinates
        goal: (y, x) tuple in grid coordinates
        obstacle_grid: 2D array where 1=obstacle, 0=free
    
    Returns:
        path: list of (y, x) grid coordinates, or None if no path
    """
    print(f"Running A* from {start} to {goal}")
    
    # Heuristic: Euclidean distance
    def heuristic(pos):
        return np.sqrt((pos[0] - goal[0])**2 + (pos[1] - goal[1])**2)
    
    # 8-connected grid (can move diagonally)
    neighbors_8 = [
        (-1, 0), (1, 0), (0, -1), (0, 1),      # 4-connected
        (-1, -1), (-1, 1), (1, -1), (1, 1)     # diagonals
    ]
    
    # Movement costs
    straight_cost = 1.0
    diagonal_cost = np.sqrt(2)
    
    open_set = []
    heapq.heappush(open_set, (0 + heuristic(start), start))
    
    came_from = {}
    g_score = {start: 0}
    
    nodes_expanded = 0
    
    while open_set:
        _, current = heapq.heappop(open_set)
        nodes_expanded += 1
        
        # Goal check
        if current == goal:
            print(f"✓ Path found! ({nodes_expanded} nodes expanded)")
            # Reconstruct path
            path = []
            while current in came_from:
                path.append(current)
                current = came_from[current]
            path.append(start)
            return path[::-1]
        
        # Expand neighbors
        for dy, dx in neighbors_8:
            neighbor = (current[0] + dy, current[1] + dx)
            
            # Bounds check
            if (neighbor[0] < 0 or neighbor[0] >= obstacle_grid.shape[0] or
                neighbor[1] < 0 or neighbor[1] >= obstacle_grid.shape[1]):
                continue
            
            # Obstacle check
            if obstacle_grid[neighbor[0], neighbor[1]] == 1:
                continue
            
            # Calculate cost
            if abs(dy) + abs(dx) == 2:  # Diagonal
                move_cost = diagonal_cost
            else:  # Straight
                move_cost = straight_cost
            
            tentative_g = g_score[current] + move_cost
            
            if neighbor not in g_score or tentative_g < g_score[neighbor]:
                came_from[neighbor] = current
                g_score[neighbor] = tentative_g
                f_score = tentative_g + heuristic(neighbor)
                heapq.heappush(open_set, (f_score, neighbor))
    
    print(f"No path found ({nodes_expanded} nodes expanded)")
    return None
