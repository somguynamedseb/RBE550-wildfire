import heapq
import numpy as np

class wumpus:
    def __init__(self,pos=[0,0],target=[0,0],obstacle_grid=[]):
        self.y = pos[0]
        self.x = pos[1]
        self.tar_y = target[0]
        self.tar_x = target[1]
        self.obstacle_grid = obstacle_grid
        self.path = astar_grid((self.y,self.x),(self.tar_y,self.tar_x),self.obstacle_grid)
        
    def update_path(self):
        self.path = astar_grid((self.y,self.x),(self.tar_y,self.tar_x),self.obstacle_grid) 
    
    def timestep(self,burn_grid):
        y,x = self.path[0]
        self.y = y
        self.x = x
        path = path[1:]
        self.arson(burn_grid)
        
    def arson(self,burn_grid):
        #1 = free square | 0 = obstacle | -1 = obstacle on fire
        dirs = [(-1,0),(-1,1),(-1,-1),(1,0),(1,1),(1,-1),(0,-1),(0,1)]
        rows, cols = burn_grid.shape
        for dir in dirs:
            dy,dx = dir
            y = self.y+dy
            x = self.x+dx
            if 0 < y < rows-1 and 0 < x < cols-1:
                if burn_grid[y][x] == 0:
                    burn_grid[y][x] = -1 #on fire
                    
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
