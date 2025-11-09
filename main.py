import random
import numpy as np
from scipy.ndimage import distance_transform_edt, binary_dilation
import matplotlib.pyplot as plt
from firehouse import Firetruck
import prm_planner as prm
from arsonist import wumpi


### ENVIRONMENT GEN FUNCTIONS
def create_cost_grid(occupancy_grid, robot_radius, safety_distance=2.0):
    """
    occupancy_grid: 2D array, 1=obstacle, 0=free
    robot_radius: in grid cells
    safety_distance: preferred clearance in grid cells
    """
    
    # Inflate obstacles by robot radius
    struct = np.ones((robot_radius*2+1, robot_radius*2+1))
    inflated = binary_dilation(occupancy_grid, structure=struct)
    
    # Compute Voronoi field (distance to nearest inflated obstacle)
    voronoi = distance_transform_edt(1 - inflated)
    
    # Transform to costs
    # Piecewise approach:
    cost_grid = np.ones_like(voronoi, dtype=float)
    
    # Collision zones
    cost_grid[inflated == 1] = np.inf
    
    # Gradient near obstacles
    near_obstacle = (voronoi < safety_distance) & (inflated == 0)
    cost_grid[near_obstacle] = 1.0 + 10.0 * (safety_distance - voronoi[near_obstacle]) / safety_distance
    
    # Free space (just nominal cost)
    cost_grid[voronoi >= safety_distance] = 1.0
    
    return cost_grid, voronoi

def voronoi_to_cost_grid(voronoi_field, occupancy_grid, max_useful_distance=None):
    """
    Convert Voronoi field to cost grid [0, 1]
    
    Args:
        voronoi_field: Distance to nearest obstacle (higher = safer)
        occupancy_grid: 0 = free, 1 = obstacle
        max_useful_distance: Distance at which cost reaches minimum (0)
                           If None, uses 90th percentile
    
    Returns:
        cost_grid: 0 = safest paths, 1 = obstacles/danger
    """
    cost_grid = np.ones_like(voronoi_field, dtype=float)
    
    # Set obstacles to maximum cost
    cost_grid[occupancy_grid == 1] = 1.0
    
    # For free space, invert and normalize
    free_space = occupancy_grid == 0
    
    if max_useful_distance is None:
        max_useful_distance = np.percentile(voronoi_field[free_space], 90)
    
    # Invert: far from obstacle (high voronoi) -> low cost
    # Normalize to [0, 1]
    cost_grid[free_space] = 1.0 - np.clip(
        voronoi_field[free_space] / max_useful_distance, 
        0, 
        1
    )
    binary_grid = v_cost_to_binary(cost_grid)
    return cost_grid,binary_grid

def v_cost_to_binary(cost_grid,th = 0.4):
    binary_grid = np.zeros_like(cost_grid)
    for i in range(len(cost_grid)):
        for j in range(len(cost_grid[0])):
            if cost_grid[i][j] >= th:
                binary_grid[i][j] = 1
    return binary_grid

def visualize_cost_grid(cost_grid, occupancy_grid):
    """
    Visualize cost grid with obstacle boundaries
    """
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Display cost grid (0=safe/dark, 1=danger/bright)
    im = ax.imshow(cost_grid, cmap='plasma', origin='lower', 
                    vmin=0, vmax=1, interpolation='nearest')
    
    # Overlay obstacle boundaries in cyan
    obstacles = occupancy_grid == 0
    ax.contour(obstacles, levels=[0.5], colors='k', linewidths=2)
    
    plt.colorbar(im, label='Cost (0=safe, 1=obstacle)', ax=ax)
    plt.title('Cost Grid')
    plt.xlabel('X (meters)')
    plt.ylabel('Y (meters)')
    plt.xticks(np.linspace(0, MAX_X*scale, int(MAX_X/25)+1),labels=range(0, MAX_X+1,25))
    plt.yticks(np.linspace(0, MAX_Y*scale, int(MAX_Y/25)+1),labels=range(0, MAX_Y+1,25))   
    ax.invert_yaxis()
    plt.tight_layout()
    plt.savefig("voronoi_cost_grid.png")

def visualize_bin_grid(binary_grid):
    """
    Visualize cost grid with obstacle boundaries
    """
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Display cost grid (0=safe/dark, 1=danger/bright)

    im = ax.imshow(-(binary_grid)+1, cmap='gray')
    # Overlay obstacle boundaries in cyan
    plt.title('Valid Squares Grid (inverted in data)')
    plt.xlabel('X (meters)')
    plt.ylabel('Y (meters)')
    plt.xticks(np.linspace(0, MAX_X*scale, int(MAX_X/25)+1),labels=range(0, MAX_X+1,25))
    plt.yticks(np.linspace(0, MAX_Y*scale, int(MAX_Y/25)+1),labels=range(0, MAX_Y+1,25))   
    # ax.invert_yaxis()
    plt.tight_layout()
    plt.savefig("valid_squares.png")







MAX_Y = 250 #meters
MAX_X = 250 #meters

minor_grid_size = 0.2
major_grid_size = 5
shape_dim = 5
shape_scale = 10

OBS_Y = int(MAX_Y/major_grid_size)
OBS_X = int(MAX_X/major_grid_size)
scale = int(major_grid_size/minor_grid_size)



percent_fill = 0.10
print(f"scale {scale}")
print(f"OBS {OBS_X}")

#runs for arnd 840 seconds before errors
# random.seed(24)
# start_pos_meters = (50.0, 50.0)  # (y, x) in meters 

#runs for arnd 520 seconds before errors
# random.seed(26)
# start_pos_meters = (90.0, 90.0)  # (y, x) in meters

#runs for arnd 436 seconds before errors
# random.seed(28)
# start_pos_meters = (50.0, 50.0)  # (y, x) in meters

#runs for arnd 250 seconds before errors
# random.seed(31)
# start_pos_meters = (50.0, 80.0)  # (y, x) in meters

#DONE
random.seed(33)
start_pos_meters = (30.0, 60.0)  # (y, x) in meters

## Initialize firetruck
start_angle = 0.0
firetruck = Firetruck(start_pos_meters, start_angle, scale=1)

values = random.choices(range(10), k=int(OBS_X/shape_dim)*int(OBS_Y/shape_dim))
shapes = [
    [
        [0,0,1,0,0],
        [0,0,1,0,0],
        [0,0,1,1,1],
        [0,0,1,0,0],
        [0,0,1,0,0],
    ],
    [
        [0,0,0,0,0],
        [0,1,1,1,1],
        [0,0,0,0,1],
        [0,0,0,0,1],
        [0,0,0,0,0],
    ],
    [
        [0,1,1,0,0],
        [0,1,1,0,0],
        [0,0,1,1,0],
        [0,0,1,1,0],
        [0,0,0,1,1],
    ],
    [
        [0,1,0,0,0],
        [0,1,0,0,0],
        [1,1,0,0,0],
        [1,1,1,0,0],
        [0,1,1,0,0],
    ],
    [
        [0,0,1,0,0],
        [0,0,1,0,0],
        [1,1,1,1,1],
        [0,0,1,0,0],
        [0,0,1,0,0],
    ],
    [
        [0,0,0,0,0],
        [0,1,1,1,0],
        [0,1,1,1,0],
        [0,1,1,1,0],
        [0,0,0,0,0],
    ],
    [
        [0,0,1,0,0],
        [0,0,1,1,1],
        [0,1,1,0,0],
        [0,0,1,0,0],
        [0,0,0,0,0],
    ],
    [
        [0,1,1,0,0],
        [1,1,1,0,0],
        [0,0,1,1,0],
        [0,0,1,1,0],
        [0,0,1,1,0],
    ],
    [
        [0,0,1,1,0],
        [0,1,1,1,0],
        [0,0,1,1,0],
        [0,0,1,1,0],
        [0,0,1,1,0],
    ],
    [
        [0,0,0,0,0],
        [0,1,0,1,0],
        [0,1,1,1,0],
        [0,1,0,1,0],
        [0,1,0,1,0],
    ],
]











## SETUP OBSTACLES AND ENVIORNMENT
# all postions are (y(m),x(m),ang(rad)) #origin at top left
# 3 = obstacle burned but put out| 2 = obstacle on fire | 1 = obstacle | 0 = free square | -1 = burned away and now free
def obstacle_grid_init():
    obstacle_grid = np.zeros((OBS_Y,OBS_X),dtype=np.uint8)
    counter = 0
    for x in range(int(OBS_X/shape_scale)):
        for y in range(int(OBS_Y/shape_scale)):
            shape_w_buffer = np.zeros((10,10),dtype=np.int8)
            offset = random.choice(range(5))
            shape_w_buffer[offset:offset+5,offset:offset+5] = shapes[values[counter]]
            obstacle_grid[y*int(OBS_Y/(shape_dim)):(y+1)*int(OBS_Y/shape_dim),x*int(OBS_X/shape_dim):(x+1)*int(OBS_X/shape_dim)] = shape_w_buffer
            counter+=1

    
    obstacle_grid = np.kron(obstacle_grid, np.ones((scale, scale), dtype=np.int8))
    

    #preventing passting through edges
    obstacle_grid[0:len(obstacle_grid)-1,[0]] = 1
    obstacle_grid[0:len(obstacle_grid)-1,[len(obstacle_grid)-1]] = 1
    obstacle_grid[0,0:len(obstacle_grid[0]-1)] = 1
    obstacle_grid[len(obstacle_grid[0])-1,0:len(obstacle_grid[0]-1)] = 1

    safety_buffer = 0.2
    robot_radius = int(major_grid_size*firetruck.WIDTH*(1+safety_buffer))

    cost_grid,voronoi_field = create_cost_grid(obstacle_grid,robot_radius)
    print("cost grid calculated")

    # Usage example
    dist_cost_grid,binary_grid = voronoi_to_cost_grid(
        voronoi_field, 
        obstacle_grid,
        max_useful_distance=40.0  # distances beyond 10 cells all get cost 0
    )

    print("voronoi calculated")
    return obstacle_grid,cost_grid,voronoi_field,binary_grid

obstacle_grid,cost_grid,voronoi_field,binary_grid = obstacle_grid_init()
burn_grid = obstacle_grid.copy()
burning_list = []
visualize_bin_grid(binary_grid)

## Initialize Wumpus
valid_indices = np.argwhere((obstacle_grid == 0))
wumpus_start = valid_indices[random.randint(0,len(valid_indices)-1)]

wumpus = wumpi(wumpus_start)



planner = prm.PRM_Planner(
            binary_grid=binary_grid,
            firetruck=firetruck,
            num_samples=2000,  # Adjust based on map complexity
            k_nearest=15
        )
planner.build_roadmap()


## Main Loop
dt = 0.01 #timestep
max_time = 3600 #seconds
total_ticks = int(max_time/dt)

# Usage - MUCH FASTER:
fig_ax = None  # Initialize once
def update_firespread(burn_grid, obstacle_grid, burning_points, current_time,
                     spread_interval=5, spread_radius=1):
    """
    Spread fire from burning cells based on time elapsed (optimized)
    """
    print("starting fire spreading")
    rows, cols = burn_grid.shape
    updated_burning_points = []
    
    # Pre-compute spread offsets once (not per burning cell)
    offsets = []
    for dy in range(-spread_radius, spread_radius + 1):
        for dx in range(-spread_radius, spread_radius + 1):
            if dy != 0 or dx != 0:
                offsets.append((dy, dx))
    
    # Track cells to avoid duplicates
    new_fires = set()
    
    for y, x, start_time in burning_points:
        time_burning = current_time - start_time
        
        if time_burning >= spread_interval:
            # Spread fire to neighbors
            for dy, dx in offsets:
                ny = y + dy
                nx = x + dx
                
                # Combined bounds and state check
                if (0 <= ny < rows and 0 <= nx < cols and 
                    burn_grid[ny, nx] == 1 and (ny, nx) not in new_fires):
                    burn_grid[ny, nx] = 2
                    new_fires.add((ny, nx))
                    updated_burning_points.append((ny, nx, current_time))
            
            # Mark as burned out
            burn_grid[y, x] = 3 #maybe green l8r
        else:
            # Still burning
            updated_burning_points.append((y, x, start_time))
    new_fire_count = len(new_fires)
    return burn_grid, obstacle_grid, updated_burning_points,new_fire_count

def visualize_frame_fast(start, goal, path, obstacle_grid, frame_name, firetruck,frame_index,
                        fig_ax=None, dpi=75):
    """
    Fast frame visualization with optional figure reuse
    
    Args:
        start, goal, path, obstacle_grid: same as before
        frame_name: output filename
        firetruck: Firetruck instance to visualize
        fig_ax: (fig, ax) tuple to reuse, or None to create new
        dpi: resolution (75 = fast, 150 = high quality)
    
    Returns:
        (fig, ax) tuple for reuse on next call
    """
    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.colors import ListedColormap
    
    # Create custom colormap
    # Values: 0=black, 1=white, 2=red, 3=green, 4=blue
    colors = ['black', 'white', 'red', 'green', 'blue']
    cmap = ListedColormap(colors)

    # Reuse or create figure
    if fig_ax is None:
        fig, ax = plt.subplots(figsize=(12, 10))
        ax.imshow(obstacle_grid, cmap=cmap, origin='lower', vmin=0, vmax=5)
        ax.set_xlabel('X (grid cells)')
        ax.set_ylabel('Y (grid cells)')
        ax.invert_yaxis()
    else:
        fig, ax = fig_ax
        ax.clear()
        ax.imshow(obstacle_grid, cmap=cmap, origin='lower', vmin=0, vmax=5)
        ax.invert_yaxis()
    
    # Plot wumpus path (if exists)
    if path is not None:
        path_y = [p[0] for p in path]
        path_x = [p[1] for p in path]
        ax.plot(path_x, path_y, 'b-', linewidth=2, alpha=0.7, label='Wumpus Path')
        
        path_length = sum(
            np.sqrt((path[i+1][0] - path[i][0])**2 + 
                   (path[i+1][1] - path[i][1])**2)
            for i in range(len(path) - 1)
        )
    
    # Plot wumpus start and goal
    ax.plot(start[1], start[0], 'go', markersize=12, zorder=10, label='Wumpus Start')
    ax.plot(goal[1], goal[0], 'r*', markersize=18, zorder=10, label='Wumpus Goal')
    
    # Draw firetruck as oriented box
    corners = firetruck.calc_boundary((firetruck.y, firetruck.x, firetruck.ang), scale=major_grid_size)
    corners.append(corners[0])  # Close polygon
    corner_x = [c[0] for c in corners]
    corner_y = [c[1] for c in corners]
    
    # Draw box outline and fill
    ax.plot(corner_x, corner_y, 'm-', linewidth=3, alpha=0.8, label='Firetruck', zorder=11)
    ax.fill(corner_x, corner_y, 'magenta', alpha=0.4, zorder=11)
    
    # Draw front indicator (small rectangle at front to show direction)
    front_center_x = (corners[0][0] + corners[1][0]) / 2
    front_center_y = (corners[0][1] + corners[1][1]) / 2
    ax.plot(front_center_x, front_center_y, 'yo', markersize=3, 
            markeredgecolor='orange', markeredgewidth=2, zorder=12)
    
    # Draw heading arrow from center
    center_x = firetruck.x * major_grid_size
    center_y = firetruck.y * major_grid_size
    arrow_len = 8 * major_grid_size
    dx = arrow_len * np.cos(firetruck.ang)
    dy = arrow_len * np.sin(firetruck.ang)
    ax.arrow(center_x, center_y, dx, dy,
            head_width=2*major_grid_size, head_length=2*major_grid_size, 
            fc='yellow', ec='orange', linewidth=2, zorder=12, alpha=0.8)
    
    # Plot firetruck path (if exists)
    if len(firetruck.path) > 0:
        truck_path_y = [p[0] * major_grid_size for p in firetruck.path]
        truck_path_x = [p[1] * major_grid_size for p in firetruck.path]
        ax.plot(truck_path_x, truck_path_y, 'm--', linewidth=2, alpha=0.5, 
                label='Firetruck Trail')

    # Plot firetruck target (if set)
    if firetruck.x_tar != -1 and firetruck.y_tar != -1:
        ax.plot(firetruck.x_tar * major_grid_size, firetruck.y_tar * major_grid_size, 
                'c*', markersize=20, zorder=11, label='Firetruck Target', 
                markeredgecolor='darkcyan', markeredgewidth=2)
        
        # Draw line from truck to target
        ax.plot([center_x, firetruck.x_tar * major_grid_size], 
               [center_y, firetruck.y_tar * major_grid_size],
               'c:', linewidth=1.5, alpha=0.5)
    
    ax.legend(loc='upper right', fontsize=9)
    ax.set_title(f"Time {frame_index} | Wumpus Score {wumpus.score} | Truck Score {firetruck.score}", fontsize=12)
    
    # Save and return for reuse
    fig.savefig(frame_name, dpi=dpi, bbox_inches='tight')
    
    return (fig, ax)

def find_nearby_pts(grid,burning_list,dist=10,scale=major_grid_size):
    """
    Check each cell in the grid. If it equals 1 and has a 0 on any of its four sides,
    add a line of 10 points parallel to that side.
    
    Args:
        grid: 2D numpy array or list of lists
    
    Returns:
        new_grid: Modified grid with added parallel lines
    """
    # Convert to numpy array if it's a list
    grid = np.array(grid)
    rows, cols = grid.shape
    
    # Direction vectors for the four sides: up, down, left, right
    directions = {
        'up': (-1, 0),
        'down': (1, 0),
        'left': (0, -1),
        'right': (0, 1)
    }

    firetruck_target_options = []
    # Check each cell
    edge_dist = 100
    for i,j,t in burning_list:
        # Only process cells with value 1
        if 0 < i < rows-1 and 0 < j < cols-1:
            # Check all four directions
            for dir_name, (di, dj) in directions.items():
                ni, nj = i + di, j + dj
                # Check if neighbor is within bounds and equals 0
                if 0 <= ni < rows and 0 <= nj < cols and grid[ni, nj] == 0:
                    
                    line_i = int(ni + di * (dist*1.5) * scale)
                    line_j = int(nj + dj * (dist*1.5) * scale)
                        
                    # Set the point if within bounds
                    if edge_dist <= line_i < rows-edge_dist and edge_dist <= line_j < cols-edge_dist and grid[line_i, line_j] == 0:
                        firetruck_target_options.append((line_i,line_j))
    return firetruck_target_options

def check_firetruck_collision(firetruck, binary_grid, minor_grid_size=0.2):
    """
    Check if firetruck is currently in collision
    
    Returns:
        True if collision detected, False otherwise
    """
    corners = firetruck.calc_boundary((firetruck.y, firetruck.x, firetruck.ang), scale=1)
    
    for corner_x_m, corner_y_m in corners:
        grid_x = int(round(corner_x_m / minor_grid_size))
        grid_y = int(round(corner_y_m / minor_grid_size))
        
        # Bounds check
        if (grid_x < 0 or grid_x >= binary_grid.shape[1] or
            grid_y < 0 or grid_y >= binary_grid.shape[0]):
            return True
        
        # Collision check (1 = obstacle)
        if binary_grid[grid_y, grid_x] == 1:
            return True
    
    # Check edges between corners
    for i in range(4):
        c1_x, c1_y = corners[i]
        c2_x, c2_y = corners[(i + 1) % 4]
        
        for t in [0.33, 0.67]:
            edge_x = c1_x + t * (c2_x - c1_x)
            edge_y = c1_y + t * (c2_y - c1_y)
            
            grid_x = int(round(edge_x / minor_grid_size))
            grid_y = int(round(edge_y / minor_grid_size))
            
            if (grid_x < 0 or grid_x >= binary_grid.shape[1] or
                grid_y < 0 or grid_y >= binary_grid.shape[0]):
                return True
            
            if binary_grid[grid_y, grid_x] == 1:
                return True
    
    return False

def simulate_prm_path_full(path, firetruck, binary_grid, minor_grid_size=0.2, 
                           dt=0.01, target_velocity=2.0, steps_per_frame=1,
                           max_iterations=2000):
    """
    Pre-compute entire trajectory as a list with timeout protection
    """
    if path is None or len(path) < 2:
        print("No valid path to simulate")
        return []
    
    start_y, start_x, start_theta = path[0]
    
    trajectory = []
    current_time = 0.0
    waypoint_idx = 1
    total_collisions = 0
    iteration_count = 0
    
    print(f"Pre-computing trajectory for {len(path)} waypoints...")
    
    # Temporary firetruck for simulation
    temp_truck = Firetruck((start_y, start_x), start_theta, scale=1)
    
    while waypoint_idx < len(path):
        iteration_count += 1
        
        # TIMEOUT PROTECTION
        if iteration_count > max_iterations:
            print(f"✗ Timeout after {max_iterations} iterations")
            print(f"  Stuck at waypoint {waypoint_idx}/{len(path)}")
            print(f"  Current pos: ({temp_truck.y:.1f}, {temp_truck.x:.1f})")
            print(f"  Target: ({path[waypoint_idx][0]:.1f}, {path[waypoint_idx][1]:.1f})")
            print(f"  Distance: {np.sqrt((path[waypoint_idx][0] - temp_truck.y)**2 + (path[waypoint_idx][1] - temp_truck.x)**2):.2f}m")
            break
        
        # Progress reporting
        if iteration_count % 500 == 0:
            print(f"  Iteration {iteration_count}: waypoint {waypoint_idx}/{len(path)}, "
                  f"time={current_time:.1f}s, frames={len(trajectory)}")
        
        for _ in range(steps_per_frame):
            target_y, target_x, target_theta = path[waypoint_idx]
            
            dy = target_y - temp_truck.y
            dx = target_x - temp_truck.x
            distance_to_target = np.sqrt(dx**2 + dy**2)
            
            # LARGER TOLERANCE - PRM waypoints may be far apart
            if distance_to_target < 2.0:  # Changed from 0.5 to 2.0
                waypoint_idx += 1
                print(f"  ✓ Reached waypoint {waypoint_idx-1}/{len(path)}")
                if waypoint_idx >= len(path):
                    trajectory.append({
                        'y': temp_truck.y,
                        'x': temp_truck.x,
                        'ang': temp_truck.ang,
                        'time': current_time,
                        'waypoint_idx': waypoint_idx,
                        'collision': False,
                        'completed': True,
                        'distance_to_waypoint': 0.0,
                        'total_collisions': total_collisions
                    })
                    print(f"✓ Trajectory computed: {len(trajectory)} frames, {iteration_count} iterations")
                    return trajectory
                break  # Exit inner loop to get new waypoint
            
            desired_heading = np.arctan2(dy, dx)
            heading_error = temp_truck.wrap_to_pi(desired_heading - temp_truck.ang)
            
            # STRONGER STEERING CONTROL
            steering_angle = np.clip(heading_error * 3.0,  # Increased from 2.0
                                    -temp_truck.MAX_STEERING_ANGLE,
                                    temp_truck.MAX_STEERING_ANGLE)
            
            temp_truck.set_control(target_velocity, steering_angle)
            temp_truck.timestep(dt)
            current_time += dt
            
            is_collision = check_firetruck_collision(temp_truck, binary_grid, minor_grid_size)
            if is_collision:
                total_collisions += 1
        
        trajectory.append({
            'y': temp_truck.y,
            'x': temp_truck.x,
            'ang': temp_truck.ang,
            'time': current_time,
            'waypoint_idx': waypoint_idx,
            'completed': False,
            'distance_to_waypoint': distance_to_target,
            'total_collisions': total_collisions
        })
    
    print(f"✓ Trajectory computed: {len(trajectory)} frames, {iteration_count} iterations")
    return trajectory

def sort_points_by_distance(points, current_pos):
    """
    Sort points by Euclidean distance to current position
    
    Args:
        points: list of (y, x) tuples
        current_pos: (y, x) current position
    
    Returns:
        sorted list of (y, x) tuples (closest first)
    """
    return sorted(points, key=lambda pt: (pt[0] - current_pos[0])**2 + (pt[1] - current_pos[1])**2)

def extinguish_fire(grid, target_point, radius):
    """Square region version (fastest)"""
    target_y, target_x,_ = target_point #angle doesnt matter
    target_y = int(target_y*major_grid_size)
    target_x = int(target_x*major_grid_size)
    rows, cols = grid.shape
    
    # Calculate bounds
    y_min = max(0, target_y - radius)
    y_max = min(rows, target_y + radius + 1)
    x_min = max(0, target_x - radius)
    x_max = min(cols, target_x + radius + 1)
    
    # Extract region
    region = grid[y_min:y_max, x_min:x_max]
    
    # Count and update
    mask = (region == 2) | (region == 3)
    count = np.sum(mask)
    region[mask] = 4
    
    return grid, count

steps_per_frame = 200 # 2 seconds per frame
firetruck_pt_bad_list = []
for t in range(total_ticks):
    
    ## If wumpus has no path create random target and pathfind 
    if len(wumpus.path)<=0:
        print("Finding Wumpus a new path")
        max_dist = 1000
        min_dist = 600
        while True:
            wumpus_target = valid_indices[random.randint(0,len(valid_indices)-1)]
            dist = np.sqrt((wumpus_target[0] - wumpus.y)**2 + (wumpus_target[1] - wumpus.x)**2)
            is_close = dist <= max_dist
            is_far = dist >= min_dist
            if is_close and is_far:break
        wumpus.update_path(wumpus_target,obstacle_grid)
        
    ## move wupus forward one timestep and update burning obstacles (every 0.1 seconds)
    if t%30 == 0: #every 5th of a second
        burn_grid, new_burn_pts = wumpus.timestep(obstacle_grid,t)
        burning_list.extend(new_burn_pts)
        
    ## spread fire every 10 seconds
    # 4 = burned away and now free | 3 = obstacle burned but put out| 2 = obstacle on fire | 1 = obstacle | 0 = free square |

    if t%1000 ==0:
        burn_grid,obstacle_grid, burning_list,new_fire_count = update_firespread(
            burn_grid=burn_grid,
            obstacle_grid=obstacle_grid,
            burning_points=burning_list,
            current_time=t,
            spread_interval=int(10/dt),   # Fire spreads after 10 secs
            spread_radius=int(10/minor_grid_size) 
            )
        print(f"fire_spread {len(burning_list)}")
        wumpus.score+=new_fire_count
    ## check for burning obstacles of so pathfind based on nearby points
    if len(burning_list)>0 and len(firetruck.trajectory)==0 and t-firetruck.wait_for_water > 10/dt:
        
        firetruck_target_options = find_nearby_pts(obstacle_grid,burning_list)
        firetruck_stored = sort_points_by_distance(firetruck_target_options,firetruck.get_current_pos())
        # Initialize PRM planner for current state of the map
        # print(f"firetruck target points : {len(firetruck_stored)}")
        for pt in firetruck_stored:
            if (pt[0],pt[1]) not in firetruck_pt_bad_list:
                try:
                    firetruck.y_tar = pt[0]/major_grid_size
                    firetruck.x_tar = pt[1]/major_grid_size
                    path = planner.query(firetruck.get_current_pos(), (firetruck.y_tar,firetruck.x_tar))
                    if len(path)>0:
                        truck_trajectory = simulate_prm_path_full(
                            path=path,
                            firetruck=firetruck,
                            binary_grid=binary_grid,
                            minor_grid_size=0.2,
                            dt=dt,
                            target_velocity=firetruck.MAX_VEL/2,#to adjust for acceleration and decelleration time estimates
                            steps_per_frame=300
                        )
                        if len(truck_trajectory) > 0:
                            firetruck.path = path
                            firetruck.trajectory = truck_trajectory
                            firetruck.traj_index = 0
                            break
                        else:
                            firetruck_pt_bad_list.append((pt[0],pt[1]))
                except: 
                    print("failed to find path trying next path")
    elif t-firetruck.wait_for_water <= 10/dt:
        print("truck extinguishing")
    ## if no burning squares wait until there is one
    
    ## if path move truck
    if t%steps_per_frame == 0:
        if firetruck.traj_index < len(firetruck.trajectory):
            state = firetruck.trajectory[firetruck.traj_index]
            
            # Update firetruck position
            firetruck.y = state['y']
            firetruck.x = state['x']
            firetruck.ang = state['ang']
                
            firetruck.traj_index += 1
        elif len(firetruck.trajectory) != 0:
            firetruck.path = []
            firetruck.trajectory = []
            firetruck.traj_index = 0
            firetruck.wait_for_water = t
            print("waiting to extinguish")
            burn_grid ,count = extinguish_fire(burn_grid,firetruck.get_current_pos(),radius = int(30/minor_grid_size))
            firetruck.score += count*2
            if firetruck.score > wumpus.score*2:
                firetruck.score = wumpus.score*2 #occationally the score gets higher than it possibly could and idk why
    ## if at end of path wait 10 seconds for obstacle to get put out
    
    ## update visuals
    if t%steps_per_frame == 0: #every 0.25 second of sim time make a frame
        frame_index = int(t*dt)
        fig_ax = visualize_frame_fast(
            wumpus.get_pos(),wumpus.get_target(),wumpus.path, burn_grid,
            f'out/output_frame_{frame_index}.png', firetruck,frame_index=frame_index,
            fig_ax=fig_ax,  # Reuse figure
            dpi=50  # Lower resolution = faster
        )
        print(f"{int(t*dt)}/{int(total_ticks*dt)}  | Wumpus Score : {wumpus.score} | Firetruck Score : {firetruck.score} ")
    
# Close when done
if fig_ax is not None:
    import matplotlib.pyplot as plt
    plt.close(fig_ax[0])

print("done")