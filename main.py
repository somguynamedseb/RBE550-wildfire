import random
import time
import numpy as np
from scipy.ndimage import distance_transform_edt, binary_dilation
import  matplotlib.pyplot as plt

from firehouse import Firetruck
from pathfinding import DubinsPath
from pathfinding import State
import pathfinding as pf

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

def v_cost_to_binary(cost_grid,th = 0.5):
    binary_grid = np.zeros_like(cost_grid)
    for i in range(len(cost_grid)):
        for j in range(len(cost_grid[0])):
            if cost_grid[i][j] >= th:
                binary_grid[i][j] = 1
    return binary_grid

MAX_Y = 250 #meters
MAX_X = 250 #meters

minor_grid_size = 0.2
major_grid_size = 5
shape_dim = 5
shape_scale = 10

OBS_Y = int(MAX_Y/major_grid_size)
OBS_X = int(MAX_X/major_grid_size)
scale = int(major_grid_size/minor_grid_size)

dt = 0.05 # seconds
global_time = 0.0 #time starts at 0
percent_fill = 0.10
print(f"scale {scale}")
print(f"OBS {OBS_X}")

random.seed(24)


values = random.choices(range(10), k=int(OBS_X/shape_dim)*int(OBS_Y/shape_dim))
shapes = [
    # 1. I shape (vertical)
    [
        [0,0,1,0,0],
        [0,0,1,0,0],
        [0,0,1,0,0],
        [0,0,1,0,0],
        [0,0,0,0,0],
    ],
    # 2. I shape (horizontal)
    [
        [0,0,0,0,0],
        [0,1,1,1,1],
        [0,0,0,0,0],
        [0,0,0,0,0],
        [0,0,0,0,0],
    ],
    # 3. L shape
    [
        [0,0,1,0,0],
        [0,0,1,0,0],
        [0,0,1,1,0],
        [0,0,0,0,0],
        [0,0,0,0,0],
    ],
    # 4. J shape (mirrored L)
    [
        [0,1,0,0,0],
        [0,1,0,0,0],
        [1,1,0,0,0],
        [0,0,0,0,0],
        [0,0,0,0,0],
    ],
    # 5. T shape
    [
        [0,0,0,0,0],
        [0,1,1,1,0],
        [0,0,1,0,0],
        [0,0,0,0,0],
        [0,0,0,0,0],
    ],
    # 6. O shape (square)
    [
        [0,0,0,0,0],
        [0,1,1,0,0],
        [0,1,1,0,0],
        [0,0,0,0,0],
        [0,0,0,0,0],
    ],
    # 7. S shape
    [
        [0,0,0,0,0],
        [0,0,1,1,0],
        [0,1,1,0,0],
        [0,0,0,0,0],
        [0,0,0,0,0],
    ],
    # 8. Z shape
    [
        [0,0,0,0,0],
        [0,1,1,0,0],
        [0,0,1,1,0],
        [0,0,0,0,0],
        [0,0,0,0,0],
    ],
    # 9. Cross shape
    [
        [0,0,1,0,0],
        [0,1,1,1,0],
        [0,0,1,0,0],
        [0,0,0,0,0],
        [0,0,0,0,0],
    ],
    # 10. U shape
    [
        [0,0,0,0,0],
        [0,1,0,1,0],
        [0,1,1,1,0],
        [0,0,0,0,0],
        [0,0,0,0,0],
    ],
]


# all postions are (y(m),x(m),ang(rad)) #origin at top left
## SETUP OBSTACLES AND ENVIORNMENT
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

obstacle_grid[0:len(obstacle_grid)-1,[0]] = 1
obstacle_grid[0:len(obstacle_grid)-1,[len(obstacle_grid)-1]] = 1
obstacle_grid[0,0:len(obstacle_grid[0]-1)] = 1
obstacle_grid[len(obstacle_grid[0])-1,0:len(obstacle_grid[0]-1)] = 1

robot_width = 2.2 #based on assignment details
safety_buffer = 0.1
robot_radius = int(scale*(robot_width / 2)* (1+safety_buffer))

cost_grid,voronoi_field = create_cost_grid(obstacle_grid,robot_radius)
print("cost grid calculated")

# Usage example
dist_cost_grid,binary_grid = voronoi_to_cost_grid(
    voronoi_field, 
    obstacle_grid,
    max_useful_distance=40.0  # distances beyond 10 cells all get cost 0
)

print("voronoi calculated")
# voronoi_field_up = np.kron(voronoi_field, np.ones((major_grid_size, major_grid_size), dtype=int))
obstacle_grid_up = np.kron(obstacle_grid, np.ones((major_grid_size, major_grid_size), dtype=int))
dist_cost_grid_up = np.kron(dist_cost_grid, np.ones((major_grid_size, major_grid_size), dtype=int))
binary_grid_up = np.kron(binary_grid, np.ones((major_grid_size, major_grid_size), dtype=int))

print("scaling complete")
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
    # plt.show()
    
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
    # plt.show()


# Usage
visualize_cost_grid(dist_cost_grid_up, obstacle_grid_up)
visualize_bin_grid(binary_grid_up)


print("Setting up pathfinding...")

minor_grid_size = 0.2  # meters per cell

start_y_meters = 30 
start_x_meters = 50
start_angle = 0

goal_y_meters = 190
goal_x_meters = 190
goal_angle = np.deg2rad(45)

start_grid_y = int(start_y_meters / minor_grid_size)  
start_grid_x = int(start_x_meters / minor_grid_size)  
goal_grid_y = int(goal_y_meters / minor_grid_size)    
goal_grid_x = int(goal_x_meters / minor_grid_size)    

print(f"Start: grid[{start_grid_y}, {start_grid_x}] = ({start_y_meters}m, {start_x_meters}m)")
print(f"Goal: grid[{goal_grid_y}, {goal_grid_x}] = ({goal_y_meters}m, {goal_x_meters}m)")
print(f"Distance: {abs(goal_x_meters - start_x_meters)}m")

# Create firetruck with meter coordinates
firetruck = Firetruck((start_y_meters, start_x_meters), start_angle, scale=1)

# Create states with meter coordinates
start_state = pf.State(y=start_y_meters, x=start_x_meters, theta=start_angle)
goal_state = pf.State(y=goal_y_meters, x=goal_x_meters, theta=goal_angle)


start_time = time.time()

# Right before calling pathfinding, add these debug checks:
start_pos = (start_grid_y,start_grid_x)
goal_pos = (goal_grid_y,goal_grid_x)
print("\n=== DEBUG INFO ===")
print(f"Start position: ({start_pos[0]}, {start_pos[1]})")
print(f"Goal position: ({goal_pos[0]}, {goal_pos[1]})")

# Check if start is valid
start_y_idx = int(round(start_pos[0]))
start_x_idx = int(round(start_pos[1]))
print(f"Start grid indices: ({start_y_idx}, {start_x_idx})")
print(f"  binary_grid value: {binary_grid[start_y_idx, start_x_idx]} (should be 0 for valid)")
print(f"  cost_grid value: {dist_cost_grid[start_y_idx, start_x_idx]:.3f}")

# Check if goal is valid
goal_y_idx = int(round(goal_pos[0]))
goal_x_idx = int(round(goal_pos[1]))
print(f"Goal grid indices: ({goal_y_idx}, {goal_x_idx})")
print(f"  binary_grid value: {binary_grid[goal_y_idx, goal_x_idx]} (should be 0 for valid)")
print(f"  cost_grid value: {dist_cost_grid[goal_y_idx, goal_x_idx]:.3f}")

# Check firetruck footprint at start
start_corners = firetruck.calc_boundary((start_pos[0], start_pos[1], start_angle), scale=1)
print(f"\nStart vehicle corners:")
for i, corner in enumerate(start_corners):
    cx, cy = int(round(corner[0])), int(round(corner[1]))
    if 0 <= cy < binary_grid.shape[0] and 0 <= cx < binary_grid.shape[1]:
        print(f"  Corner {i}: ({corner[0]:.1f}, {corner[1]:.1f}) -> grid[{cy},{cx}] = {binary_grid[cy, cx]}")
    else:
        print(f"  Corner {i}: ({corner[0]:.1f}, {corner[1]:.1f}) -> OUT OF BOUNDS")

# Check grid statistics
print(f"\nGrid statistics:")
print(f"  binary_grid shape: {binary_grid.shape}")
print(f"  Valid squares (0s): {np.sum(binary_grid == 0)} ({100*np.sum(binary_grid == 0)/binary_grid.size:.1f}%)")
print(f"  Obstacles (1s): {np.sum(binary_grid == 1)} ({100*np.sum(binary_grid == 1)/binary_grid.size:.1f}%)")
print(f"  Turning radius: {firetruck.MIN_TURN_RAD}m")
print("==================\n")

result = pf.hybrid_astar_with_NHO(
    start=start_state,
    goal=goal_state,
    valid_grid=binary_grid,
    cost_grid=dist_cost_grid,
    truck=firetruck,
    minor_grid_size=minor_grid_size,  # ADD THIS
    timeout=120.0
)

elapsed_time = time.time() - start_time


if isinstance(result, tuple):
    path, expanded_nodes = result
else:
    path = result
    expanded_nodes = []

# Visualize the search
if len(expanded_nodes) > 0:
    print(f"\nVisualizing {len(expanded_nodes)} expanded nodes...")
    
    pf.visualize_search_tree(
        start_state=start_state,
        goal_state=goal_state,
        expanded_nodes=expanded_nodes,
        binary_grid=binary_grid,
        dist_cost_grid=dist_cost_grid,
        minor_grid_size=0.2
    )
    
    # Show search progress over time
    pf.visualize_search_progress(
        expanded_nodes=expanded_nodes,
        start_state=start_state,
        goal_state=goal_state,
        binary_grid=binary_grid,
        minor_grid_size=0.2,
        num_frames=10
    )


# Step 5: Process and visualize result
if path is not None:
    print(f"Path found with {len(path)} waypoints in {elapsed_time:.2f}s")
    
    # Visualize path on upscaled grid
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # Show cost grid as background
    ax.imshow(dist_cost_grid_up, cmap='plasma', origin='lower', 
              vmin=0, vmax=1, alpha=0.7)
    
    # Show obstacles
    ax.contour(obstacle_grid_up, levels=[0.5], colors='cyan', linewidths=2)
    
    # Convert path to upscaled coordinates for visualization
    path_y = [p[0] * scale for p in path]
    path_x = [p[1] * scale for p in path]
    
    # Plot path
    ax.plot(path_x, path_y, 'g-', linewidth=2, label='Path', zorder=10)
    ax.plot(path_x[0], path_y[0], 'go', markersize=5, label='Start', zorder=11)
    ax.plot(path_x[-1], path_y[-1], 'ro', markersize=5, label='Goal', zorder=11)
    
    # Show vehicle footprint at waypoints (every 10th waypoint)
    step = max(1, len(path) // 15)  # Show ~15 vehicle poses
    for i in range(0, len(path), step):
        corners = firetruck.calc_boundary(path[i], scale=scale)
        corners.append(corners[0])  # close the polygon
        corner_x = [c[0] for c in corners]
        corner_y = [c[1] for c in corners]
        ax.plot(corner_x, corner_y, 'b-', alpha=0.4, linewidth=1.5)
        
        # Draw heading arrow
        y, x, theta = path[i]
        arrow_len = 3 * scale
        dx = arrow_len * np.cos(theta)
        dy = arrow_len * np.sin(theta)
        ax.arrow(x * scale, y * scale, dx, dy, 
                head_width=scale, head_length=scale, 
                fc='yellow', ec='orange', alpha=0.6)
    
    # Show final vehicle position
    final_corners = firetruck.calc_boundary(path[-1], scale=scale)
    final_corners.append(final_corners[0])
    final_x = [c[0] for c in final_corners]
    final_y = [c[1] for c in final_corners]
    ax.plot(final_x, final_y, 'r-', linewidth=2, label='Final pose')
    
    ax.legend(loc='upper right')
    ax.set_title(f'Hybrid A* Path ({len(path)} waypoints, {elapsed_time:.2f}s)')
    ax.set_xlabel('X (meters)')
    ax.set_ylabel('Y (meters)')
    
    # Set tick labels in meters
    ax.set_xticks(np.linspace(0, MAX_X*scale, int(MAX_X/25)+1))
    ax.set_xticklabels(range(0, MAX_X+1, 25))
    ax.set_yticks(np.linspace(0, MAX_Y*scale, int(MAX_Y/25)+1))
    ax.set_yticklabels(range(0, MAX_Y+1, 25))
    
    ax.invert_yaxis()
    plt.tight_layout()
    plt.savefig("path_result.png", dpi=150)
    print("Saved path_result.png")
    # plt.show()

else:
    print("no path found")
    fig, ax = plt.subplots(figsize=(10, 10))
    
    # Show obstacles
    ax.imshow(binary_grid, cmap='binary', origin='lower')
    
    # Convert meters to grid cells
    start_x = start_state.x / minor_grid_size
    start_y = start_state.y / minor_grid_size
    goal_x = goal_state.x / minor_grid_size
    goal_y = goal_state.y / minor_grid_size
    
    # Plot positions
    ax.plot(start_x, start_y, 'go', markersize=5, label='Start')
    ax.plot(goal_x, goal_y, 'r*', markersize=5, label='Goal')
    
    # Plot vehicle corners
    start_corners = firetruck.calc_boundary((start_state.y, start_state.x, start_state.theta), scale=1)
    for i, (cx, cy) in enumerate(start_corners):
        ax.plot(cx / minor_grid_size, cy / minor_grid_size, 'b.', markersize= 4)
    
    goal_corners = firetruck.calc_boundary((goal_state.y, goal_state.x, goal_state.theta), scale=1)
    for i, (cx, cy) in enumerate(goal_corners):
        ax.plot(cx / minor_grid_size, cy / minor_grid_size, 'b.', markersize= 4)
    
    ax.legend()
    ax.set_title('Start and Goal Positions')
    ax.invert_yaxis()
    plt.tight_layout()
    plt.savefig("start_goal_debug.png")
    # plt.show()
    