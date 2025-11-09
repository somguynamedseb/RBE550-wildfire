import random
import time
import numpy as np
from scipy.ndimage import distance_transform_edt, binary_dilation
import  matplotlib.pyplot as plt

from firehouse import Firetruck
from Hybrid_A_Star import DubinsPath
from Hybrid_A_Star import State
import Hybrid_A_Star as pf
import matplotlib.animation as animation

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

def v_cost_to_binary(cost_grid,th = 0.3):
    binary_grid = np.zeros_like(cost_grid)
    for i in range(len(cost_grid)):
        for j in range(len(cost_grid[0])):
            if cost_grid[i][j] >= th:
                binary_grid[i][j] = 1
    return binary_grid

def path_to_motion_commands(path, truck, dt=0.05):
    """
    Convert path waypoints to motion commands for the truck
    
    Args:
        path: list of (y, x, theta) tuples from pathfinding
        truck: Firetruck instance
        dt: simulation timestep
    
    Returns:
        commands: list of (velocity, steering_angle, duration) tuples
    """
    commands = []
    
    for i in range(len(path) - 1):
        y_curr, x_curr, theta_curr = path[i]
        y_next, x_next, theta_next = path[i + 1]
        
        # Calculate distance and angle change
        dx = x_next - x_curr
        dy = y_next - y_curr
        distance = np.sqrt(dx**2 + dy**2)
        
        # Calculate steering angle needed
        # This is approximate - actual motion primitives were used in planning
        dtheta = truck.wrap_to_pi(theta_next - theta_curr)
        
        # Use fixed velocity
        velocity = 2.0  # m/s
        
        # Estimate steering angle from heading change
        if abs(dtheta) < 1e-6:
            steering_angle = 0.0
        else:
            # Approximate: R = distance / dtheta, steering = atan(L/R)
            R = distance / abs(dtheta) if abs(dtheta) > 1e-6 else 1e10
            steering_angle = np.arctan(truck.WHEELBASE / R)
            if dtheta < 0:
                steering_angle = -steering_angle
        
        # Clamp steering
        steering_angle = np.clip(steering_angle, 
                                -truck.MAX_STEERING_ANGLE, 
                                truck.MAX_STEERING_ANGLE)
        
        # Duration to cover this segment
        duration = distance / velocity
        
        commands.append((velocity, steering_angle, duration))
    
    return commands

def simulate_path_following(path, truck, binary_grid, minor_grid_size=0.2, dt=0.05):
    """
    Simulate truck following the planned path
    
    Args:
        path: list of (y, x, theta) waypoints
        truck: Firetruck instance (will be reset to start)
        binary_grid: for visualization
        minor_grid_size: meters per grid cell
        dt: simulation timestep
    
    Returns:
        trajectory: list of (y, x, theta, t) actual positions over time
        commands_used: list of commands executed
    """
    # Reset truck to start position
    start_y, start_x, start_theta = path[0]
    truck.y = start_y
    truck.x = start_x
    truck.ang = start_theta
    
    # Get motion commands from path
    commands = path_to_motion_commands(path, truck, dt)
    
    # Simulate execution
    trajectory = [(truck.y, truck.x, truck.ang, 0.0)]  # (y, x, theta, time)
    current_time = 0.0
    
    print(f"Simulating path with {len(commands)} segments...")
    
    for cmd_idx, (velocity, steering_angle, duration) in enumerate(commands):
        print(f"  Segment {cmd_idx+1}/{len(commands)}: v={velocity:.1f}m/s, "
            f"steer={np.degrees(steering_angle):.1f}°, dur={duration:.2f}s")
        
        # Set control
        truck.set_control(velocity, steering_angle)
        
        # Simulate for duration
        steps = int(duration / dt)
        for step in range(steps):
            truck.timestep(dt)
            current_time += dt
            trajectory.append((truck.y, truck.x, truck.ang, current_time))
    
    print(f"Simulation complete: {len(trajectory)} timesteps, {current_time:.2f}s total")
    
    return trajectory, commands

def visualize_trajectory(path, trajectory, binary_grid, dist_cost_grid, obstacle_grid,
                        firetruck, minor_grid_size=0.2):
    """
    Visualize planned path vs actual simulated trajectory
    
    Args:
        path: planned waypoints from A*
        trajectory: actual simulated positions
        binary_grid: obstacles
        dist_cost_grid: cost grid
        firetruck: Firetruck instance
        minor_grid_size: meters per grid cell
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8))
    
    # Left: Path vs Trajectory
    ax1.imshow(dist_cost_grid, cmap='plasma', origin='lower', vmin=0, vmax=1, alpha=0.5)
    ax1.contour(binary_grid, levels=[0.5], colors='cyan', linewidths=1)
    ax1.contour(obstacle_grid, levels=[0.5], colors='k', linewidths=2)
    # Plot planned path
    path_y = [p[0] / minor_grid_size for p in path]
    path_x = [p[1] / minor_grid_size for p in path]
    ax1.plot(path_x, path_y, 'b--', linewidth=2, label='Planned path', alpha=0.7)
    ax1.scatter(path_x, path_y, c='blue', s=50, marker='o', zorder=5, alpha=0.7)
    
    # Plot actual trajectory
    traj_y = [t[0] / minor_grid_size for t in trajectory]
    traj_x = [t[1] / minor_grid_size for t in trajectory]
    ax1.plot(traj_x, traj_y, 'g-', linewidth=1.5, label='Actual trajectory', alpha=0.8)
    
    # Mark start and end
    ax1.plot(path_x[0], path_y[0], 'go', markersize=12, label='Start', zorder=10)
    ax1.plot(path_x[-1], path_y[-1], 'r*', markersize=20, label='Goal', zorder=10)
    
    ax1.set_title('Planned Path vs Simulated Trajectory')
    ax1.set_xlabel('X (grid cells)')
    ax1.set_ylabel('Y (grid cells)')
    ax1.legend()
    ax1.invert_yaxis()
    ax1.grid(True, alpha=0.3)
    
    # Right: Vehicle footprints along trajectory
    ax2.imshow(binary_grid, cmap='binary', origin='lower', alpha=0.3)
    ax2.plot(traj_x, traj_y, 'g-', linewidth=1.5, alpha=0.5)
    ax2.contour(obstacle_grid, levels=[0.5], colors='k', linewidths=2)

    # Show vehicle footprint at intervals
    step_size = max(1, len(trajectory) // 20)  # Show ~20 footprints
    for i in range(0, len(trajectory), step_size):
        y, x, theta, t = trajectory[i]
        corners = firetruck.calc_boundary((y, x, theta), scale=1)
        corners.append(corners[0])  # Close polygon
        
        corner_x = [c[0] / minor_grid_size for c in corners]
        corner_y = [c[1] / minor_grid_size for c in corners]
        
        # Color by time (blue=early, red=late)
        color = plt.cm.coolwarm(i / len(trajectory))
        ax2.plot(corner_x, corner_y, color=color, linewidth=1.5, alpha=0.6)
        ax2.fill(corner_x, corner_y, color=color, alpha=0.1)
    
    # Final position highlighted
    y_final, x_final, theta_final, t_final = trajectory[-1]
    final_corners = firetruck.calc_boundary((y_final, x_final, theta_final), scale=1)
    final_corners.append(final_corners[0])
    final_x = [c[0] / minor_grid_size for c in final_corners]
    final_y = [c[1] / minor_grid_size for c in final_corners]
    ax2.plot(final_x, final_y, 'r-', linewidth=3, label='Final position')
    
    ax2.set_title(f'Vehicle Motion Over Time ({t_final:.1f}s)')
    ax2.set_xlabel('X (grid cells)')
    ax2.set_ylabel('Y (grid cells)')
    ax2.legend()
    ax2.invert_yaxis()
    
    plt.tight_layout()
    plt.savefig('trajectory_simulation.png', dpi=150)
    print("Saved trajectory_simulation.png")

def check_trajectory_collisions(trajectory, binary_grid, firetruck, minor_grid_size=0.2):
    """
    Check if simulated trajectory collides with obstacles
    
    Returns:
        (is_valid, collision_indices)
    """
    collision_indices = []
    
    for i, (y, x, theta, t) in enumerate(trajectory):
        corners = firetruck.calc_boundary((y, x, theta), scale=1)
        
        for corner_x_m, corner_y_m in corners:
            grid_x = int(round(corner_x_m / minor_grid_size))
            grid_y = int(round(corner_y_m / minor_grid_size))
            
            # Check bounds
            if (grid_x < 0 or grid_x >= binary_grid.shape[1] or
                grid_y < 0 or grid_y >= binary_grid.shape[0]):
                collision_indices.append(i)
                break
            
            # Check collision
            if binary_grid[grid_y, grid_x] == 1:
                collision_indices.append(i)
                break
    
    is_valid = len(collision_indices) == 0
    return is_valid, collision_indices

def animate_trajectory(trajectory, binary_grid, firetruck, minor_grid_size=0.2, 
                       frame_skip=5):
    """
    Create animated GIF of truck following trajectory
    """
    fig, ax = plt.subplots(figsize=(10, 10))
    
    # Static background
    ax.imshow(binary_grid, cmap='binary', origin='lower', alpha=0.3)
    
    # Initialize plots
    traj_line, = ax.plot([], [], 'g-', linewidth=2, label='Trajectory')
    vehicle_patch, = ax.plot([], [], 'r-', linewidth=2, label='Vehicle')
    time_text = ax.text(0.02, 0.98, '', transform=ax.transAxes, 
                       verticalalignment='top', fontsize=12,
                       bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    ax.legend(loc='upper right')
    ax.set_xlabel('X (grid cells)')
    ax.set_ylabel('Y (grid cells)')
    ax.invert_yaxis()
    
    traj_x_hist = []
    traj_y_hist = []
    
    def init():
        traj_line.set_data([], [])
        vehicle_patch.set_data([], [])
        time_text.set_text('')
        return traj_line, vehicle_patch, time_text
    
    def animate(frame):
        i = frame * frame_skip
        if i >= len(trajectory):
            i = len(trajectory) - 1
        
        y, x, theta, t = trajectory[i]
        
        # Update trajectory history
        traj_x_hist.append(x / minor_grid_size)
        traj_y_hist.append(y / minor_grid_size)
        traj_line.set_data(traj_x_hist, traj_y_hist)
        
        # Update vehicle footprint
        corners = firetruck.calc_boundary((y, x, theta), scale=1)
        corners.append(corners[0])
        corner_x = [c[0] / minor_grid_size for c in corners]
        corner_y = [c[1] / minor_grid_size for c in corners]
        vehicle_patch.set_data(corner_x, corner_y)
        
        # Update time
        time_text.set_text(f'Time: {t:.2f}s')
        
        return traj_line, vehicle_patch, time_text
    
    num_frames = len(trajectory) // frame_skip
    anim = animation.FuncAnimation(fig, animate, init_func=init,
                                  frames=num_frames, interval=50, 
                                  blit=True, repeat=True)
    
    anim.save('trajectory_animation.gif', writer='pillow', fps=20)
    print("Saved trajectory_animation.gif")
    plt.close()

def visualize_roadmap_distribution(self, binary_grid, minor_grid_size=0.2):
    """
    Heatmap showing density of randomly sampled PRM nodes
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8))
    
    # Left: All nodes as scatter plot
    ax1.imshow(binary_grid, cmap='binary', origin='lower', alpha=0.3)
    
    if len(self.nodes) > 2:  # Exclude start/goal if present
        nodes_to_plot = self.nodes[:-2] if len(self.nodes) > 2 else self.nodes
        node_y = [n[0] / minor_grid_size for n in nodes_to_plot]
        node_x = [n[1] / minor_grid_size for n in nodes_to_plot]
        
        ax1.scatter(node_x, node_y, c='blue', s=5, alpha=0.5)
        ax1.set_title(f'PRM Node Distribution\n({len(nodes_to_plot)} samples)')
    
    ax1.set_xlabel('X (grid cells)')
    ax1.set_ylabel('Y (grid cells)')
    ax1.invert_yaxis()
    
    # Right: Density heatmap
    density_grid = np.zeros_like(binary_grid, dtype=float)
    
    for y, x, theta in self.nodes[:-2] if len(self.nodes) > 2 else self.nodes:
        grid_y = int(round(y / minor_grid_size))
        grid_x = int(round(x / minor_grid_size))
        
        if 0 <= grid_y < density_grid.shape[0] and 0 <= grid_x < density_grid.shape[1]:
            density_grid[grid_y, grid_x] += 1
    
    im = ax2.imshow(density_grid, cmap='hot', origin='lower', interpolation='gaussian')
    ax2.contour(binary_grid, levels=[0.5], colors='cyan', linewidths=1)
    plt.colorbar(im, ax=ax2, label='Node density')
    
    ax2.set_title('Sampling Density Heatmap')
    ax2.set_xlabel('X (grid cells)')
    ax2.set_ylabel('Y (grid cells)')
    ax2.invert_yaxis()
    
    plt.tight_layout()
    plt.savefig('prm_distribution.png', dpi=150)
    print("Saved prm_distribution.png")
    plt.show()











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
safety_buffer = 0.2
robot_radius = int(major_grid_size*robot_width*(1+safety_buffer))

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



# # Usage
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

# goal_y_meters = 190
# goal_x_meters = 190
# goal_angle = np.deg2rad(45)

# goal_y_meters = 190
# goal_x_meters = 190
# goal_angle = np.deg2rad(45)

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
    minor_grid_size=minor_grid_size,
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
    ax.imshow(dist_cost_grid_up, cmap='plasma', origin='lower', vmin=0, vmax=1, alpha=0.7)
    
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
    plt.cla()
    
    print(f"\n{'='*50}")
    print("PATH FOUND - Starting Simulation")
    print(f"{'='*50}\n")
    
    # Simulate truck following the path
    trajectory, commands = simulate_path_following(
        path=path,
        truck=firetruck,
        binary_grid=binary_grid,
        minor_grid_size=0.2,
        dt=0.01
    )
    
    # Check for collisions in simulated trajectory
    is_valid, collision_indices = check_trajectory_collisions(
        trajectory=trajectory,
        binary_grid=obstacle_grid,
        firetruck=firetruck,
        minor_grid_size=0.2
    )
    
    if is_valid:
        print("✓ Trajectory is collision-free!")
    else:
        print(f"✗ Warning: {len(collision_indices)} collision timesteps detected!")
        print(f"  First collision at t={trajectory[collision_indices[0]][3]:.2f}s")
    
    # Visualize results
    visualize_trajectory(
        path=path,
        trajectory=trajectory,
        binary_grid=binary_grid,
        dist_cost_grid=dist_cost_grid,
        obstacle_grid=obstacle_grid,
        firetruck=firetruck,
        minor_grid_size=0.2
    )
    
    # Calculate path statistics
    total_distance = sum(
        np.sqrt((trajectory[i+1][0] - trajectory[i][0])**2 + 
                (trajectory[i+1][1] - trajectory[i][1])**2)
        for i in range(len(trajectory) - 1)
    )
    
    final_time = trajectory[-1][3]
    avg_speed = total_distance / final_time if final_time > 0 else 0
    
    print(f"\n{'='*50}")
    print("TRAJECTORY STATISTICS")
    print(f"{'='*50}")
    print(f"Total distance traveled: {total_distance:.2f}m")
    print(f"Total time: {final_time:.2f}s")
    print(f"Average speed: {avg_speed:.2f}m/s")
    print(f"Number of waypoints: {len(path)}")
    print(f"Number of timesteps: {len(trajectory)}")
    print(f"Timestep size: 0.05s")
    
    # Calculate error from planned path
    path_endpoints_dist = np.sqrt(
        (path[-1][0] - trajectory[-1][0])**2 + 
        (path[-1][1] - trajectory[-1][1])**2
    )
    print(f"Final position error: {path_endpoints_dist:.3f}m")

    # animate_trajectory(trajectory, binary_grid, firetruck, minor_grid_size=0.2)
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

    