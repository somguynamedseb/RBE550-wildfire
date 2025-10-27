import numpy as np
from matplotlib import pyplot as plt
from robots import square_bot
import random
from shapely.geometry import Polygon
from shapely import contains_xy
from time import sleep
import os 
import glob
from PIL import Image
import re

MAX_Y = 36
MAX_X = 36
minor_grid_size = 0.1
major_grid_size = 3 
scale = int(major_grid_size/minor_grid_size)
dt = 0.05 # seconds
global_time = 0.0 #time starts at 0
percent_fill = 0.10
print(f"scale {scale}")

## INITIAL POSITION SETUP AND TARGET POSITION SETUP
target = (32,26) 
bot = square_bot((2,1),0,scale)
random.seed(bot.test_seed)

values = random.sample(range(MAX_Y*MAX_X), MAX_Y*MAX_X)
max_segments = 20

dirs = [(-1,-1),(-1,0),(-1,1), #actual directions
            (0,-1),      (0,1),
            (1,-1),(1,0),(1,1)] 

colors = [ #colors for segment visualization by direction
    (0.90, 0.10, 0.10),  # Red
    (0.10, 0.60, 0.90),  # Blue
    (0.10, 0.80, 0.20),  # Green
    (0.95, 0.75, 0.10),  # Yellow/Gold
    (0.60, 0.20, 0.80),  # Purple
    (0.00, 0.70, 0.70),  # Teal
    (0.90, 0.40, 0.00),  # Orange
    (0.40, 0.40, 0.40)   # Gray 
]

def check_collision(bot, obstacle_grid):
    pos = bot.get_current_pos()
    obstacles = [tuple(idx) for idx in np.argwhere(obstacle_grid == 0)]
    # --- Compute robot boundary ---
    buffer_pts = bot.calc_boundry((pos[0],pos[1],pos[2]), buffer=0.3,scale = scale) 
    
    
    # --- Determine grid bounds ---
    rect_polygon = Polygon(buffer_pts)
    min_y, min_x, max_y, max_x = rect_polygon.bounds

    # check for ouside grid size
    min_y_clip = (min_y < 0)
    min_x_clip = (min_x < 0)
    max_y_clip = (max_y > obstacle_grid.shape[0])
    max_x_clip = (max_x > obstacle_grid.shape[1])

    y_coords = np.arange(int(min_y), int(max_y))
    x_coords = np.arange(int(min_x), int(max_x))
    yy,xx = np.meshgrid(y_coords, x_coords)
    points = np.column_stack([yy.ravel(), xx.ravel()])


    # --- Vectorized point-in-polygon check ---
    mask = contains_xy(rect_polygon, points[:, 0], points[:, 1])
    inside_points = points[mask]
    overlapping_pts = []
    for pt in inside_points:
        y,x = pt
        if obstacle_grid[x][y] == 0:
            overlapping_pts.append(pt)
    overlap_check = len(overlapping_pts)>0
    
    return (min_y_clip and min_x_clip and max_y_clip and max_x_clip and overlap_check)

def motion_occupency(bot, pos_list, obstacle_grid,scaled = True):
    # Create occupancy grid once
    occupency_grid = np.zeros_like(obstacle_grid)
    for i in range(len(pos_list)):
        pos = pos_list[i]
        # --- Compute robot boundary ---
        if scaled:
            buffer_pts = bot.calc_boundry((pos[0]/scale,pos[1]/scale,pos[2]), buffer=0.3,scale = scale) 
        else:
            buffer_pts = bot.calc_boundry((pos[0],pos[1],pos[2]), buffer=0.3,scale = scale) 
        rect_polygon = Polygon(buffer_pts)
        # --- Determine grid bounds ---
        min_y, min_x, max_y, max_x = rect_polygon.bounds

        # Clip to grid size
        min_y = max(min_y, 0)
        min_x = max(min_x, 0)
        max_y = min(max_y, obstacle_grid.shape[0])
        max_x = min(max_x, obstacle_grid.shape[1])

        y_coords = np.arange(int(min_y), int(max_y))
        x_coords = np.arange(int(min_x), int(max_x))
        yy,xx = np.meshgrid(y_coords, x_coords)
        points = np.column_stack([yy.ravel(), xx.ravel()])

        if len(points) == 0:
            continue

        # --- Vectorized point-in-polygon check ---
        mask = contains_xy(rect_polygon, points[:, 0], points[:, 1])
        inside_points = points[mask]
        if len(inside_points) == 0:
            continue

        indices = np.floor(inside_points).astype(int)
        valid = (
            (indices[:, 0] >= 0) & (indices[:, 0] < occupency_grid.shape[0]) &
            (indices[:, 1] >= 0) & (indices[:, 1] < occupency_grid.shape[1])
        )
        occupency_grid[indices[valid, 1], indices[valid, 0]] = 1
        # break
    
    return np.array(occupency_grid,np.uint8)

def show_cost_heatmap(cost_grid,filename,start = None,goal = None):
    """
    Displays a heatmap of the cost grid.
    
    Parameters:
    - cost_grid: 2D NumPy array with integer cost values (e.g., from BFS or A*)
    - start: optional tuple (row, col) to mark start location
    - goal: optional tuple (row, col) to mark goal location
    - cmap: matplotlib colormap
    - title: title of the plot
    """
    fig2, ax = plt.subplots()

    # Mask unreachable (-1) cells
    masked_grid = np.ma.masked_where(cost_grid < 0, cost_grid)
    
    # Show heatmap
    heatmap = plt.imshow(masked_grid, cmap='plasma', origin='upper')
    plt.colorbar(heatmap, label='Cost (steps from start)')
    if start:plt.scatter(start[1],start[0], color='b',s=10)
    if goal:plt.scatter(goal[1],goal[0], color='g',s=10)
    plt.xlim(0, MAX_X*scale)    
    plt.ylim(0, MAX_Y*scale)
    plt.xticks(np.linspace(0, MAX_X*scale, MAX_X+1),labels=range(0, MAX_X+1))
    plt.yticks(np.linspace(0, MAX_Y*scale, MAX_Y+1),labels=range(0, MAX_Y+1))   
    plt.axis('equal')
    ax.set_aspect('equal', adjustable='box') 
    plt.gca().invert_yaxis()
    plt.gca().set_aspect('equal')

    plt.title('Cost Heatmap')
    plt.tight_layout()
    plt.savefig(filename)
    plt.close(fig2)
    # plt.show()

def segment_path(path):
    #(Dy,Dx)
    # orientation =   [135,90,45,
    #                 180,    0,
    #                 225,270,315]
    orientation =   [225,270,315,
                    180,    0,
                    135,90,45]
    dirs = [(-1,-1),(-1,0),(-1,1), #actual directions
            (0,-1),      (0,1),
            (1,-1),(1,0),(1,1)] 
    d_sign = [(-1,-1),(-2,0),(-1,1), #Delta 1->3  that will correlate with dirs
            (0,-2),      (0,2),
            (1,-1),(2,0),(1,1)] 
    segments = []
    path_index = 0
    current_dir = None
    for i in range(path_index,len(path)-2):
        pose1 = path[i]
        pose2 = path[i+1]
        pose3 = path[i+2]
        d_pose = np.subtract(pose3,pose1)
        try: #will fail on initial None case
            all(current_dir) #intended to cause except case to occur
            if all(d_pose != current_dir):
                dir_i = d_sign.index(tuple(current_dir)) #set from privious direction
                ang = np.deg2rad(orientation[dir_i])
                path_w_ang = [(x,y,ang) for x,y in [pos for pos in path[path_index:i]]]
                scaled_path_w_ang = [(x/scale,y/scale,ang) for x,y in [pos for pos in path[path_index:i]]]
                segments.append((path_w_ang,dir_i,scaled_path_w_ang))
                current_dir = d_pose
                path_index = i
            if i == len(path)-3:
                dir_i = d_sign.index(tuple(current_dir)) #set from privious direction
                ang = np.deg2rad(orientation[dir_i])
                path_w_ang = [(x,y,ang) for x,y in [pos for pos in path[path_index:i]]]
                scaled_path_w_ang = [(x/scale,y/scale,ang) for x,y in [pos for pos in path[path_index:i]]]
                segments.append((path_w_ang,dir_i,scaled_path_w_ang))
                current_dir = d_pose
                path_index = i

                # print(f"current_dir : {current_dir} path_index : {path_index}")
                continue
        except: #intitial setting of current dir
            if current_dir != None:
                raise ValueError(f"current_dir should be None if its here but its not, instead its {current_dir}")
            current_dir = d_pose
            # print(f"current_dir : {current_dir}")
            
    print(f"Number of Segments : {len(segments)}")
    return segments

def clear_folders(directory:list):
    for dir in directory:
        png_files = glob.glob(os.path.join(dir, "*.png"))
        for f in png_files:
            try:
                os.remove(f)
                # print(f"Deleted: {f}")
            except Exception as e:
                print(f"Failed to delete {f}: {e}")
        print(f"cleared: {dir}")

def generate_animation():
    # Folder containing your images
    folder = "map_frames"

    # Collect only valid image files
    files = [f for f in os.listdir(folder) if f.endswith(('.png', '.jpg', '.jpeg'))]

    # Sort by numeric index in the filename
    files.sort(key=lambda f: int(re.search(r'\d+', f).group()))

    # Load images
    frames = [Image.open(os.path.join(folder, f)) for f in files]

    # Save as GIF
    output_path = "map_animation.gif"
    frames[0].save(
        output_path,
        save_all=True,
        append_images=frames[1:],
        duration=50,   
        loop=0          # 0 = infinite loop
    )

    print(f"GIF saved as {output_path}")

clear_folders([r"map_frames",r"cost_frames"])

# all postions are (y(m),x(m),ang(rad)) #origin at top left
## SETUP OBSTACLES AND ENVIORNMENT
obstacle_grid = np.ones((int(MAX_X),int(MAX_Y)),dtype=np.uint8)
counter = 0
for x in range(MAX_X):
    for y in range(MAX_Y):
        if values[counter] <= int((MAX_Y*MAX_X)*percent_fill):
            obstacle_grid[y][x] = 0
        counter+=1

obstacle_grid = np.kron(obstacle_grid, np.ones((scale, scale), dtype=np.int8))
occupancy_cmap = plt.cm.Blues.copy()
occupancy_cmap.set_under(alpha=0)

#INITIATE GRAPH
fig, ax = plt.subplots()

frame_index = 0
recalc_index = 0
path_flag = True
segments = None
path_occupency_grid = None

while True:
    if path_flag:
        ax.clear()
        plt.xlim(0, MAX_X*scale)    
        plt.ylim(0, MAX_Y*scale)
        plt.xlabel("X axis (Meters)")
        plt.xlabel("Y axis (Meters)")
        plt.xticks(np.linspace(0, MAX_X*scale, MAX_X+1),labels=range(0, MAX_X+1))
        plt.yticks(np.linspace(0, MAX_Y*scale, MAX_Y+1),labels=range(0, MAX_Y+1))   
        plt.axis('equal')
        ax.set_aspect('equal', adjustable='box') 
        plt.gca().invert_yaxis()
        # GENERATE COST GRID
        cost_grid,start,goal = bot.calc_cost_grid(obstacle_grid,bot.get_current_pos(),target)
        show_cost_heatmap(cost_grid,f"cost_frames/cost_grid_{recalc_index}.png",start=start,goal=goal)
        recalc_index += 1
        # PATHFIND TO TARGET
        path = bot.pathfind(bot.get_current_pos(),target,cost_grid)

        # SEGMENT PATH
        segments = segment_path(path)
        path_segment,dir_segment,scaled_segment = segments[0]
        
        # GENERATE COMMANDS FROM SEGMENT
        commands,pos_estimates = bot.calc_move_cmd(scaled_segment[-1],global_time)
        bot.import_commands(commands)
        
        # GENERATE OCCUPENCY FROM COMMANDS
        path_occupency_grid = motion_occupency(bot,pos_estimates,obstacle_grid,scaled = False)

        
        path_flag = False
        ax.imshow(obstacle_grid,cmap="gray",interpolation="nearest")
        ax.imshow(path_occupency_grid,cmap=occupancy_cmap,interpolation="nearest",vmin=0.1,alpha=0.8)
        print("path updated")
        
    
    
    # SHOW BOT POSITION
    pts = bot.calc_boundry(bot.get_current_pos(),scale = scale)
    y_pts = [pt[0] for pt in pts]
    x_pts = [pt[1] for pt in pts]
    y_pts.append(y_pts[0])
    x_pts.append(x_pts[0])
    ax.plot(y_pts,x_pts, 'b-',linewidth = 1) #bot shape
    for i,segment in enumerate(segments):
        path_w_ang,dir_i,scaled_segment = segment
        a = 0.75 if i ==0 else 0.1
        ax.scatter([x for y,x,ang in path_w_ang],[y for y,x,ang in path_w_ang], color=colors[dir_i],s=5,alpha=a) #pathfinding path
    ax.scatter(bot.x*scale,bot.y*scale, color='red',s=20)  # bots center of rotation
    ax.scatter(goal[1],goal[0], color='r',s=20) # goal point

    
    plt.savefig(f"map_frames/map_frame_{frame_index}.png")
    frame_index +=1
    # print(frame_index)

    plt.draw()
    for artist in list(ax.lines) + list(ax.collections):
        artist.remove()
        
    
    #INITIATE MOTIONS 
    path_flag = bot.timestep(global_time)
    # CHECK FOR COLLISIONS
    collide_check = check_collision(bot,obstacle_grid)
    if collide_check:print("collision!!!!");break
    
    #DETERMINE WHEN TARGET REACHED TO END PROCESS
    pos = bot.get_current_pos()
    dist = np.sqrt(abs((pos[0]-target[0])**2+(pos[1]-target[1])**2))
    if dist < 0.1:
        print(f"Target Position reached")
        print(f"Current Position:{pos}")
        print(f"Offset by {dist:.2f} meters")
        generate_animation()
        break




