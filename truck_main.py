import numpy as np
from matplotlib import pyplot as plt
import random
from shapely.geometry import Polygon
from shapely import contains_xy
from time import sleep
import os 
import glob
from PIL import Image
import re

from robots import *

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
target = (20,22,0) 
bot = truck_bot((13,4),np.deg2rad(90),scale)
trail = trailer_bot((bot.y-5,4),np.deg2rad(90),scale)
random.seed(128)

values = random.sample(range(MAX_Y*MAX_X), MAX_Y*MAX_X)
max_segments = 20


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


# SHOW BOT POSITION
ax.clear()
plt.xlim(0, MAX_X*scale)    
plt.ylim(0, MAX_Y*scale)
plt.xlabel("X axis (Meters)")
plt.xlabel("Y axis (Meters)")
plt.xticks(np.linspace(0, MAX_X*scale, MAX_X+1),labels=range(0, MAX_X+1))
plt.yticks(np.linspace(0, MAX_Y*scale, MAX_Y+1),labels=range(0, MAX_Y+1))   
ax.imshow(obstacle_grid,cmap="gray",interpolation="nearest")
plt.axis('equal')
ax.set_aspect('equal', adjustable='box') 
plt.gca().invert_yaxis()


# SHOW TRUCK POS
pts = bot.calc_boundry(bot.get_current_pos(),scale = scale)
y_pts = [pt[0] for pt in pts]
x_pts = [pt[1] for pt in pts]
y_pts.append(y_pts[0])
x_pts.append(x_pts[0])
wheel_pts = bot.calc_wheel_pts(bot.get_current_pos(),scale = scale)
wheel_y_pts = [pt[0] for pt in wheel_pts]
wheel_x_pts = [pt[1] for pt in wheel_pts]
ax.plot(y_pts,x_pts, 'b-',linewidth = 1) #bot shape
ax.scatter(wheel_y_pts,wheel_x_pts, color='g',s=20) #bot shape
ax.scatter(bot.x*scale,bot.y*scale, color='red',s=20)  # bots center of rotation

# SHOW TRAILER POS
pts = trail.calc_boundry(trail.get_current_pos(),scale = scale)
y_pts = [pt[0] for pt in pts]
x_pts = [pt[1] for pt in pts]
y_pts.append(y_pts[0])
x_pts.append(x_pts[0])
wheel_pts = trail.calc_wheel_pts(trail.get_current_pos(),scale = scale)
wheel_y_pts = [pt[0] for pt in wheel_pts]
wheel_x_pts = [pt[1] for pt in wheel_pts]
ax.plot(y_pts,x_pts, 'b-',linewidth = 1) #bot shape
ax.scatter(wheel_y_pts,wheel_x_pts, color='g',s=20) #bot shape
ax.scatter(trail.x*scale,trail.y*scale, color='red',s=20)  # bots center of rotation



ax.scatter(target[1]*scale,target[0]*scale, color='k',s=20) # goal point

plt.savefig("truck_setup.png")

ax.cla()

fig, ax = plt.subplots()
test_vels = [1.5,1,0.5,1,0.75,1,1 ]
test_angs = [0, 15,-20,0,-10,10,0]
test_times = [100,150,50,100,50,300,100]
frame_i = 0
for i in range(len(test_vels)):
    vel = test_vels[i]
    ang = test_angs[i]
    time = test_times[i]
    for t in range(time):
        new_pos_bot = bot.update_pos(bot.get_current_pos(),vel,ang)
        bot.y = new_pos_bot[0]
        bot.x = new_pos_bot[1]
        bot.ang = new_pos_bot[2]
        
        new_pos_trail = trail.update_pos(new_pos_bot,vel)
        trail.y = new_pos_trail[0]
        trail.x = new_pos_trail[1]
        trail.ang = new_pos_trail[2]
        
        ax.cla()
        plt.xlim(0, MAX_X*scale)    
        plt.ylim(0, MAX_Y*scale)
        plt.xlabel("X axis (Meters)")
        plt.xlabel("Y axis (Meters)")
        plt.xticks(np.linspace(0, MAX_X*scale, MAX_X+1),labels=range(0, MAX_X+1))
        plt.yticks(np.linspace(0, MAX_Y*scale, MAX_Y+1),labels=range(0, MAX_Y+1))   
        ax.imshow(obstacle_grid,cmap="gray",interpolation="nearest")
        plt.axis('equal')
        ax.set_aspect('equal', adjustable='box') 
        plt.gca().invert_yaxis()


        # SHOW TRUCK POS
        pts = bot.calc_boundry(bot.get_current_pos(),scale = scale)
        y_pts = [pt[0] for pt in pts]
        x_pts = [pt[1] for pt in pts]
        y_pts.append(y_pts[0])
        x_pts.append(x_pts[0])
        wheel_pts = bot.calc_wheel_pts(bot.get_current_pos(),scale = scale)
        wheel_y_pts = [pt[0] for pt in wheel_pts]
        wheel_x_pts = [pt[1] for pt in wheel_pts]
        ax.plot(y_pts,x_pts, 'b-',linewidth = 1) #bot shape
        ax.scatter(wheel_y_pts,wheel_x_pts, color='g',s=20) #bot shape
        ax.scatter(bot.x*scale,bot.y*scale, color='red',s=20)  # bots center of rotation

        # SHOW TRAILER POS
        pts = trail.calc_boundry(trail.get_current_pos(),scale = scale)
        y_pts = [pt[0] for pt in pts]
        x_pts = [pt[1] for pt in pts]
        y_pts.append(y_pts[0])
        x_pts.append(x_pts[0])
        wheel_pts = trail.calc_wheel_pts(trail.get_current_pos(),scale = scale)
        wheel_y_pts = [pt[0] for pt in wheel_pts]
        wheel_x_pts = [pt[1] for pt in wheel_pts]
        ax.plot(y_pts,x_pts, 'b-',linewidth = 1) #bot shape
        ax.scatter(wheel_y_pts,wheel_x_pts, color='g',s=20) #bot shape
        ax.scatter(trail.x*scale,trail.y*scale, color='red',s=20) 
        
        ax.scatter(target[1]*scale,target[0]*scale, color='k',s=20) # goal point
        plt.savefig(f"map_frames/truck_frame_{frame_i}")
        frame_i += 1
        # break
generate_animation()
print("animation and path complete")