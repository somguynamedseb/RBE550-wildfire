import numpy as np
from numpy import pi,cos,sin
from collections import deque
import heapq
import time as t
from scipy.ndimage import binary_dilation
import copy
from skimage.draw import polygon as draw_polygon
from shapely.geometry import Polygon, Point
from shapely.ops import unary_union
from shapely import contains_xy
import numpy as np
import heapq
from scipy.spatial import cKDTree
from matplotlib.path import Path

class square_bot:
    def __init__(self,pos:tuple[float,float],angle:float,scale:int):
        # units in meters and seconds
        self.WIDTH = 0.57
        self.LENGTH = 0.7
        self.scale = scale
        self.test_seed = 20

        #vectors from wheels
        self.MAX_ACC = 1 # meters per second^2 
        self.MAX_VEL = 5 # meters per second
        self.MAX_ROT_ACC = np.arctan2(self.MAX_ACC,self.WIDTH/2)
        self.MAX_ROT_VEL = np.arctan2(self.MAX_VEL,self.WIDTH/2)
        
        self.y = pos[0]
        self.x = pos[1]
        self.ang = angle
        
        self.R_wheel_vel = 0 #linear not rotationally
        self.L_wheel_vel = 0 #linear not rotationally
        self.R_wheel_acc = 0 #linear not rotationally
        self.L_wheel_acc = 0 #linear not rotationally
        self.ang_vel = 0
        self.ang_acc = 0
        
        self.boundry_offsets = [(self.LENGTH-self.WIDTH/2,-self.WIDTH/2),(self.LENGTH-self.WIDTH/2,self.WIDTH/2),(-self.WIDTH/2,-self.WIDTH/2),(-self.WIDTH/2,self.WIDTH/2)] #FR,FL,RR,RL
        self.boundry = self.calc_boundry((self.x,self.y,self.ang),scale = self.scale)
        
        
        # Compute inflation radius based on robot boundary
        self.c_space = -1
        for x, y in self.boundry_offsets:
            tmp_c = int(np.sqrt(x**2 + y**2) * self.scale) + 1
            if tmp_c >  self.c_space:
                self.c_space = tmp_c
        print(f"c_space: { self.c_space}")
        
        self.target_pos = []
        self.command_list = [] 
        self.command_history = []

    def calc_boundry(self,pos,buffer:float = 0.0,scale = 1): 
        # at a set position return the 4 corners for where the robot is at that timestep
        y,x,ang = pos
        y = y*scale
        x = x*scale
        # print(y,x,ang)
        x_off,y_off = np.array(self.boundry_offsets[0]) * (1+buffer) * scale
        FR = (x+(x_off*cos(ang))-(y_off*sin(ang)), y+(x_off*sin(ang))+(y_off*cos(ang)))
        x_off,y_off = np.array(self.boundry_offsets[1]) * (1+buffer) * scale
        FL = (x+(x_off*cos(ang))-(y_off*sin(ang)), y+(x_off*sin(ang))+(y_off*cos(ang)))
        x_off,y_off = np.array(self.boundry_offsets[2]) * (1+buffer) * scale
        RR = (x+(x_off*cos(ang))-(y_off*sin(ang)), y+(x_off*sin(ang))+(y_off*cos(ang)))
        x_off,y_off = np.array(self.boundry_offsets[3]) * (1+buffer) * scale
        RL = (x+(x_off*cos(ang))-(y_off*sin(ang)), y+(x_off*sin(ang))+(y_off*cos(ang)))
        # print([FR,RR,RL,FL])
        return [FR,RR,RL,FL]

    def get_current_pos(self):
        return (self.y,self.x,self.ang)

    def calc_cost_grid(self, grid, initial_pos, target_pos):
        from collections import deque
        import numpy as np
        import time as t
        from scipy.ndimage import binary_dilation

        def inflate_obstacles(grid, inflation_radius):
            struct = np.ones((2*inflation_radius+1, 2*inflation_radius+1), dtype=bool)
            obstacle_mask = (grid == 0)
            inflated_mask = binary_dilation(obstacle_mask, structure=struct)
            valid_mask = ~inflated_mask
            return valid_mask.astype(int)  # 1 = free, 0 = obstacle

        def to_index(pos):
            x = int(pos[1] * self.scale)
            y = int(pos[0] * self.scale)
            return y, x  # (row, col)

        def edge_cspace(grid,buffer):
            grid[:buffer, :] = -1        
            grid[-buffer:, :] = -1         
            grid[:, :buffer] = -1        
            grid[:, -buffer:] = -1         
            return grid
            
        start_time = t.time()
        shape = grid.shape

        

        grid_with_cspace = inflate_obstacles(grid, int(self.c_space * 1.5))
        valid_mask = (grid_with_cspace == 1)
        cost_grid = np.full(shape, -1, dtype=float)

        start = to_index(initial_pos)
        goal = to_index(target_pos)

        # Only 4-way (Manhattan) movement
        directions = [(0, 1), (-1, 0), (1, 0), (0, -1)]

        queue = deque()
        queue.append((start[0], start[1], 0.0))
        cost_grid[start] = 0

        while queue:
            i, j, cost = queue.popleft()

            for dx, dy in directions:
                ni, nj = i + dx, j + dy
                if 0 <= ni < shape[0] and 0 <= nj < shape[1]:
                    if valid_mask[ni, nj] and cost_grid[ni, nj] == -1:
                        new_cost = cost + 1  # uniform cost for Manhattan movement
                        cost_grid[ni, nj] = new_cost
                        queue.append((ni, nj, new_cost))

        # Optional normalization (if you want to visualize)
        mask = (cost_grid == -1)
        valid_values = cost_grid[~mask]
        if valid_values.size > 0:
            min_cost, max_cost = valid_values.min(), valid_values.max()
            if max_cost > min_cost:
                cost_grid[~mask] = 100 * (cost_grid[~mask] - min_cost) / (max_cost - min_cost)
            else:
                cost_grid[~mask] = 0

        cost_grid = edge_cspace(cost_grid,self.c_space)
        print(f"Base cost grid calculated in {t.time() - start_time:.3f} seconds")
        return cost_grid, start, goal

    def pathfind(self, initial_pos, target_pos, cost_grid, straight_weight=0.9): 
        """
        A* pathfinding with 8-direction movement and straight-line preference.

        Parameters:
            initial_pos: (x, y) starting position
            target_pos: (x, y) goal position
            cost_grid: 2D numpy array of traversal costs (-1 = blocked)
            straight_weight: 0.0–1.0, strength of preference for continuing straight
                            (0 = no preference, 1 = very strong straight bias)
        """


        # --- Heuristic: Octile distance (works well for 8-way movement)
        def heuristic(a, b):
            dx = abs(a[0] - b[0])
            dy = abs(a[1] - b[1])
            return (dx + dy) + (np.sqrt(2) - 2) * min(dx, dy)

        def to_index(pos):
            y = int(pos[0] * self.scale)
            x = int(pos[1] * self.scale)
            return y,x  # (col,row)

        start = to_index(initial_pos)
        goal = to_index(target_pos)
        shape = cost_grid.shape

        # # --- 8 cardinal + diagonal directions
        # directions = [
        #     (-1, 0), (1, 0), (0, -1), (0, 1),    # N, S, W, E
        #     (-1, -1), (-1, 1), (1, -1), (1, 1)   # NW, NE, SW, SE
        # ]
        # direction_costs = [1, 1, 1, 1, np.sqrt(2), np.sqrt(2), np.sqrt(2), np.sqrt(2)]
        
        
        # --- 4 cardinal
        directions = [
            (-1, 0), (1, 0), (0, -1), (0, 1),    # N, S, W, E

        ]
        direction_costs = [1, 1, 1, 1]

        # --- Initialize A* structures
        open_set = []
        heapq.heappush(open_set, (heuristic(start, goal), 0, start, None))  # (f, g, pos, last_dir)
        came_from = {}
        g_score = {start: 0}

        while open_set:
            _, current_g, current, last_dir = heapq.heappop(open_set)

            if current == goal:
                # --- Reconstruct path
                path = []
                while current in came_from:
                    path.append(current)
                    current = came_from[current]
                path.append(start)
                print("path found")
                return path[::-1]

            for d_idx, (dx, dy) in enumerate(directions):
                nx, ny = current[0] + dx, current[1] + dy

                if 0 <= nx < shape[0] and 0 <= ny < shape[1]:
                    cell_cost = cost_grid[nx, ny]
                    if cell_cost == -1:
                        continue  # obstacle or invalid

                    base_move_cost = direction_costs[d_idx]

                    # --- Straight-line preference
                    if last_dir is not None and d_idx == last_dir:
                        # Reduce cost slightly when continuing straight
                        move_cost = base_move_cost * (1.0 - straight_weight)
                    else:
                        move_cost = base_move_cost

                    tentative_g = current_g + move_cost + cell_cost
                    neighbor = (nx, ny)

                    if neighbor not in g_score or tentative_g < g_score[neighbor]:
                        g_score[neighbor] = tentative_g
                        f_score = tentative_g + heuristic(neighbor, goal)
                        heapq.heappush(open_set, (f_score, tentative_g, neighbor, d_idx))
                        came_from[neighbor] = current

        print("no path found")
        return None

    def calc_move_cmd(self,target,curr_t): 
        # pos = (y,x,ang)
        # main passes in commands for where the bot should go, this will break it up into relevent commands, linear and rotational motion and then return the total location and space reservation for that motion
        # just calculates, does not intiate as collision check has to be confirmed before timestep
        self.target_pos = target
        
        y_tar = self.target_pos[0] 
        x_tar = self.target_pos[1]
            
        ang_tar = self.target_pos[2]
        
        delta_pos_ang = np.arctan2(y_tar-self.y,x_tar-self.x)
        print(f"initial_pos: {self.get_current_pos()}")
        print(f"terget_ pos:{self.target_pos}")
        commands = []
        pos_estimates = []
        motion_start_time = curr_t
        verbose = True
        if ang_tar-self.ang != 0: #initial turn
            motion_cmds,pos_est,delta_t = self.calc_turn((self.y,self.x,self.ang),(self.y,self.x,ang_tar),motion_start_time)
            if verbose:print(f"initial turn calculated to take {delta_t:.2f} seconds")
            if len(commands)>0:
                commands = np.concatenate((commands,motion_cmds))
                pos_estimates = np.concatenate(pos_estimates,pos_est)
            else:
                commands = motion_cmds
                pos_estimates = pos_est
            motion_start_time += delta_t
        if y_tar-self.y!=0 or x_tar-self.x != 0: #linear drive
            motion_cmds,pos_est,delta_t = self.calc_drive((self.y,self.x,ang_tar),self.target_pos,motion_start_time)
            if verbose:print(f"main drive calculated to take {delta_t:.2f} seconds")
            if len(commands)>0:
                commands = np.concatenate((commands,motion_cmds))
                pos_estimates = np.concatenate((pos_estimates,pos_est))
            else:
                commands = motion_cmds
                pos_estimates = pos_est
            motion_start_time += delta_t
        
        return commands,pos_estimates

    def LSPB_calc(self, maxV, maxA, val):
        """
        Compute LSPB motion parameters (trapezoidal or triangular velocity profile).

        Returns:
            (t_total, t_acc, t_const, v_peak)
        """
        sign = 1.0 if val >= 0 else -1.0
        val = abs(val)

        # Time to reach maxV at maxA
        t_acc = maxV / maxA

        # Distance covered during both accel+decel
        d_accel_total = maxA * t_acc**2  # (a * t_acc^2) = 2*(½ a t^2)

        if val <= d_accel_total:
            # --- Triangular profile (never reaches max velocity)
            v_peak = np.sqrt(val * maxA)
            t_acc = v_peak / maxA
            t_const = 0.0
            t_total = 2 * t_acc
        else:
            # --- Trapezoidal profile (hits max velocity)
            d_accel = 0.5 * maxA * t_acc**2  # per phase
            d_const = val - 2 * d_accel
            t_const = d_const / maxV
            v_peak = maxV
            t_total = 2 * t_acc + t_const

        return t_total, t_acc, t_const, sign * v_peak

    def wrap_to_pi(self,angle):
        """Wrap angle to [-pi, pi)."""
        return (angle + pi) % (2.0 * pi) - pi

    def update_pos(self, pos, vL, vR, dt=0.05):
        """
        pos: [x, y, theta]
        vL, vR: linear wheel speeds (m/s)
        dt: timestep
        WIDTH: full track width between wheels
        """
        y ,x , theta = pos
        v = 0.5 * (vR + vL)
        omega = (vR - vL) / self.WIDTH  # ✅ fixed

        eps = 1e-12

        if abs(omega) < eps:
            # Straight motion
            x_new = x + v * np.cos(theta) * dt
            y_new = y + v * np.sin(theta) * dt
            theta_new = theta
        elif abs(v) < eps:
            # Pure rotation
            x_new = x
            y_new = y
            theta_new = theta + omega * dt
        else:
            # General case (circular arc)
            delta_theta = omega * dt
            R = v / omega
            x_new = x + R * (np.sin(theta + delta_theta) - np.sin(theta))
            y_new = y - R * (np.cos(theta + delta_theta) - np.cos(theta))
            theta_new = theta + delta_theta

        theta_new = self.wrap_to_pi(theta_new)
        return [y_new,x_new, theta_new]

    def calc_drive(self, initial_pos,target_pos,curr_t,dt = 0.05):
        # calculates where the bot will be at set time steps, returns its bounding boxes for the desired motion witht timestamps
        tmp_pos = initial_pos
        dist = np.sqrt(abs((initial_pos[0]-target_pos[0])**2+(initial_pos[1]-target_pos[1])**2))
        if dist != 0:
            t_total,t_acc,t_max,maxV = self.LSPB_calc(self.MAX_VEL,self.MAX_ACC,dist)
            motion_vals = []  # [accL, accR, velL, velR, t]
            pos_est = []

            # Initialize
            vL = 0.0
            vR = 0.0
            tmp_t = curr_t
            pos_est.append(tmp_pos)
            motion_vals.append([0, 0, vL, vR, tmp_t])

            # --- Phase 1: Acceleration
            accL = accR = self.MAX_ACC
            t_elapsed = 0.0
            while t_elapsed < t_acc - 1e-9:
                dt_eff = min(dt, t_acc - t_elapsed)
                tmp_t += dt_eff
                vL += accL * dt_eff
                vR += accR * dt_eff
                tmp_pos = self.update_pos(tmp_pos, vL, vR, dt_eff)
                pos_est.append(tmp_pos)
                motion_vals.append([accL, accR, vL, vR, tmp_t])
                t_elapsed += dt_eff

            # --- Phase 2: Constant velocity (if any)
            if t_max > 0:
                accL = accR = 0.0
                t_elapsed = 0.0
                while t_elapsed < t_max - 1e-9:
                    dt_eff = min(dt, t_max - t_elapsed)
                    tmp_t += dt_eff
                    tmp_pos = self.update_pos(tmp_pos, vL, vR, dt_eff)
                    pos_est.append(tmp_pos)
                    motion_vals.append([accL, accR, vL, vR, tmp_t])
                    t_elapsed += dt_eff

            # --- Phase 3: Deceleration
            accL = accR = -self.MAX_ACC
            t_elapsed = 0.0
            while t_elapsed < t_acc - 1e-9:
                dt_eff = min(dt, t_acc - t_elapsed)
                tmp_t += dt_eff
                vL += accL * dt_eff
                vR += accR * dt_eff
                tmp_pos = self.update_pos(tmp_pos, vL, vR, dt_eff)
                pos_est.append(tmp_pos)
                motion_vals.append([accL, accR, vL, vR, tmp_t])
                t_elapsed += dt_eff
            print(f"target_pos : {target_pos}")
            print(f"pos_est : {pos_est[-1]}")
            return motion_vals, pos_est,t_total
        return [-1]

    def calc_turn(self, initial_pos, target_pos, curr_t, dt=0.05):
        d_ang = self.wrap_to_pi(target_pos[2] - initial_pos[2])
        if abs(d_ang) < 1e-12:
            return np.zeros((0,5)), np.array([initial_pos]), curr_t

        # --- LSPB parameters
        t_total, t_acc, t_flat, peak_omega = self.LSPB_calc(
            self.MAX_ROT_VEL, self.MAX_ROT_ACC, d_ang)
        sign = 1.0 if d_ang >= 0 else -1.0
        A = sign * self.MAX_ROT_ACC
        W = peak_omega

        def omega_of(t):
            if t < t_acc:
                return A * t
            elif t < t_acc + t_flat:
                return W
            else:
                return W - A * (t - (t_acc + t_flat))

        def theta_of(t):
            if t < t_acc:
                return 0.5 * A * t**2
            elif t < t_acc + t_flat:
                return 0.5 * A * t_acc**2 + W * (t - t_acc)
            else:
                td = t - (t_acc + t_flat)
                return (0.5 * A * t_acc**2 +
                        W * t_flat + W * td - 0.5 * A * td**2)

        def ang_acc_of(t):
            if t < t_acc:
                return A
            elif t < t_acc + t_flat:
                return 0.0
            else:
                return -A

        pos = [*initial_pos]
        t = 0.0
        t_global = curr_t
        motion_vals, pos_est = [], [pos]

        while t < t_total - 1e-12:
            next_t = min(t + dt, t_total)
            theta_now = theta_of(t)
            theta_next = theta_of(next_t)
            dtheta = theta_next - theta_now
            omega_mid = (omega_of(t) + omega_of(next_t)) * 0.5
            ang_acc_mid = ang_acc_of((t + next_t) * 0.5)

            vR = omega_mid * self.WIDTH/2
            vL = -vR
            accR = ang_acc_mid * self.WIDTH/2
            accL = -accR

            t_global += (next_t - t)
            pos = self.update_pos(pos, vL, vR, next_t - t)
            motion_vals.append([accL, accR, vL, vR, t_global])
            pos_est.append(pos)
            t = next_t

        # ensure final heading matches exactly
        pos_est[-1][2] = initial_pos[2] + d_ang
        return np.array(motion_vals), np.array(pos_est), t_global

    def check_collisions(self): 
        # takes boundary point list 
        raise NotImplementedError

    def import_commands(self,commands):
        self.command_list = np.array(commands)

    def timestep(self,global_time, dt=0.05):
        # updates position based on accelerations from each wheel
        # puts command into history after enacting it
        # command format : [accL,accR,vL,vR,tmp_t]
        # Assumes linear motion or rotating in place for this robot
        accL,accR,vL,vR,tmp_t = self.command_list[0]
        self.R_wheel_acc = accR
        self.L_wheel_acc = accL
        # self.R_wheel_vel += accR * dt #more error on turns from this; not sure why
        # self.L_wheel_vel += accL * dt
        self.R_wheel_vel = vR
        self.L_wheel_vel = vL

        delta_acc = self.R_wheel_acc-self.L_wheel_acc
        
        self.ang_acc = np.arctan2(delta_acc/2,self.WIDTH/2)
        self.ang_vel += self.ang_acc * dt
        
        new_pos = self.update_pos((self.y,self.x,self.ang),self.L_wheel_vel,self.R_wheel_vel)
        self.y = new_pos[0]
        self.x = new_pos[1]
        self.ang = new_pos[2]
        self.command_history.append((copy.copy(self.command_list[0])))
        self.command_list = np.delete(self.command_list,0, axis=0)
        if len(self.command_list)==0: self.R_wheel_vel=0;self.L_wheel_vel=0
        return len(self.command_list)==0

class car_bot:
    def __init__(self,pos:tuple[float,float],angle:float,scale:int):
        # units in meters and seconds
        self.WIDTH = 1.8
        self.LENGTH = 5.2
        self.WHEELBASE = 2.8
        self.scale = scale
        self.test_seed = 10

        #vectors from wheels
        self.MAX_ACC = 1 # meters per second^2 
        self.MAX_VEL = 5 # meters per second
        self.MAX_ROT_ACC = np.arctan2(self.MAX_ACC,self.WIDTH/2)
        self.MAX_ROT_VEL = np.arctan2(self.MAX_VEL,self.WIDTH/2)
        
        self.y = pos[0]
        self.x = pos[1]
        self.ang = angle
        
        self.R_wheel_vel = 0 #linear not rotationally
        self.L_wheel_vel = 0 #linear not rotationally
        self.R_wheel_acc = 0 #linear not rotationally
        self.L_wheel_acc = 0 #linear not rotationally
        self.ang_vel = 0
        self.ang_acc = 0
        
        #based on center point location between back wheels
        self._wheelbase_offset = (self.LENGTH-self.WHEELBASE)/2 #distance between each axle on a centered wheelbase and the front and rear of the car
        self.boundry_offsets = [(self.WHEELBASE+self._wheelbase_offset,self.WIDTH/2),(self.WHEELBASE+self._wheelbase_offset,-self.WIDTH/2),(-self._wheelbase_offset,self.WIDTH/2),(-self._wheelbase_offset,-self.WIDTH/2)] #FR,FL,RR,RL
        self.wheel_points = [(self.WHEELBASE,self.WIDTH/2),(self.WHEELBASE,-self.WIDTH/2),(0,self.WIDTH/2),(0,-self.WIDTH/2)] #FR,FL,RR,RL
        
        print(self.boundry_offsets)
        
        self.target_pos = []
        self.command_list = [] 
        self.command_history = []

    def calc_boundry(self,pos,buffer:float = 0.0,scale = 1): 
        # at a set position return the 4 corners for where the robot is at that timestep
        y,x,ang = pos
        y = y*scale
        x = x*scale
        # print(y,x,ang)
        x_off,y_off = np.array(self.boundry_offsets[0]) * (1+buffer) * scale
        FR = (x+(x_off*cos(ang))-(y_off*sin(ang)), y+(x_off*sin(ang))+(y_off*cos(ang)))
        x_off,y_off = np.array(self.boundry_offsets[1]) * (1+buffer) * scale
        FL = (x+(x_off*cos(ang))-(y_off*sin(ang)), y+(x_off*sin(ang))+(y_off*cos(ang)))
        x_off,y_off = np.array(self.boundry_offsets[2]) * (1+buffer) * scale
        RR = (x+(x_off*cos(ang))-(y_off*sin(ang)), y+(x_off*sin(ang))+(y_off*cos(ang)))
        x_off,y_off = np.array(self.boundry_offsets[3]) * (1+buffer) * scale
        RL = (x+(x_off*cos(ang))-(y_off*sin(ang)), y+(x_off*sin(ang))+(y_off*cos(ang)))
        # print([FR,RR,RL,FL])
        return [FR,RR,RL,FL]

    def calc_wheel_pts(self,pos,buffer:float = 0.0,scale = 1): 
        # at a set position return the 4 corners for where the robot is at that timestep
        y,x,ang = pos
        y = y*scale
        x = x*scale
        # print(y,x,ang)
        x_off,y_off = np.array(self.wheel_points[0]) * (1+buffer) * scale
        FR = (x+(x_off*cos(ang))-(y_off*sin(ang)), y+(x_off*sin(ang))+(y_off*cos(ang)))
        x_off,y_off = np.array(self.wheel_points[1]) * (1+buffer) * scale
        FL = (x+(x_off*cos(ang))-(y_off*sin(ang)), y+(x_off*sin(ang))+(y_off*cos(ang)))
        x_off,y_off = np.array(self.wheel_points[2]) * (1+buffer) * scale
        RR = (x+(x_off*cos(ang))-(y_off*sin(ang)), y+(x_off*sin(ang))+(y_off*cos(ang)))
        x_off,y_off = np.array(self.wheel_points[3]) * (1+buffer) * scale
        RL = (x+(x_off*cos(ang))-(y_off*sin(ang)), y+(x_off*sin(ang))+(y_off*cos(ang)))
        # print([FR,RR,RL,FL])
        return [FR,RR,RL,FL]

    def get_current_pos(self):
        return (self.y,self.x,self.ang)

    def find_path_to_goal(self, goal_pos, obstacle_grid):
        """
        Plan path to goal and populate command_list.
        
        Args:
            goal_pos: (y, x, angle) target pose
            obstacle_grid: 2D numpy array where 0=obstacle, 1=free
            
        Returns:
            bool: True if path found, False otherwise
        """
        planner = AckermannPathfinder(self, obstacle_grid)
        start_pos = (self.y, self.x, self.ang)
        
        path = planner.plan_path(start_pos, goal_pos)
        
        if path is None:
            print("No path found!")
            return False,None
        
        print(f"Path found with {len(path)} waypoints")
        
        # Convert path to commands
        # self._path_to_commands(path)
        return True,path
    
    def wrap_to_pi(self,angle):
        """Wrap angle to [-pi, pi)."""
        return (angle + pi) % (2.0 * pi) - pi

    def update_pos(self, pos, drive_vel, wheel_ang, dt=0.05):
        """
        Update the vehicle pose using a kinematic bicycle (Ackermann) model.
        Reference point: midpoint between the rear wheels (rear axle midpoint).

        Args:
            pos: iterable or array-like (x, y, theta) -- current pose
            drive_vel: linear velocity at the rear axle midpoint (m/s)
            wheel_ang: front-wheel steering angle (radians)
            dt: timestep (s)

        Returns:
            np.ndarray([x_new, y_new, theta_new])
        """
        y, x, theta = float(pos[0]), float(pos[1]), float(pos[2])
        L = float(self.WHEELBASE)

        # threshold to treat small steering as straight-line motion
        eps = 1e-8

        if abs(wheel_ang) < eps:
            # Straight motion
            dx = drive_vel * np.cos(theta) * dt
            dy = drive_vel * np.sin(theta) * dt
            x_new = x + dx
            y_new = y + dy
            theta_new = theta
        else:
            # Turning motion (ICC-based integration)
            # turning radius (signed) and angular velocity
            R = L / np.tan(wheel_ang)        # radius of curvature (m)
            omega = drive_vel / R           # yaw rate (rad/s)
            dtheta = omega * dt

            # ICC update (exact integration over dt)
            x_new = x + R * (np.sin(theta + dtheta) - np.sin(theta))
            y_new = y - R * (np.cos(theta + dtheta) - np.cos(theta))
            theta_new = theta + dtheta

        # normalize theta to [-pi, pi)
        theta_new = self.wrap_to_pi(theta_new)

        return np.array([y_new,x_new, theta_new])

    def calc_drive(self, initial_pos,target_pos,curr_t,dt = 0.05):
        # calculates where the bot will be at set time steps, returns its bounding boxes for the desired motion witht timestamps
        tmp_pos = initial_pos
        dist = np.sqrt(abs((initial_pos[0]-target_pos[0])**2+(initial_pos[1]-target_pos[1])**2))
        if dist != 0:
            t_total,t_acc,t_max,maxV = self.LSPB_calc(self.MAX_VEL,self.MAX_ACC,dist)
            motion_vals = []  # [accL, accR, velL, velR, t]
            pos_est = []

            # Initialize
            vL = 0.0
            vR = 0.0
            tmp_t = curr_t
            pos_est.append(tmp_pos)
            motion_vals.append([0, 0, vL, vR, tmp_t])

            # --- Phase 1: Acceleration
            accL = accR = self.MAX_ACC
            t_elapsed = 0.0
            while t_elapsed < t_acc - 1e-9:
                dt_eff = min(dt, t_acc - t_elapsed)
                tmp_t += dt_eff
                vL += accL * dt_eff
                vR += accR * dt_eff
                tmp_pos = self.update_pos(tmp_pos, vL, vR, dt_eff)
                pos_est.append(tmp_pos)
                motion_vals.append([accL, accR, vL, vR, tmp_t])
                t_elapsed += dt_eff

            # --- Phase 2: Constant velocity (if any)
            if t_max > 0:
                accL = accR = 0.0
                t_elapsed = 0.0
                while t_elapsed < t_max - 1e-9:
                    dt_eff = min(dt, t_max - t_elapsed)
                    tmp_t += dt_eff
                    tmp_pos = self.update_pos(tmp_pos, vL, vR, dt_eff)
                    pos_est.append(tmp_pos)
                    motion_vals.append([accL, accR, vL, vR, tmp_t])
                    t_elapsed += dt_eff

            # --- Phase 3: Deceleration
            accL = accR = -self.MAX_ACC
            t_elapsed = 0.0
            while t_elapsed < t_acc - 1e-9:
                dt_eff = min(dt, t_acc - t_elapsed)
                tmp_t += dt_eff
                vL += accL * dt_eff
                vR += accR * dt_eff
                tmp_pos = self.update_pos(tmp_pos, vL, vR, dt_eff)
                pos_est.append(tmp_pos)
                motion_vals.append([accL, accR, vL, vR, tmp_t])
                t_elapsed += dt_eff
            print(f"target_pos : {target_pos}")
            print(f"pos_est : {pos_est[-1]}")
            return motion_vals, pos_est,t_total
        return [-1]

    def calc_turn(self, initial_pos, target_pos, curr_t, dt=0.05):
        d_ang = self.wrap_to_pi(target_pos[2] - initial_pos[2])
        if abs(d_ang) < 1e-12:
            return np.zeros((0,5)), np.array([initial_pos]), curr_t

        # --- LSPB parameters
        t_total, t_acc, t_flat, peak_omega = self.LSPB_calc(
            self.MAX_ROT_VEL, self.MAX_ROT_ACC, d_ang)
        sign = 1.0 if d_ang >= 0 else -1.0
        A = sign * self.MAX_ROT_ACC
        W = peak_omega

        def omega_of(t):
            if t < t_acc:
                return A * t
            elif t < t_acc + t_flat:
                return W
            else:
                return W - A * (t - (t_acc + t_flat))

        def theta_of(t):
            if t < t_acc:
                return 0.5 * A * t**2
            elif t < t_acc + t_flat:
                return 0.5 * A * t_acc**2 + W * (t - t_acc)
            else:
                td = t - (t_acc + t_flat)
                return (0.5 * A * t_acc**2 +
                        W * t_flat + W * td - 0.5 * A * td**2)

        def ang_acc_of(t):
            if t < t_acc:
                return A
            elif t < t_acc + t_flat:
                return 0.0
            else:
                return -A

        pos = [*initial_pos]
        t = 0.0
        t_global = curr_t
        motion_vals, pos_est = [], [pos]

        while t < t_total - 1e-12:
            next_t = min(t + dt, t_total)
            theta_now = theta_of(t)
            theta_next = theta_of(next_t)
            dtheta = theta_next - theta_now
            omega_mid = (omega_of(t) + omega_of(next_t)) * 0.5
            ang_acc_mid = ang_acc_of((t + next_t) * 0.5)

            vR = omega_mid * self.WIDTH/2
            vL = -vR
            accR = ang_acc_mid * self.WIDTH/2
            accL = -accR

            t_global += (next_t - t)
            pos = self.update_pos(pos, vL, vR, next_t - t)
            motion_vals.append([accL, accR, vL, vR, t_global])
            pos_est.append(pos)
            t = next_t

        # ensure final heading matches exactly
        pos_est[-1][2] = initial_pos[2] + d_ang
        return np.array(motion_vals), np.array(pos_est), t_global

    def check_collisions(self): 
        # takes boundary point list 
        raise NotImplementedError

    def import_commands(self,commands):
        self.command_list = np.array(commands)

    def timestep(self,global_time, dt=0.05):
        # updates position based on accelerations from each wheel
        # puts command into history after enacting it
        # command format : [accL,accR,vL,vR,tmp_t]
        # Assumes linear motion or rotating in place for this robot
        accL,accR,vL,vR,tmp_t = self.command_list[0]
        self.R_wheel_acc = accR
        self.L_wheel_acc = accL
        # self.R_wheel_vel += accR * dt #more error on turns from this; not sure why
        # self.L_wheel_vel += accL * dt
        self.R_wheel_vel = vR
        self.L_wheel_vel = vL

        delta_acc = self.R_wheel_acc-self.L_wheel_acc
        
        self.ang_acc = np.arctan2(delta_acc/2,self.WIDTH/2)
        self.ang_vel += self.ang_acc * dt
        
        new_pos = self.update_pos((self.y,self.x,self.ang),self.L_wheel_vel,self.R_wheel_vel)
        self.y = new_pos[0]
        self.x = new_pos[1]
        self.ang = new_pos[2]
        self.command_history.append((copy.copy(self.command_list[0])))
        self.command_list = np.delete(self.command_list,0, axis=0)
        if len(self.command_list)==0: self.R_wheel_vel=0;self.L_wheel_vel=0
        return len(self.command_list)==0

class truck_bot:
    def __init__(self,pos:tuple[float,float],angle:float,scale:int):
        # units in meters and seconds
        self.WIDTH = 2.0
        self.LENGTH = 5.4
        self.WHEELBASE = 3.4
        self.scale = scale
        self.test_seed = 10

        #vectors from wheels
        self.MAX_ACC = 1 # meters per second^2 
        self.MAX_VEL = 5 # meters per second
        self.MAX_ROT_ACC = np.arctan2(self.MAX_ACC,self.WIDTH/2)
        self.MAX_ROT_VEL = np.arctan2(self.MAX_VEL,self.WIDTH/2)
        
        self.y = pos[0]
        self.x = pos[1]
        self.ang = angle
        
        self.R_wheel_vel = 0 #linear not rotationally
        self.L_wheel_vel = 0 #linear not rotationally
        self.R_wheel_acc = 0 #linear not rotationally
        self.L_wheel_acc = 0 #linear not rotationally
        self.ang_vel = 0
        self.ang_acc = 0
        
        #based on center point location between back wheels
        self._wheelbase_offset = (self.LENGTH-self.WHEELBASE)/2 #distance between each axle on a centered wheelbase and the front and rear of the car
        self.boundry_offsets = [(self.WHEELBASE+self._wheelbase_offset,self.WIDTH/2),(self.WHEELBASE+self._wheelbase_offset,-self.WIDTH/2),(-self._wheelbase_offset,self.WIDTH/2),(-self._wheelbase_offset,-self.WIDTH/2)] #FR,FL,RR,RL
        self.wheel_points = [(self.WHEELBASE,self.WIDTH/2),(self.WHEELBASE,-self.WIDTH/2),(0,self.WIDTH/2),(0,-self.WIDTH/2)] #FR,FL,RR,RL
        
        print(self.boundry_offsets)
        
        self.target_pos = []
        self.command_list = [] 
        self.command_history = []

    def calc_boundry(self,pos,buffer:float = 0.0,scale = 1): 
        # at a set position return the 4 corners for where the robot is at that timestep
        y,x,ang = pos
        y = y*scale
        x = x*scale
        # print(y,x,ang)
        x_off,y_off = np.array(self.boundry_offsets[0]) * (1+buffer) * scale
        FR = (x+(x_off*cos(ang))-(y_off*sin(ang)), y+(x_off*sin(ang))+(y_off*cos(ang)))
        x_off,y_off = np.array(self.boundry_offsets[1]) * (1+buffer) * scale
        FL = (x+(x_off*cos(ang))-(y_off*sin(ang)), y+(x_off*sin(ang))+(y_off*cos(ang)))
        x_off,y_off = np.array(self.boundry_offsets[2]) * (1+buffer) * scale
        RR = (x+(x_off*cos(ang))-(y_off*sin(ang)), y+(x_off*sin(ang))+(y_off*cos(ang)))
        x_off,y_off = np.array(self.boundry_offsets[3]) * (1+buffer) * scale
        RL = (x+(x_off*cos(ang))-(y_off*sin(ang)), y+(x_off*sin(ang))+(y_off*cos(ang)))
        # print([FR,RR,RL,FL])
        return [FR,RR,RL,FL]

    def calc_wheel_pts(self,pos,buffer:float = 0.0,scale = 1): 
        # at a set position return the 4 corners for where the robot is at that timestep
        y,x,ang = pos
        y = y*scale
        x = x*scale
        # print(y,x,ang)
        x_off,y_off = np.array(self.wheel_points[0]) * (1+buffer) * scale
        FR = (x+(x_off*cos(ang))-(y_off*sin(ang)), y+(x_off*sin(ang))+(y_off*cos(ang)))
        x_off,y_off = np.array(self.wheel_points[1]) * (1+buffer) * scale
        FL = (x+(x_off*cos(ang))-(y_off*sin(ang)), y+(x_off*sin(ang))+(y_off*cos(ang)))
        x_off,y_off = np.array(self.wheel_points[2]) * (1+buffer) * scale
        RR = (x+(x_off*cos(ang))-(y_off*sin(ang)), y+(x_off*sin(ang))+(y_off*cos(ang)))
        x_off,y_off = np.array(self.wheel_points[3]) * (1+buffer) * scale
        RL = (x+(x_off*cos(ang))-(y_off*sin(ang)), y+(x_off*sin(ang))+(y_off*cos(ang)))
        # print([FR,RR,RL,FL])
        return [FR,RR,RL,FL]

    def get_current_pos(self):
        return (self.y,self.x,self.ang)

    def wrap_to_pi(self,angle):
        """Wrap angle to [-pi, pi)."""
        return (angle + pi) % (2.0 * pi) - pi

    def update_pos(self, pos, drive_vel, wheel_ang, dt=0.05):
        """
        Update the vehicle pose using a kinematic bicycle (Ackermann) model.
        Reference point: midpoint between the rear wheels (rear axle midpoint).

        Args:
            pos: iterable or array-like (x, y, theta) -- current pose
            drive_vel: linear velocity at the rear axle midpoint (m/s)
            wheel_ang: front-wheel steering angle (radians)
            dt: timestep (s)

        Returns:
            np.ndarray([x_new, y_new, theta_new])
        """
        y, x, theta = float(pos[0]), float(pos[1]), float(pos[2])
        L = float(self.WHEELBASE)

        # threshold to treat small steering as straight-line motion
        eps = 1e-8

        if abs(wheel_ang) < eps:
            # Straight motion
            dx = drive_vel * np.cos(theta) * dt
            dy = drive_vel * np.sin(theta) * dt
            x_new = x + dx
            y_new = y + dy
            theta_new = theta
        else:
            # Turning motion (ICC-based integration)
            # turning radius (signed) and angular velocity
            R = L / np.tan(wheel_ang)        # radius of curvature (m)
            omega = drive_vel / R           # yaw rate (rad/s)
            dtheta = omega * dt

            # ICC update (exact integration over dt)
            x_new = x + R * (np.sin(theta + dtheta) - np.sin(theta))
            y_new = y - R * (np.cos(theta + dtheta) - np.cos(theta))
            theta_new = theta + dtheta

        # normalize theta to [-pi, pi)
        theta_new = self.wrap_to_pi(theta_new)

        return np.array([y_new,x_new, theta_new])

    def find_path_to_goal(self, goal_pos, obstacle_grid):
        """
        Plan path to goal and populate command_list.
        
        Args:
            goal_pos: (y, x, angle) target pose
            obstacle_grid: 2D numpy array where 0=obstacle, 1=free
            
        Returns:
            bool: True if path found, False otherwise
        """
        planner = AckermannPathfinder(self, obstacle_grid)
        start_pos = (self.y, self.x, self.ang)
        
        path = planner.plan_path(start_pos, goal_pos)
        
        if path is None:
            print("No path found!")
            return False,None
        
        print(f"Path found with {len(path)} waypoints")
        
        # Convert path to commands
        # self._path_to_commands(path)
        return True,path

    def calc_drive(self, initial_pos,target_pos,curr_t,dt = 0.05):
        # calculates where the bot will be at set time steps, returns its bounding boxes for the desired motion witht timestamps
        tmp_pos = initial_pos
        dist = np.sqrt(abs((initial_pos[0]-target_pos[0])**2+(initial_pos[1]-target_pos[1])**2))
        if dist != 0:
            t_total,t_acc,t_max,maxV = self.LSPB_calc(self.MAX_VEL,self.MAX_ACC,dist)
            motion_vals = []  # [accL, accR, velL, velR, t]
            pos_est = []

            # Initialize
            vL = 0.0
            vR = 0.0
            tmp_t = curr_t
            pos_est.append(tmp_pos)
            motion_vals.append([0, 0, vL, vR, tmp_t])

            # --- Phase 1: Acceleration
            accL = accR = self.MAX_ACC
            t_elapsed = 0.0
            while t_elapsed < t_acc - 1e-9:
                dt_eff = min(dt, t_acc - t_elapsed)
                tmp_t += dt_eff
                vL += accL * dt_eff
                vR += accR * dt_eff
                tmp_pos = self.update_pos(tmp_pos, vL, vR, dt_eff)
                pos_est.append(tmp_pos)
                motion_vals.append([accL, accR, vL, vR, tmp_t])
                t_elapsed += dt_eff

            # --- Phase 2: Constant velocity (if any)
            if t_max > 0:
                accL = accR = 0.0
                t_elapsed = 0.0
                while t_elapsed < t_max - 1e-9:
                    dt_eff = min(dt, t_max - t_elapsed)
                    tmp_t += dt_eff
                    tmp_pos = self.update_pos(tmp_pos, vL, vR, dt_eff)
                    pos_est.append(tmp_pos)
                    motion_vals.append([accL, accR, vL, vR, tmp_t])
                    t_elapsed += dt_eff

            # --- Phase 3: Deceleration
            accL = accR = -self.MAX_ACC
            t_elapsed = 0.0
            while t_elapsed < t_acc - 1e-9:
                dt_eff = min(dt, t_acc - t_elapsed)
                tmp_t += dt_eff
                vL += accL * dt_eff
                vR += accR * dt_eff
                tmp_pos = self.update_pos(tmp_pos, vL, vR, dt_eff)
                pos_est.append(tmp_pos)
                motion_vals.append([accL, accR, vL, vR, tmp_t])
                t_elapsed += dt_eff
            print(f"target_pos : {target_pos}")
            print(f"pos_est : {pos_est[-1]}")
            return motion_vals, pos_est,t_total
        return [-1]

    def calc_turn(self, initial_pos, target_pos, curr_t, dt=0.05):
        d_ang = self.wrap_to_pi(target_pos[2] - initial_pos[2])
        if abs(d_ang) < 1e-12:
            return np.zeros((0,5)), np.array([initial_pos]), curr_t

        # --- LSPB parameters
        t_total, t_acc, t_flat, peak_omega = self.LSPB_calc(
            self.MAX_ROT_VEL, self.MAX_ROT_ACC, d_ang)
        sign = 1.0 if d_ang >= 0 else -1.0
        A = sign * self.MAX_ROT_ACC
        W = peak_omega

        def omega_of(t):
            if t < t_acc:
                return A * t
            elif t < t_acc + t_flat:
                return W
            else:
                return W - A * (t - (t_acc + t_flat))

        def theta_of(t):
            if t < t_acc:
                return 0.5 * A * t**2
            elif t < t_acc + t_flat:
                return 0.5 * A * t_acc**2 + W * (t - t_acc)
            else:
                td = t - (t_acc + t_flat)
                return (0.5 * A * t_acc**2 +
                        W * t_flat + W * td - 0.5 * A * td**2)

        def ang_acc_of(t):
            if t < t_acc:
                return A
            elif t < t_acc + t_flat:
                return 0.0
            else:
                return -A

        pos = [*initial_pos]
        t = 0.0
        t_global = curr_t
        motion_vals, pos_est = [], [pos]

        while t < t_total - 1e-12:
            next_t = min(t + dt, t_total)
            theta_now = theta_of(t)
            theta_next = theta_of(next_t)
            dtheta = theta_next - theta_now
            omega_mid = (omega_of(t) + omega_of(next_t)) * 0.5
            ang_acc_mid = ang_acc_of((t + next_t) * 0.5)

            vR = omega_mid * self.WIDTH/2
            vL = -vR
            accR = ang_acc_mid * self.WIDTH/2
            accL = -accR

            t_global += (next_t - t)
            pos = self.update_pos(pos, vL, vR, next_t - t)
            motion_vals.append([accL, accR, vL, vR, t_global])
            pos_est.append(pos)
            t = next_t

        # ensure final heading matches exactly
        pos_est[-1][2] = initial_pos[2] + d_ang
        return np.array(motion_vals), np.array(pos_est), t_global

    def check_collisions(self): 
        # takes boundary point list 
        raise NotImplementedError

    def import_commands(self,commands):
        self.command_list = np.array(commands)

    def timestep(self,global_time, dt=0.05):
        # updates position based on accelerations from each wheel
        # puts command into history after enacting it
        # command format : [accL,accR,vL,vR,tmp_t]
        # Assumes linear motion or rotating in place for this robot
        accL,accR,vL,vR,tmp_t = self.command_list[0]
        self.R_wheel_acc = accR
        self.L_wheel_acc = accL
        # self.R_wheel_vel += accR * dt #more error on turns from this; not sure why
        # self.L_wheel_vel += accL * dt
        self.R_wheel_vel = vR
        self.L_wheel_vel = vL

        delta_acc = self.R_wheel_acc-self.L_wheel_acc
        
        self.ang_acc = np.arctan2(delta_acc/2,self.WIDTH/2)
        self.ang_vel += self.ang_acc * dt
        
        new_pos = self.update_pos((self.y,self.x,self.ang),self.L_wheel_vel,self.R_wheel_vel)
        self.y = new_pos[0]
        self.x = new_pos[1]
        self.ang = new_pos[2]
        self.command_history.append((copy.copy(self.command_list[0])))
        self.command_list = np.delete(self.command_list,0, axis=0)
        if len(self.command_list)==0: self.R_wheel_vel=0;self.L_wheel_vel=0
        return len(self.command_list)==0

class trailer_bot:
    def __init__(self,pos:tuple[float,float],angle:float,scale:int):
        # units in meters and seconds
        self.WIDTH = 2.0
        self.LENGTH = 4.5
        self.AXLEOFFSET = 3.4
        self.HITCHOFFSET = 5.0
        self.scale = scale
        self.test_seed = 10

        #vectors from wheels
        self.MAX_ACC = 1 # meters per second^2 
        self.MAX_VEL = 5 # meters per second
        self.MAX_ROT_ACC = np.arctan2(self.MAX_ACC,self.WIDTH/2)
        self.MAX_ROT_VEL = np.arctan2(self.MAX_VEL,self.WIDTH/2)
        
        self.y = pos[0]
        self.x = pos[1]
        self.ang = angle
        
        self.R_wheel_vel = 0 #linear not rotationally
        self.L_wheel_vel = 0 #linear not rotationally
        self.R_wheel_acc = 0 #linear not rotationally
        self.L_wheel_acc = 0 #linear not rotationally
        self.ang_vel = 0
        self.ang_acc = 0
        
        #based on center point location between back wheels
        self.boundry_offsets = [(self.AXLEOFFSET,self.WIDTH/2),(self.AXLEOFFSET,-self.WIDTH/2),(-(self.LENGTH-self.AXLEOFFSET),self.WIDTH/2),(-(self.LENGTH-self.AXLEOFFSET),-self.WIDTH/2)] #FR,FL,RR,RL
        self.wheel_points = [(0,self.WIDTH/2),(0,-self.WIDTH/2)] #R,L
        
        print(self.boundry_offsets)
        
        self.target_pos = []
        self.command_list = [] 
        self.command_history = []

    def calc_boundry(self,pos,buffer:float = 0.0,scale = 1): 
        # at a set position return the 4 corners for where the robot is at that timestep
        y,x,ang = pos
        y = y*scale
        x = x*scale
        # print(y,x,ang)
        x_off,y_off = np.array(self.boundry_offsets[0]) * (1+buffer) * scale
        FR = (x+(x_off*cos(ang))-(y_off*sin(ang)), y+(x_off*sin(ang))+(y_off*cos(ang)))
        x_off,y_off = np.array(self.boundry_offsets[1]) * (1+buffer) * scale
        FL = (x+(x_off*cos(ang))-(y_off*sin(ang)), y+(x_off*sin(ang))+(y_off*cos(ang)))
        x_off,y_off = np.array(self.boundry_offsets[2]) * (1+buffer) * scale
        RR = (x+(x_off*cos(ang))-(y_off*sin(ang)), y+(x_off*sin(ang))+(y_off*cos(ang)))
        x_off,y_off = np.array(self.boundry_offsets[3]) * (1+buffer) * scale
        RL = (x+(x_off*cos(ang))-(y_off*sin(ang)), y+(x_off*sin(ang))+(y_off*cos(ang)))
        # print([FR,RR,RL,FL])
        return [FR,RR,RL,FL]

    def calc_wheel_pts(self,pos,buffer:float = 0.0,scale = 1): 
        # at a set position return the 4 corners for where the robot is at that timestep
        y,x,ang = pos
        y = y*scale
        x = x*scale
        # print(y,x,ang)
        x_off,y_off = np.array(self.wheel_points[0]) * (1+buffer) * scale
        R = (x+(x_off*cos(ang))-(y_off*sin(ang)), y+(x_off*sin(ang))+(y_off*cos(ang)))
        x_off,y_off = np.array(self.wheel_points[1]) * (1+buffer) * scale
        L = (x+(x_off*cos(ang))-(y_off*sin(ang)), y+(x_off*sin(ang))+(y_off*cos(ang)))
        return [R,L]

    def get_current_pos(self):
        return (self.y,self.x,self.ang)

    def wrap_to_pi(self,angle):
        """Wrap angle to [-pi, pi)."""
        return (angle + pi) % (2.0 * pi) - pi
    def update_pos(self, hitch_pos, hitch_vel, dt=0.05):
        """
        Update the trailer pose using a simplified trailer model.
        The trailer follows a hitch point at a fixed offset distance.
        The trailer orientation is determined by the hitch-trailer geometry.
        
        Args:
            hitch_pos: iterable or array-like (y, x, theta) -- pose of the hitch point on the towing vehicle
            hitch_vel: linear velocity magnitude of the hitch point (m/s)
            dt: timestep (s)
        
        Returns:
            np.ndarray([y_new, x_new, theta_new])
        """
        hitch_y, hitch_x, hitch_theta = float(hitch_pos[0]), float(hitch_pos[1]), float(hitch_pos[2])
        v_hitch = float(hitch_vel)
        
        # Current trailer state
        trailer_y, trailer_x, trailer_theta = self.y, self.x, self.ang
        L = float(self.HITCHOFFSET)
        
        # Hitch velocity components in world frame
        vx = v_hitch * np.cos(hitch_theta)
        vy = v_hitch * np.sin(hitch_theta)
        
        # Update hitch position based on velocity
        hitch_x_new = hitch_x + vx * dt
        hitch_y_new = hitch_y + vy * dt
        
        # Vector from current trailer center to new hitch position
        dx = hitch_x_new - trailer_x
        dy = hitch_y_new - trailer_y
        current_dist = np.sqrt(dx**2 + dy**2)
        
        # Calculate new trailer angle: trailer must point toward the hitch
        theta_new = np.arctan2(dy, dx)
        
        # Update trailer position to maintain hitch offset at distance L
        # Place trailer center at distance L behind the hitch, along the line connecting them
        if current_dist > 1e-6:
            # Unit vector from trailer to hitch
            ux = dx / current_dist
            uy = dy / current_dist
            
            # New trailer position: L distance from hitch in opposite direction
            x_new = hitch_x_new - L * ux
            y_new = hitch_y_new - L * uy
        else:
            # Fallback if distances are too small
            x_new = hitch_x_new - L * np.cos(theta_new)
            y_new = hitch_y_new - L * np.sin(theta_new)
        
        # Normalize angle
        theta_new = self.wrap_to_pi(theta_new)
        
        # Update internal state
        self.x = x_new
        self.y = y_new
        self.ang = theta_new
        
        return np.array([y_new, x_new, theta_new])


class AckermannPathfinder:
    def __init__(self, car_bot, obstacle_grid):
        """
        Hybrid A* pathfinder for Ackermann steering vehicles.
        
        Args:
            car_bot: Your car_bot instance
            obstacle_grid: 2D numpy array where 0=obstacle, 1=free
        """
        self.car = car_bot
        self.grid = obstacle_grid
        self.grid_h, self.grid_w = obstacle_grid.shape
        
        # Discretization parameters
        self.angle_bins = 16  # Discretize heading into 16 directions
        self.angle_res = 2 * np.pi / self.angle_bins
        
        # Motion primitives (steering angles to try)
        self.steering_angles = np.array([
            -np.pi/4,  # hard left
            -np.pi/6,  # medium left
            -np.pi/12, # soft left
            0,         # straight
            np.pi/12,  # soft right
            np.pi/6,   # medium right
            np.pi/4    # hard right
        ])
        
        # Step size for motion primitives (in meters)
        self.step_size = 1.0
        self.dt = 0.1  # timestep for simulation
        
    def plan_path(self, start_pos, goal_pos, max_iterations=5000):
        """
        Find path from start to goal using Hybrid A*.
        
        Args:
            start_pos: (y, x, angle) in world coordinates
            goal_pos: (y, x, angle) in world coordinates
            
        Returns:
            list of (y, x, angle) waypoints, or None if no path found
        """
        start_node = self._create_node(start_pos)
        goal_node = self._create_node(goal_pos)
        
        # Priority queue: (f_score, counter, node)
        open_set = []
        counter = 0
        heapq.heappush(open_set, (self._heuristic(start_node, goal_node), counter, start_node))
        counter += 1
        
        # Visited states (discretized)
        visited = set()
        
        # Cost so far
        g_score = {self._discretize_state(start_node): 0}
        
        # Parent tracking for path reconstruction
        came_from = {}
        
        iterations = 0
        
        while open_set and iterations < max_iterations:
            iterations += 1
            _, _, current = heapq.heappop(open_set)
            
            current_key = self._discretize_state(current)
            
            # Check if reached goal
            if self._is_goal(current, goal_node):
                return self._reconstruct_path(came_from, current)
            
            if current_key in visited:
                continue
                
            visited.add(current_key)
            
            # Try all motion primitives
            for steering_angle in self.steering_angles:
                # Simulate motion
                new_node = self._simulate_motion(current, steering_angle)
                
                if new_node is None:
                    continue  # Collision or out of bounds
                
                new_key = self._discretize_state(new_node)
                
                if new_key in visited:
                    continue
                
                # Calculate cost
                tentative_g = g_score[current_key] + self._edge_cost(current, new_node, steering_angle)
                
                if new_key not in g_score or tentative_g < g_score[new_key]:
                    g_score[new_key] = tentative_g
                    f_score = tentative_g + self._heuristic(new_node, goal_node)
                    heapq.heappush(open_set, (f_score, counter, new_node))
                    counter += 1
                    came_from[new_key] = (current, steering_angle)
        
        return None  # No path found
    
    def _create_node(self, pos):
        """Create a node dictionary from position."""
        return {
            'y': pos[0],
            'x': pos[1],
            'angle': self.car.wrap_to_pi(pos[2])
        }
    
    def _discretize_state(self, node):
        """Discretize continuous state for duplicate detection."""
        y_bin = int(node['y'] * self.car.scale)
        x_bin = int(node['x'] * self.car.scale)
        angle_bin = int((node['angle'] + np.pi) / self.angle_res)
        return (y_bin, x_bin, angle_bin)
    
    def _simulate_motion(self, node, steering_angle):
        """Simulate motion primitive and check for collisions."""
        y, x, angle = node['y'], node['x'], node['angle']
        
        # Simulate forward motion with given steering angle
        num_steps = int(self.step_size / (self.car.MAX_VEL * self.dt))
        drive_vel = self.car.MAX_VEL * 0.5  # Use moderate speed
        
        trajectory = []
        current_pos = np.array([y, x, angle])
        
        for _ in range(num_steps):
            current_pos = self.car.update_pos(current_pos, drive_vel, steering_angle, self.dt)
            trajectory.append(current_pos.copy())
            
            # Check collision at each step
            if not self._is_collision_free(current_pos):
                return None
        
        final_pos = trajectory[-1]
        return {
            'y': final_pos[0],
            'x': final_pos[1],
            'angle': self.car.wrap_to_pi(final_pos[2]),
            'trajectory': trajectory
        }
    
    def _is_collision_free(self, pos):
        """Check if robot at given pose collides with obstacles."""
        corners = self.car.calc_boundry(pos, buffer=0.0, scale=self.car.scale)
        
        # Convert corners to path
        path = Path(corners)
        
        # Get bounding box
        corners_array = np.array(corners)
        min_x = max(0, int(np.floor(corners_array[:, 0].min())))
        max_x = min(self.grid_w, int(np.ceil(corners_array[:, 0].max())))
        min_y = max(0, int(np.floor(corners_array[:, 1].min())))
        max_y = min(self.grid_h, int(np.ceil(corners_array[:, 1].max())))
        
        # Check if out of bounds
        if min_x < 0 or max_x >= self.grid_w or min_y < 0 or max_y >= self.grid_h:
            return False
        
        # Check grid cells in bounding box
        for y in range(min_y, max_y):
            for x in range(min_x, max_x):
                if self.grid[y, x] == 0:  # Obstacle
                    # Check if cell overlaps with robot
                    cell_center = np.array([[x + 0.5, y + 0.5]])
                    if path.contains_points(cell_center)[0]:
                        return False
        
        return True
    
    def _edge_cost(self, node1, node2, steering_angle):
        """Calculate cost of moving from node1 to node2."""
        # Euclidean distance
        dist = np.sqrt((node2['y'] - node1['y'])**2 + (node2['x'] - node1['x'])**2)
        
        # Penalize turning
        turn_penalty = abs(steering_angle) * 0.5
        
        # Penalize reversing direction changes
        angle_diff = abs(self.car.wrap_to_pi(node2['angle'] - node1['angle']))
        direction_penalty = angle_diff * 0.3
        
        return dist + turn_penalty + direction_penalty
    
    def _heuristic(self, node1, node2):
        """Heuristic for A* (Euclidean distance + heading difference)."""
        dist = np.sqrt((node2['y'] - node1['y'])**2 + (node2['x'] - node1['x'])**2)
        angle_diff = abs(self.car.wrap_to_pi(node2['angle'] - node1['angle']))
        return dist + angle_diff * 0.5
    
    def _is_goal(self, node, goal):
        """Check if node is close enough to goal."""
        dist = np.sqrt((node['y'] - goal['y'])**2 + (node['x'] - goal['x'])**2)
        angle_diff = abs(self.car.wrap_to_pi(node['angle'] - goal['angle']))
        return dist < 0.5 and angle_diff < np.pi / 8
    
    def _reconstruct_path(self, came_from, current):
        """Reconstruct path from goal to start."""
        path = [current]
        current_key = self._discretize_state(current)
        
        while current_key in came_from:
            current, steering = came_from[current_key]
            path.append(current)
            current_key = self._discretize_state(current)
        
        path.reverse()
        
        # Convert to simple (y, x, angle) tuples
        return [(node['y'], node['x'], node['angle']) for node in path]


# Usage example:
def car_pathfinding(car_bot,initial_pos,goal_pos,obstacle_grid):
    """Add pathfinding method to existing car_bot class."""
    
    
    
    def _path_to_commands(self, path):
        """Convert waypoint path to command_list."""
        self.command_list = []
        
        for i in range(len(path) - 1):
            y1, x1, ang1 = path[i]
            y2, x2, ang2 = path[i + 1]
            
            # Calculate required steering
            dy = y2 - y1
            dx = x2 - x1
            target_angle = np.arctan2(dy, dx)
            
            # Calculate steering angle (simplified)
            angle_diff = self.wrap_to_pi(target_angle - ang1)
            steering_angle = np.clip(angle_diff, -np.pi/4, np.pi/4)
            
            # Set wheel velocities for Ackermann steering
            drive_vel = self.MAX_VEL * 0.5
            
            # Duration to reach next waypoint
            dist = np.sqrt(dy**2 + dx**2)
            duration = dist / drive_vel if drive_vel > 0 else 0.1
            
            # Simple command: [accL, accR, vL, vR, duration]
            # For Ackermann, both wheels move forward but at different rates
            self.command_list.append([0, 0, drive_vel, steering_angle, duration])
        
        self.command_list = np.array(self.command_list)
    
    # Attach methods to car_bot instance
    car_bot.find_path_to_goal = find_path_to_goal(car_bot, goal_pos,obstacle_grid)
    car_bot._path_to_commands = _path_to_commands()
    
    return car_bot