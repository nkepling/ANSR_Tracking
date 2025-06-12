from utils import *
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse,Circle,Polygon,Rectangle
import numpy as np
from dummy_pwm import Evader,forward, get_straight_away_trajectories, generate_goal_directed_trajectories,predict_evader_paths
import heapq # Used for the priority queue in A*
from rrt_grid import RRT
from scipy.interpolate import splprep, splev
import cv2
import math
from fov_solver import solve_uav_tracking_with_fov,get_half_planes_vectorized,grab_obstacles
import time


def map_coordinates(
    coords_source: np.ndarray,
    source_dims: tuple[int, int],
    target_dims: tuple[int, int],
    round_to_int: bool = False
) -> np.ndarray:

    if not isinstance(coords_source, np.ndarray) or coords_source.ndim != 2 or coords_source.shape[1] != 2:
        raise ValueError("coords_source must be a 2D NumPy array with shape (N, 2).")
    if not (isinstance(source_dims, tuple) and len(source_dims) == 2 and
            isinstance(target_dims, tuple) and len(target_dims) == 2):
        raise ValueError("source_dims and target_dims must be tuples of (width, height).")
    if not all(isinstance(d, int) and d > 0 for d in source_dims + target_dims):
        raise ValueError("Dimensions must be positive integers.")

    source_width, source_height = source_dims
    target_width, target_height = target_dims

    # Calculate scaling factors for x and y dimensions
    scale_x = target_width / source_width
    scale_y = target_height / source_height

    # Create a scaling array for broadcasting
    scaling_factors = np.array([scale_x, scale_y])

    # Perform the mapping using NumPy's efficient broadcasting
    coords_target = coords_source * scaling_factors

    if round_to_int:
        return np.round(coords_target).astype(int) # Use astype(int) if you want integer types
    else:
        return coords_target


def smooth_path_bspline(path, roads_map, smoothness=5.0):
    print("Smoothing path...")
    path = np.array(path)
    x, y = path[:, 0], path[:, 1]
    
    if len(x) <= 3:
        print("Path too short to smooth.")
        return path
    try:
        tck, u = splprep([x, y], s=smoothness, k=3)
    except Exception as e:
        print(f"Could not create spline: {e}")
        return path

    num_points = len(x) * 2
    u_new = np.linspace(u.min(), u.max(), num_points)
    x_new, y_new = splev(u_new, tck)
    
    smoothed_path = np.vstack((x_new, y_new)).T

    # --- VALIDATION STEP ---
    print("Validating smoothed path...")
    for point in smoothed_path:
        ix, iy = int(point[0]), int(point[1])
        
        # Check bounds
        if not (0 <= ix < roads_map.shape[1] and 0 <= iy < roads_map.shape[0]):
            print("Validation failed: Smoothed path went out of bounds.")
            return path,False # Return original path if validation fails
            
        # Check if on a road
        if roads_map[iy, ix] == 0:
            print("Validation failed: Smoothed path hit an obstacle.")
            return path,False # Return original path if validation fails
            
    print("Smoothing successful and validated.")
    return smoothed_path,True


def get_ground_truth_evader_path():
    segmap_file = "city_1000_1000_seg_segids.npz"
    mission_description_file = "description.json"
    obstacle_map_file = "city_1000_1000.npz"
    depth = 10
    roads, resolution = load_roads(segmap_file, visualize=False)
    roads = roads.T
    roads = np.rot90(roads)
    obstacle_map, resolution, (origin_x, origin_y) = load_obstacle_map(obstacle_map_file, depth=depth)
    obstacle_map = obstacle_map.T
    obstacle_map = np.rot90(obstacle_map)
    start_pos = np.array([360, 470])
    end_pos = np.array([470, 335])

    print("Finding path...")
    map_width = obstacle_map.shape[0]
    map_height = obstacle_map.shape[1]
    rand_area = [0, map_width, 0, map_height]

    validated = False
    path = None
    
    while not validated or (path is None):
        # # --- Instantiate and run RRT ---
        rrt = RRT(
            start=start_pos,
            goal=end_pos,
            roads_map=roads, # Pass the binary road map
            rand_area=rand_area,
            expand_dis=15.0, # Larger step size can speed up search on large maps
            path_resolution=1.0,
            max_iter=10000 # More iterations may be needed for complex maps
        )
        path = rrt.planning(animation=False)
        print(path)
        print("Path found!")

        smoothed_path,validated = smooth_path_bspline(path,roads,smoothness=15)

    # --- Visualization ---
    fig, ax = plt.subplots(figsize=(10, 10))
    # Use the roads map for visualization, as it's cleaner
    # ax.imshow(obstacle_map, cmap='gray', origin='lower')
    ax.imshow(obstacle_map,cmap="binary",alpha=1.0)
    ax.imshow(roads,"grey", alpha=0.4)
    ax.scatter(start_pos[0], start_pos[1], c='lime', s=100, label='Start', zorder=5, marker='o')
    ax.scatter(end_pos[0], end_pos[1], c='red', s=100, label='End', zorder=5, marker='X')

    if path is not None:
        plt.plot([x for (x, y) in smoothed_path], [y for (x, y) in smoothed_path], '-r', linewidth=2, label="Final Path")
    else:
        print("No path could be found between start and end.")

    ax.legend()
    plt.show()

    return smoothed_path



def main():
    pass
    
# (All your functions and classes like Evader, solve_uav_tracking_with_fov, etc., remain above this block)

if __name__ == "__main__":
       # --- 1. SETUP ENVIRONMENT AND LOAD DATA ---
    segmap_file = "city_1000_1000_seg_segids.npz"
    obstacle_map_file = "city_1000_1000.npz"

    # ground_truth_file_name = "2025-05-14-sprint1-scenarios-v1/scenario-028-000/trajectories/target-pzFVxwf.csv" # good check for non-convex polygon
    # ground_truth_file_name = "2025-05-14-sprint1-scenarios-v1/scenario-021-000/trajectories/target-giQdQiL.csv" # this one is hard
    # ground_truth_file_name = "2025-05-14-sprint1-scenarios-v1/scenario-012-000/trajectories/target-pnsVQaV.csv"
    ground_truth_file_name = "2025-05-14-sprint1-scenarios-v1/scenario-001-000/trajectories/target-THrHZpn.csv"

    # Load map and road data
    roads, resolution = load_roads(segmap_file, visualize=False)
    roads = np.rot90(roads.T)

    obstacle_map, resolution, (origin_y,origin_x) = load_obstacle_map(obstacle_map_file, depth=10)
    obstacle_map = np.rot90(obstacle_map.T)

    # load obstacle skelaton 

    # down sample dim  
    dim = (1000, 1000)
    skeletonized_roads = skeletonize_roads(roads)
    low_res_skeleton_map, skeleton_centroids = discretize_obstacle_map(skeletonized_roads, dim, obs_thresh=0.01)
    graph_from_skeleton_lil = create_grid_csgraph(low_res_skeleton_map)
    graph_from_skeleton_csr = graph_from_skeleton_lil.tocsr()


    

    
    evader_gt_path = get_evader_path_from_file(ground_truth_file_name,obstacle_map,roads,resolution,(origin_y,origin_x))
    

    # --- 2. SETUP MPC SIMULATION ---
    DT = 0.1
    MAX_SIM_TIME = 60.0
    N_TRAJ = 20
    VIEW_RANGE = 150.0 # The 20x20 meter view
    avoidance_region = 30
    uav_max_velocity = 28.0
    evader_velocity = 20.0
    PREDICTION_SIMILARITY_THRESH = 0.0

    MAX_PREDICTION_DEPTH = 15

    # Calculate the initial direction of the evader's path
    evader_start_pos = evader_gt_path[0]
    evader_next_pos = evader_gt_path[1]
    initial_angle = math.atan2(evader_next_pos[1] - evader_start_pos[1], 
                              evader_next_pos[0] - evader_start_pos[0])

    # Initialize the UAV at the evader's start, but pointing in the correct direction
    uav_state = np.array([evader_start_pos[0], evader_start_pos[1], initial_angle])

    # Initialize the evader as before
    evader = Evader(
        x=evader_start_pos[0], y=evader_start_pos[1],
        theta=initial_angle, # Use the same initial angle
        v=evader_velocity, path=evader_gt_path, lookahead_distance=5.0
    )

    
    # MPC Parameters
    

    uav_max_angular_velocity = np.deg2rad(120) # whas 120
    fov_ellipse_params = {'a': 10.0, 'b': 10.0}
    standoff_distance = 5.0
    obstacles = [] # Ignoring MPC obstacles for now
    obstacle_weight = 1.0
    fov_weight = 10.0
    tracking_weight = 500
    solver_options = {
        "print_level": 0, "acceptable_tol": 1e-3, "acceptable_iter": 5, "mu_strategy": "adaptive","max_cpu_time": 1.0
    }

    # Data Logging & Warm Start variables
    uav_history = [uav_state.copy()]
    evader_history = [evader.pos.copy()]
    prev_state_sol, prev_control_sol = None, None


    solve_times = []
    construction_times = []
    solver_failure = []

    print(f"\nStarting simulation with sliced map visualization...")

    # --- Set up the plot window ---
    fig, ax = plt.subplots(figsize=(10, 10))
    a, b = fov_ellipse_params['a'], fov_ellipse_params['b']
    

    # Get map dimensions for clipping
    map_height, map_width = obstacle_map.shape

    # --- Main Simulation & Live Visualization Loop ---
    for ind,t in enumerate(np.arange(0, MAX_SIM_TIME, DT)):
   
        state_guess, control_guess = None, None
        if prev_state_sol is not None:
            state_guess = np.roll(prev_state_sol, -1, axis=1)
            state_guess[:, -1] = state_guess[:, -2]
        if prev_control_sol is not None:
            control_guess = np.roll(prev_control_sol, -1, axis=1)
            control_guess[:, -1] = control_guess[:, -2]
        
        
        evader_prediction = evader.get_predicted_trajectory(N_TRAJ, DT)

        num_alternatives = 4
        noise_scale = 1.0  # Adjust this for more/less deviation

        alternative_evader_trajectories = []
        for i in range(num_alternatives):
            # Add Gaussian noise to each (x, y) position in the trajectory
            noisy_traj = evader_prediction[:, 1:] + np.random.normal(
                loc=0.0, scale=noise_scale, size=evader_prediction[:, 1:].shape
            )


            alternative_evader_trajectories.append(noisy_traj)

        evader_probabilities = np.ones(len(alternative_evader_trajectories))
                ####Start Solver ######
        start_time = time.perf_counter()


        # assuming ground truth evader state estimation at time t
        theta_radians = evader.theta 
        evader_direction_vector = np.array([np.cos(theta_radians), np.sin(theta_radians)])
            
        polygonal_obstacles = grab_obstacles(uav_state,obstacle_map,avoidance_region,get_ellipse=False)

        optimal_controls, planned_state,construction_time,success = solve_uav_tracking_with_fov(
            uav_state, tracking_weight,alternative_evader_trajectories, evader_probabilities,uav_max_velocity, uav_max_angular_velocity,
           polygonal_obstacles, fov_ellipse_params, fov_weight, standoff_distance,
            solver_options, state_guess, control_guess, N_TRAJ, DT,  saftey_radius=4, slack_weight=1e6 
        )
        end_time = time.perf_counter()
        solve_times.append(end_time - start_time)
        construction_times.append(construction_time)
        solver_failure.append(success)

        print("solver time", end_time-start_time)

        ###############End Solver ############

        prev_state_sol, prev_control_sol = planned_state, optimal_controls
        v, omega = optimal_controls[:, 0]
        uav_state[0] += v * np.cos(uav_state[2]) * DT
        uav_state[1] += v * np.sin(uav_state[2]) * DT
        uav_state[2] += omega * DT
        evader.update(DT)
        
        uav_history.append(uav_state.copy())
        evader_history.append(evader.pos.copy())


        
        # --- 2. Visualization in Sliced View ---
        # ax.clear()
        
        # uav_current_pos = uav_state[:2]
        # uav_current_theta = uav_state[2]
        # half_view = VIEW_RANGE / 2.0

        # win_x_min = uav_current_pos[0] - half_view
        # win_y_min = uav_current_pos[1] - half_view
        # slice_x_start = np.clip(int(win_x_min), 0, map_width)
        # slice_x_end = np.clip(int(win_x_max := win_x_min + VIEW_RANGE), 0, map_width)
        # slice_y_start = np.clip(int(win_y_min), 0, map_height)
        # slice_y_end = np.clip(int(win_y_max := win_y_min + VIEW_RANGE), 0, map_height)


        # map_slice = obstacle_map[slice_y_start:slice_y_end, slice_x_start:slice_x_end]

        # uav_path_arr = np.array(uav_history)
        # evader_path_arr = np.array(evader_history)
        
        # local_uav_path = uav_path_arr[:, :2] - [slice_x_start, slice_y_start]
        # local_evader_path = evader_path_arr - [slice_x_start, slice_y_start]
        # local_uav_pos = uav_current_pos - [slice_x_start, slice_y_start]
        # local_evader_pos = evader.pos - [slice_x_start, slice_y_start]

        # ax.imshow(map_slice, cmap='binary', origin='lower', alpha=1.0,
        #         extent=[0, slice_x_end - slice_x_start, 0, slice_y_end - slice_y_start])
        
        # road_slice = roads[slice_y_start:slice_y_end, slice_x_start:slice_x_end]
        # ax.imshow(road_slice, cmap='bone', origin='lower', alpha=0.4,
        #         extent=[0, slice_x_end - slice_x_start, 0, slice_y_end - slice_y_start])


        # local_planned_state = planned_state.copy()
        # local_planned_state[0, :] -= slice_x_start
        # local_planned_state[1, :] -= slice_y_start
        # ax.plot(local_planned_state[0, 1:], local_planned_state[1, 1:], 'g-+', alpha=0.6, label='UAV Plan')

        # local_evader_traj = evader_prediction.copy()
        # local_evader_traj[0,:] -= slice_x_start
        # local_evader_traj[1,:] -= slice_y_start
        # ax.plot(local_evader_traj[0, :], local_evader_traj[1, :], 'r-+', alpha=0.6, label='Evader Traj')

        # for alt_traj in alternative_evader_trajectories:
        #     local_alt_traj = alt_traj.copy()
        #     local_alt_traj[0, :] -= slice_x_start
        #     local_alt_traj[1, :] -= slice_y_start
        #     ax.plot(local_alt_traj[0, :], local_alt_traj[1, :], '--', alpha=0.4, label='Alt Evader Traj')

        # ax.plot(local_evader_pos[0], local_evader_pos[1], 'o', color='red', markersize=8, label="Evader")
        # ax.plot(local_uav_pos[0], local_uav_pos[1], 'o', color='blue', markersize=8, label="Pursuer")


        # # for pred_traj in predicted_trajectories:
        # #     pred = np.array(pred_traj).T
            
        # #     pred[0,:] -= slice_x_start
        # #     pred[1, :] -= slice_y_start
        # #     ax.plot(pred[0, 1:], pred[1, 1:], '--k', alpha=1.0, label='predicted_traj')

        # # Plot FOV ellipse in local coordinates
        # heading_vec = np.array([np.cos(uav_current_theta), np.sin(uav_current_theta)])
        # # The ellipse center must also be transformed to the local frame
        # ellipse_center_world = uav_current_pos + a * heading_vec
        # local_ellipse_center = ellipse_center_world - [slice_x_start, slice_y_start]
        
        # fov_ellipse = Ellipse(
        #     xy=local_ellipse_center, width=2 * a, height=2 * b, angle=np.rad2deg(uav_current_theta),
        #     edgecolor='cyan', facecolor='cyan', alpha=0.25,label="FOV"
        # )
        # ax.add_patch(fov_ellipse)

      
        # if polygonal_obstacles:
        #     for poly in polygonal_obstacles:
        #         local_polygon = poly - [slice_x_start, slice_y_start]
                
        #         ax.add_patch(Polygon(local_polygon, facecolor='orange', alpha=0.5, edgecolor='red',label="Obstacle"))


        # half_avoid = avoidance_region // 2
        

        # avoid_x_start_global = max(int(uav_current_pos[0]) - half_avoid, 0)
        # avoid_y_start_global = max(int(uav_current_pos[1]) - half_avoid, 0)
        # avoid_width_global = (min(int(uav_current_pos[0]) + half_avoid, map_width)) - avoid_x_start_global
        # avoid_height_global = (min(int(uav_current_pos[1]) + half_avoid, map_height)) - avoid_y_start_global

        # # 2. Transform the box's bottom-left corner to local view coordinates
        # #    The width and height remain the same.
        # avoid_box_local_xy = (
        #     avoid_x_start_global - slice_x_start,
        #     avoid_y_start_global - slice_y_start
        # )

        # # 3. Create the Rectangle patch
        # avoid_box_patch = Rectangle(
        #     avoid_box_local_xy,
        #     avoid_width_global,
        #     avoid_height_global,
        #     edgecolor='yellow',
        #     facecolor='none',
        #     linestyle='--',
        #     linewidth=2,
        #     label='Avoidance Region' # This label will appear in the legend
        # )
        
        # # 4. Add the patch to the plot
        # ax.add_patch(avoid_box_patch)


        
        # # Formatting
        # ax.set_title(f"Time: {t:.1f}s | UAV Velo: {v:.1f}m/s | Solve Time {solve_times[ind]*1000:.3f}ms")

        # ax.set_xlim(0, slice_x_end - slice_x_start)
        # ax.set_ylim(0, slice_y_end - slice_y_start)
        # # ax.axis('equal')
        # ax.legend(loc='upper right')
        # plt.pause(0.01)

        if evader.finished:
            print(f"Evader reached the end of the path at time {t:.1f}s.")
            break

    print("Simulation finished.")

    mean_solver_time = np.mean(solve_times)
    std_solver_time = np.std(solve_times)

    success_rate = sum(solver_failure)/len(solver_failure)
    
    
    

    print("\n--- Solver Performance ---")
    print(f"Solver success {success_rate} \n")
    print(f"Average solve time: {np.mean(solve_times):.4f} seconds")
    print(f"Median solve time:  {np.median(solve_times):.4f} seconds")
    print(f"Max solve time:     {np.max(solve_times):.4f} seconds")
    print(f"Min solve time:     {np.min(solve_times):.4f} seconds")
    print(f"Standard deviation: {np.std(solve_times):.4f} seconds")

      
    print(f"Average construction time: {np.mean(construction_times):.4f} seconds")
    print(f"Median construction time:  {np.median(construction_times):.4f} seconds")
    print(f"Max construction time:     {np.max(construction_times):.4f} seconds")
    print(f"Min construction time:     {np.min(construction_times):.4f} seconds")
    print(f"Standard deviation: {np.std(construction_times):.4f} seconds")
# plt.show()
