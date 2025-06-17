from utils import *
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse,Circle,Polygon,Rectangle
import numpy as np
from dummy_pwm import Evader,forward, get_straight_away_trajectories, generate_goal_directed_trajectories,predict_evader_paths,filter_paths_by_direction
import heapq # Used for the priority queue in A*
from rrt_grid import RRT
from ompl_planner import OMPLGridPlanner
from scipy.interpolate import splprep, splev, CubicSpline
import cv2
import math
from fov_solver import solve_uav_tracking_with_fov,get_half_planes_vectorized,grab_obstacles
import time


def filter_paths_by_direction(paths, evader):
    """
    Filters a list of paths to keep only those that are generally in the
    same direction as the evader's current velocity vector.

    Args:
        paths (list): A list of paths, where each path is a numpy array of
                      shape (2, T) representing (x, y) coordinates.
        evader (Evader): The evader object containing its current state (pos, theta).

    Returns:
        list: A new list containing only the forward-facing paths.
    """
    if not paths:
        return []

    # Get the evader's current direction vector from its heading (theta)
    evader_heading_vec = np.array([np.cos(evader.theta), np.sin(evader.theta)])
    
    forward_paths = []
    for path in paths:
        # A path needs at least two points to have a direction
        if path.shape[1] < 2:
            continue

        # The path's overall direction is from its start to its end
        path_start_pos = path[:, 0]
        path_end_pos = path[:, -1]
        path_direction_vec = path_end_pos - path_start_pos

        # Normalize the path vector to avoid issues with very short paths
        path_norm = np.linalg.norm(path_direction_vec)
        if path_norm < 1e-6:
            continue # Skip zero-length paths
        
        path_direction_vec /= path_norm

        # Calculate the dot product
        dot_product = np.dot(evader_heading_vec, path_direction_vec)

        # Keep the path only if it's pointing generally forward (angle < 90 degrees)
        if dot_product >= 0:
            forward_paths.append(path)
            
    return forward_paths


def get_longterm_predictions(
    roads, G, kdtree, nodes_list, evader, obstacle_map, N, dt, k, search_radius=50.0
):
    """
    Generates k plausible long-term trajectories, prioritizing forward-facing
    paths but supplementing with other paths to ensure k predictions are returned
    if available.
    """
    # 1. Find the evader's current node and all paths in a local subgraph

    evader_pos = evader.get_noisy_position()
    center_node, dist = find_closest_node_kdtree(kdtree, nodes_list, evader_pos)
    center_coords = G.nodes[center_node]['pos']

    indices_in_radius = kdtree.query_ball_point(center_coords, r=search_radius)
    if not indices_in_radius:
        return []

    nodes_in_radius = [nodes_list[i] for i in indices_in_radius]
    subgraph = G.subgraph(nodes_in_radius)

    if center_node not in subgraph:
        return []

    all_paths_from_center = nx.single_source_shortest_path(subgraph, center_node)

    # --- NEW LOGIC: Prioritize forward paths, but keep backward paths as a fallback ---

    # 2. Separate paths into forward and backward groups
    evader_theta = evader.get_noisy_heading()
    evader_heading_vec = np.array([np.cos(evader_theta), np.sin(evader_theta)])
    start_node_pos = np.array(G.nodes[center_node]['pos'])
    
    forward_paths = {}
    backward_paths = {}
    for node, path in all_paths_from_center.items():
        if node == center_node:
            continue
        
        end_node_pos = G.nodes[node]['pos']
        path_direction_vec = np.array(end_node_pos) - start_node_pos
        
        if np.dot(evader_heading_vec, path_direction_vec) >= -0.8:
            forward_paths[node] = path
        else:
            backward_paths[node] = path

    # 3. Sort both groups by length (farthest first)
    sorted_forward = sorted(
        [(node, len(path) - 1) for node, path in forward_paths.items()],
        key=lambda item: item[1], reverse=True
    )
    sorted_backward = sorted(
        [(node, len(path) - 1) for node, path in backward_paths.items()],
        key=lambda item: item[1], reverse=True
    )
    
    # 4. Combine the lists to create a prioritized list of potential goals
    # Start with the best (forward, long) and append the next-best (backward, long)
    prioritized_goals = sorted_forward + sorted_backward
    
    # 5. Select the top k goals from the combined list
    # This guarantees we take as many as possible, up to k, in the right order.
    num_to_get = min(k, len(prioritized_goals))
    final_goal_nodes = [node for node, length in prioritized_goals[:num_to_get]]
    
    # Get the coordinates for the final list of goals
    goal_positions = [G.nodes[node]["pos"] for node in final_goal_nodes]
    
    # --- The rest of the function proceeds as before ---

    evader_v = evader.get_noisy_velocity()

    # 6. For each goal, generate a trajectory using RRT
    predictions = []
    for goal_node in final_goal_nodes:
        # a. Retrieve the path of node IDs from our original calculation
        node_path = all_paths_from_center[goal_node]
        
        # b. Convert the path of node IDs to a path of (x, y) coordinates
        #    This is now the path for the evader to follow
        coord_path = [G.nodes[n]['pos'] for n in node_path]

        # c. Smooth the coordinate path (this step remains the same)
        #    Note: The path must start near the evader for smoothing to be effective.
        #    We can prepend the evader's actual position for a better result.
        path_for_smoothing = [evader.pos] + coord_path
        smoothed_path, validated = smooth_path_bspline(path_for_smoothing, roads, smoothness=15)
        
        if not validated:
            continue

        # d. Create a temporary evader to generate the final trajectory
        temp_evader = Evader(
            x=evader_pos[0], y=evader_pos[1],
            theta=evader_theta,
            v=evader_v, path=smoothed_path, lookahead_distance=5.0
        )
        prediction = temp_evader.get_predicted_trajectory(N, dt)
        predictions.append(prediction)

    # The fallback and padding logic remains the same
    if len(predictions) == 0:
        predictions = generate_fallback_prediction(evader, N, dt, k)
    elif len(predictions) < k:
        best_prediction = predictions[0]
        while len(predictions) < k:
            predictions.append(best_prediction)
    
    return predictions
    
    
def generate_fallback_prediction(evader, N, dt, k, v_noise_scale=1.0, theta_noise_scale=np.deg2rad(5)):
    """
    Generates a simple "continue straight" trajectory as a fallback.
    """
    evader_pos = evader.get_noisy_position()
    fallback_path = np.zeros((2, N + 1))
    fallback_path[:, 0] = evader_pos
    evader_theta = evader.get_noisy_heading()
    # noisy_v = evader.v + np.random.normal(loc=0.0, scale=v_noise_scale)
    # noisy_v = max(0, noisy_v)
    # Create the path by repeatedly taking a step in the current direction
    step_vec = evader.v * dt * np.array([np.cos(evader_theta), np.sin(evader_theta)])
    
    for i in range(N):
        fallback_path[:, i+1] = fallback_path[:, i] + step_vec
    
    evader_v = evader.get_noisy_velocity()
    # Create a temporary evader to follow this simple path
    temp_evader = Evader(
        x=evader_pos[0], y=evader_pos[1],
        theta=evader_theta, v=evader_v,
        path=fallback_path.T, # Evader class may expect a list of points
        lookahead_distance=5.0
    )
    # Use the evader object to get a properly formatted trajectory
    fallback_prediction = temp_evader.get_predicted_trajectory(N, dt)
    
    # Return a list with the fallback repeated k times to meet the requirement
    return [fallback_prediction] * k





def generate_waypoint_deviations(base_trajectory, num_alternatives, noise_scale, num_waypoints=4):
    """
    Generates smooth trajectory deviations by perturbing key waypoints and
    re-interpolating with a spline.

    Args:
        base_trajectory (np.array): The original predicted trajectory (2, T).
        num_alternatives (int): How many alternative trajectories to generate.
        noise_scale (float): The magnitude of deviation for the waypoints.
        num_waypoints (int): The number of control waypoints to define the path.

    Returns:
        list: A list of alternative trajectory arrays.
    """
    alternative_trajectories = []
    _, num_timesteps = base_trajectory.shape

    # Time axis for interpolation
    t_original = np.linspace(0, 1, num_timesteps)

    # 1. Select waypoint indices along the original trajectory
    waypoint_indices = np.linspace(0, num_timesteps - 1, num_waypoints, dtype=int)

    for _ in range(num_alternatives):
        # Get the original waypoint coordinates
        original_waypoints = base_trajectory[:, waypoint_indices]
        perturbed_waypoints = original_waypoints.copy()

        # 2. Add noise to the intermediate waypoints (don't move the start point)
        noise = np.random.normal(loc=0.0, scale=noise_scale, size=(2, num_waypoints - 1))
        perturbed_waypoints[:, 1:] += noise

        # 3. Create a new smooth trajectory using a cubic spline
        cs = CubicSpline(t_original[waypoint_indices], perturbed_waypoints, axis=1)
        new_trajectory = cs(t_original)
        
        alternative_trajectories.append(base_trajectory)
        
    return alternative_trajectories



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
    path = np.array(path)
    x, y = path[:, 0], path[:, 1]
    
    if len(x) <= 3:
        return path,False
    try:
        tck, u = splprep([x, y], s=smoothness, k=3)
    except Exception as e:
        return path,False

    num_points = len(x) * 2
    u_new = np.linspace(u.min(), u.max(), num_points)
    x_new, y_new = splev(u_new, tck)
    
    smoothed_path = np.vstack((x_new, y_new)).T

    # --- VALIDATION STEP ---
    for point in smoothed_path:
        ix, iy = int(point[0]), int(point[1])
        
        # Check bounds
        if not (0 <= ix < roads_map.shape[1] and 0 <= iy < roads_map.shape[0]):
            return path,False # Return original path if validation fails
            
        # Check if on a road
        if roads_map[iy, ix] == 0:
            return path,False # Return original path if validation fails
            
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

def analyze_simulation_results(uav_history, evader_history, obstacle_map, fov_params, dt):
    """
    Analyzes simulation history to check for collisions and time in FOV.

    Args:
        uav_history (list or np.array): A record of the UAV's state [x, y, theta] over time.
        evader_history (list or np.array): A record of the evader's position [x, y] over time.
        obstacle_map (np.array): The binary map where non-zero values are obstacles.
        fov_params (dict): Parameters of the FOV ellipse, e.g., {'a': 10.0, 'b': 10.0}.
        dt (float): The simulation time step.

    Returns:
        tuple: (collision_occurred, total_time_in_fov)
               - collision_occurred (bool): True if the UAV ever entered an obstacle.
               - total_time_in_fov (float): The total seconds the evader was in the FOV.
    """
    uav_path = np.array(uav_history)
    evader_path = np.array(evader_history)
    map_height, map_width = obstacle_map.shape

    # --- 1. Check for Collisions ---
    collision_occurred = False
    # Convert entire path to integer coordinates for map lookup
    uav_coords = uav_path[:, :2].astype(int)
    # Clamp coordinates to be within map boundaries
    uav_coords[:, 0] = np.clip(uav_coords[:, 0], 0, map_width - 1)
    uav_coords[:, 1] = np.clip(uav_coords[:, 1], 0, map_height - 1)
    # Check for collisions using numpy's vectorized indexing
    # Note: obstacle_map is indexed (row, col) which corresponds to (y, x)
    collision_points = obstacle_map[uav_coords[:, 1], uav_coords[:, 0]]
    if np.any(collision_points):
        collision_occurred = True

    # --- 2. Track Time in FOV ---
    steps_in_fov = 0
    a = fov_params['a']
    b = fov_params['b']

    for i in range(len(uav_path)):
        uav_pos = uav_path[i, :2]
        uav_theta = uav_path[i, 2]
        evader_pos = evader_path[i]

        # Transform evader's position to the UAV's local coordinate frame
        vec_world = evader_pos - uav_pos
        cos_th = np.cos(uav_theta)
        sin_th = np.sin(uav_theta)
        
        x_local = vec_world[0] * cos_th + vec_world[1] * sin_th
        y_local = -vec_world[0] * sin_th + vec_world[1] * cos_th

        # Check if the local point is inside the FOV ellipse
        # The ellipse is centered at (a, 0) in the UAV's frame
        if ((x_local - a) / a)**2 + (y_local / b)**2 <= 1:
            steps_in_fov += 1
            
    total_time_in_fov = steps_in_fov * dt

    total_sim_time = dt * len(uav_history)
    return collision_occurred, total_time_in_fov,total_sim_time

def main():
    pass
    
# (All your functions and classes like Evader, solve_uav_tracking_with_fov, etc., remain above this block)

if __name__ == "__main__":
    import os
       # --- 1. SETUP ENVIRONMENT AND LOAD DATA ---
    segmap_file = "city_1000_1000_seg_segids.npz"
    obstacle_map_file = "city_1000_1000.npz"

    def find_target_csv_files(root_dir="2025-05-14-sprint1-scenarios-v1"):
        """
        Recursively search for all CSV files starting with 'target-' in subfolders of root_dir.
        Returns a list of full file paths.
        """
        target_files = []
        for dirpath, dirnames, filenames in os.walk(root_dir):
            for filename in filenames:
                if filename.startswith("target-") and filename.endswith(".csv"):
                    target_files.append(os.path.join(dirpath, filename))
        return target_files
    
    roads, resolution = load_roads(segmap_file, visualize=False)


    roads = np.rot90(roads.T)

    obstacle_map, resolution, (origin_y,origin_x) = load_obstacle_map(obstacle_map_file, depth=10)
    obstacle_map = np.rot90(obstacle_map.T)

    G,kdtree,nodes_list = get_kdtree(roads)

    print(len(nodes_list))

    files = find_target_csv_files()

    print("Num files: ",len(files))

    performance_table = {"filename": [],
                        "mean_solve_time": [],
                        "median_solve_time": [],
                        "max_solve_time": [],
                        "min_solve_time": [],
                        "solve_success_rate": [],
                        "collisions": [],
                        "time_in_fov": [],
                        "prediction_times":[]}

    for ind,ground_truth_file_name in enumerate(files):
        # ground_truth_file_name = "2025-05-14-sprint1-scenarios-v1/scenario-028-000/trajectories/target-pzFVxwf.csv" # good check for non-convex polygon
        # ground_truth_file_name = "2025-05-14-sprint1-scenarios-v1/scenario-021-000/trajectories/target-giQdQiL.csv" # this one is hard
        # ground_truth_file_name = "2025-05-14-sprint1-scenarios-v1/scenario-012-000/trajectories/target-pnsVQaV.csv"
        ground_truth_file_name = "2025-05-14-sprint1-scenarios-v1/scenario-001-000/trajectories/target-THrHZpn.csv"
        # Load map and road data
    
       
        print(f"\nStarting simulation with sliced map visualization {ind}")


        evader_gt_path = get_evader_path_from_file(ground_truth_file_name,obstacle_map,roads,resolution,(origin_y,origin_x))
        
        # --- 2. SETUP MPC SIMULATION ---
        DT = 0.1
        MAX_SIM_TIME = 60.0
        N_TRAJ = 15
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

        if obstacle_map[int(uav_state[0]),int(uav_state[1])]:
            "spawned in obstacle skipping"
            continue


        # Initialize the evader as before
        evader = Evader(
            x=evader_start_pos[0], y=evader_start_pos[1],
            theta=initial_angle, # Use the same initial angle
            v=evader_velocity, path=evader_gt_path, lookahead_distance=15.0
        )

        
        # MPC Parameters
        

        uav_max_angular_velocity = np.deg2rad(80) # whas 120
        fov_ellipse_params = {'a': 10.0, 'b': 8.0}
        standoff_distance = 10.0
        obstacles = [] # Ignoring MPC obstacles for now
        obstacle_weight = 10.0
        fov_weight = 10.0
        tracking_weight = 1.0
        control_effort_weight = 0.1
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
        prediction_times = []

   

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

            num_alternatives = 10
            noise_scale = 2.0  # Adjust this for more/less deviation

            # alternative_evader_trajectories = generate_waypoint_deviations(evader_prediction,num_alternatives,noise_scale=noise_scale,num_waypoints=len(evader_prediction))



        
                    ####Start Solver ######
            start_time = time.perf_counter()

            start_prediction_time = time.perf_counter()
            alternative_evader_trajectories = get_longterm_predictions(roads,G,kdtree,nodes_list,evader,obstacle_map,N_TRAJ,DT,num_alternatives)
            evader_probabilities = np.ones(len(alternative_evader_trajectories))
            end_prediction_time = time.perf_counter()

            # # assuming ground truth evader state estimation at time t
            # theta_radians = evader.get_noisy_heading()
            # evdaer_direction_vector = np.array([np.cos(theta_radians), np.sin(theta_radians)])
                
            polygonal_obstacles = grab_obstacles(uav_state,obstacle_map,avoidance_region)

            optimal_controls, planned_state,construction_time,success = solve_uav_tracking_with_fov(
                uav_state, tracking_weight,alternative_evader_trajectories, evader_probabilities,uav_max_velocity, uav_max_angular_velocity,
            polygonal_obstacles, fov_ellipse_params, fov_weight, standoff_distance,
                solver_options, state_guess, control_guess, N_TRAJ, DT,  saftey_radius=2, slack_weight=1e10,control_effort_weight=control_effort_weight 
            )
            end_time = time.perf_counter()
            solve_times.append(end_time - start_time)
            construction_times.append(construction_time-start_prediction_time)
            prediction_times.append(end_prediction_time-start_prediction_time)
            solver_failure.append(success)

            # print("solver time", end_time-start_time)

            ###############End Solver ############

            prev_state_sol, prev_control_sol = planned_state, optimal_controls
            v, omega = optimal_controls[:, 0]
            uav_state[0] += v * np.cos(uav_state[2]) * DT
            uav_state[1] += v * np.sin(uav_state[2]) * DT
            uav_state[2] += omega * DT
            evader.update(DT)
            
            uav_history.append(uav_state.copy())
            evader_history.append(evader.pos.copy())


    
            ax.clear()
            
            uav_current_pos = uav_state[:2]
            uav_current_theta = uav_state[2]
            half_view = VIEW_RANGE / 2.0

            win_x_min = uav_current_pos[0] - half_view
            win_y_min = uav_current_pos[1] - half_view
            slice_x_start = np.clip(int(win_x_min), 0, map_width)
            slice_x_end = np.clip(int(win_x_max := win_x_min + VIEW_RANGE), 0, map_width)
            slice_y_start = np.clip(int(win_y_min), 0, map_height)
            slice_y_end = np.clip(int(win_y_max := win_y_min + VIEW_RANGE), 0, map_height)


            map_slice = obstacle_map[slice_y_start:slice_y_end, slice_x_start:slice_x_end]

            uav_path_arr = np.array(uav_history)
            evader_path_arr = np.array(evader_history)
            
            local_uav_path = uav_path_arr[:, :2] - [slice_x_start, slice_y_start]
            local_evader_path = evader_path_arr - [slice_x_start, slice_y_start]
            local_uav_pos = uav_current_pos - [slice_x_start, slice_y_start]
            local_evader_pos = evader.pos - [slice_x_start, slice_y_start]

            ax.imshow(map_slice, cmap='binary', origin='lower', alpha=1.0,
                    extent=[0, slice_x_end - slice_x_start, 0, slice_y_end - slice_y_start])
            
            road_slice = roads[slice_y_start:slice_y_end, slice_x_start:slice_x_end]
            ax.imshow(road_slice, cmap='bone', origin='lower', alpha=0.4,
                    extent=[0, slice_x_end - slice_x_start, 0, slice_y_end - slice_y_start])


            local_planned_state = planned_state.copy()
            local_planned_state[0, :] -= slice_x_start
            local_planned_state[1, :] -= slice_y_start
            ax.plot(local_planned_state[0, 1:], local_planned_state[1, 1:], 'g-+', alpha=0.9, label='UAV Plan')

            local_evader_traj = evader_prediction.copy()
            local_evader_traj[0,:] -= slice_x_start
            local_evader_traj[1,:] -= slice_y_start
            ax.plot(local_evader_traj[0, :], local_evader_traj[1, :], 'r-+', alpha=0.9, label='Evader Traj')

            for alt_traj in alternative_evader_trajectories:
                local_alt_traj = alt_traj.copy()
                local_alt_traj[0, :] -= slice_x_start
                local_alt_traj[1, :] -= slice_y_start
                ax.plot(local_alt_traj[0, :], local_alt_traj[1, :], 'k--', alpha=0.4, label='Alt Evader Traj')

            ax.plot(local_evader_pos[0], local_evader_pos[1], 'o', color='red', markersize=8, label="Evader")
            ax.plot(local_uav_pos[0], local_uav_pos[1], 'o', color='blue', markersize=8, label="Pursuer")


            # for pred_traj in predicted_trajectories:
            #     pred = np.array(pred_traj).T
                
            #     pred[0,:] -= slice_x_start
            #     pred[1, :] -= slice_y_start
            #     ax.plot(pred[0, 1:], pred[1, 1:], '--k', alpha=1.0, label='predicted_traj')

            # Plot FOV ellipse in local coordinates
            heading_vec = np.array([np.cos(uav_current_theta), np.sin(uav_current_theta)])
            # The ellipse center must also be transformed to the local frame
            ellipse_center_world = uav_current_pos + a * heading_vec
            local_ellipse_center = ellipse_center_world - [slice_x_start, slice_y_start]
            
            fov_ellipse = Ellipse(
                xy=local_ellipse_center, width=2 * a, height=2 * b, angle=np.rad2deg(uav_current_theta),
                edgecolor='cyan', facecolor='cyan', alpha=0.25,label="FOV"
            )
            ax.add_patch(fov_ellipse)

        
            if polygonal_obstacles:
                for poly in polygonal_obstacles:
                    local_polygon = poly - [slice_x_start, slice_y_start]
                    
                    ax.add_patch(Polygon(local_polygon, facecolor='orange', alpha=0.5, edgecolor='red',label="Obstacle"))


            half_avoid = avoidance_region // 2
            

            avoid_x_start_global = max(int(uav_current_pos[0]) - half_avoid, 0)
            avoid_y_start_global = max(int(uav_current_pos[1]) - half_avoid, 0)
            avoid_width_global = (min(int(uav_current_pos[0]) + half_avoid, map_width)) - avoid_x_start_global
            avoid_height_global = (min(int(uav_current_pos[1]) + half_avoid, map_height)) - avoid_y_start_global

            # 2. Transform the box's bottom-left corner to local view coordinates
            #    The width and height remain the same.
            avoid_box_local_xy = (
                avoid_x_start_global - slice_x_start,
                avoid_y_start_global - slice_y_start
            )

            # 3. Create the Rectangle patch
            avoid_box_patch = Rectangle(
                avoid_box_local_xy,
                avoid_width_global,
                avoid_height_global,
                edgecolor='yellow',
                facecolor='none',
                linestyle='--',
                linewidth=2,
                label='Avoidance Region' # This label will appear in the legend
            )
            
            # 4. Add the patch to the plot
            ax.add_patch(avoid_box_patch)


            
            # Formatting
            ax.set_title(f"Time: {t:.1f}s | UAV Velo: {v:.1f}m/s | Solve Time {solve_times[ind]*1000:.3f}ms")

            ax.set_xlim(0, slice_x_end - slice_x_start)
            ax.set_ylim(0, slice_y_end - slice_y_start)
            # ax.axis('equal')
            ax.legend(loc='upper right')
            plt.pause(0.01)

            if evader.finished:
                print(f"Evader reached the end of the path at time {t:.1f}s.")
                break

        print("Simulation finished.")
        plt.close('all')

        mean_solver_time = np.mean(solve_times)
        std_solver_time = np.std(solve_times)

        success_rate = sum(solver_failure)/len(solver_failure)
        
        
        print("\n--- Mission Performance ---")
        collision, fov_time,total_sim = analyze_simulation_results(
            uav_history, evader_history, obstacle_map, fov_ellipse_params, DT
        )
        print(f"Collision Occurred: {collision}")
        print(f"Total Time Evader in FOV: {fov_time:.2f}/{total_sim:.2f} seconds")
        print(f"Prediction time: {np.mean(prediction_times)}")
        print(f"Mean solve time: {np.mean(solve_times)}")
        print(f"Max solve time: {np.max(solve_times)}")

        # --- MODIFIED: Append all metrics to the table ---
        performance_table["filename"].append(os.path.basename(ground_truth_file_name))
        performance_table["mean_solve_time"].append(np.mean(solve_times))
        performance_table["median_solve_time"].append(np.median(solve_times))
        performance_table["max_solve_time"].append(np.max(solve_times))
        performance_table["min_solve_time"].append(np.min(solve_times))
        performance_table["solve_success_rate"].append(sum(solver_failure) / len(solver_failure))
        performance_table["collisions"].append(collision)
        performance_table["time_in_fov"].append(fov_time)
        performance_table["prediction_times"].append(np.mean(prediction_times))
    # plt.show()


    import pandas as pd
    df = pd.DataFrame(performance_table)
    print("\n--- Overall Performance Summary ---")
    print(df)
    df.to_csv("simulation_results.csv", index=False)