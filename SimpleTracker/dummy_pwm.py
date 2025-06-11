import numpy as np
from dataclasses import dataclass
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
import math
import typing
from rrt import RRT
import copy
import casadi as ca
import time
from scipy.interpolate import splprep, splev




def interpolate_by_time(coarse_waypoints, dt, avg_velocity):
    """
    Generates a dense set of waypoints by placing points roughly dt seconds
    apart, based on a given average velocity.

    Args:
        coarse_waypoints (np.ndarray): An array of shape (M, 2) defining the coarse path.
        dt (float): The desired time step in seconds between fine waypoints.
        avg_velocity (float): The assumed average velocity along the path.

    Returns:
        np.ndarray: A dense array of waypoints.
    """
    if dt <= 0 or avg_velocity <= 0:
        raise ValueError("dt and avg_velocity must be positive.")

    fine_waypoints = [coarse_waypoints[0]]

    for i in range(len(coarse_waypoints) - 1):
        start_point = coarse_waypoints[i]
        end_point = coarse_waypoints[i+1]
        
        # Calculate the geometry of the segment
        segment_vector = end_point - start_point
        segment_length = np.linalg.norm(segment_vector)
        
        if segment_length < 1e-6:  # Skip zero-length segments
            continue
            
        direction = segment_vector / segment_length
        
        # Calculate how many full steps of size dt fit in the segment
        time_to_travel = segment_length / avg_velocity
        num_intermediate_points = int(np.floor(time_to_travel / dt))

        # Place the intermediate points
        for j in range(1, num_intermediate_points + 1):
            travel_dist = j * dt * avg_velocity
            new_point = start_point + travel_dist * direction
            fine_waypoints.append(new_point)
        
        # Always include the coarse waypoint to ensure path completion
        fine_waypoints.append(end_point)

    # Remove duplicate points that may arise from the logic
    # by checking the distance between consecutive points.
    unique_fine_waypoints = [fine_waypoints[0]]
    for i in range(1, len(fine_waypoints)):
        if np.linalg.norm(fine_waypoints[i] - unique_fine_waypoints[-1]) > 1e-6:
            unique_fine_waypoints.append(fine_waypoints[i])
            
    return np.array(unique_fine_waypoints)


def generate_bspline_trajectory(waypoints, num_points=200, spline_degree=3):
    """
    Generates a smooth trajectory using B-splines that passes through given waypoints.

    Args:
        waypoints (np.ndarray): A (num_waypoints, 2) array of [x, y] points.
        num_points (int): The number of points to generate for the final trajectory.
        spline_degree (int): The degree of the spline (cubic is a good default).

    Returns:
        A dictionary containing the state (x, y, theta) and control (v, omega)
        trajectories.
    """
    # 1. --- Parameterize the Path ---
    # We use cumulative distance along the path as the parameter 'u'.
    # This helps create a more uniform interpolation.
    distance = np.cumsum(np.sqrt(np.sum(np.diff(waypoints, axis=0)**2, axis=1)))
    distance = np.insert(distance, 0, 0)
    
    # 2. --- Fit the B-spline ---
    # splprep finds the knots and coefficients for a spline that approximates the path.
    # 's' is a smoothing factor. s=0 means the spline must pass through all points.
    tck, u = splprep([waypoints[:, 0], waypoints[:, 1]], u=distance, k=spline_degree, s=0)

    # 3. --- Evaluate the Spline and its Derivatives ---
    u_fine = np.linspace(0, distance[-1], num_points)
    
    # Evaluate spline for position (derivative=0)
    x_fine, y_fine = splev(u_fine, tck, der=0)
    
    # Evaluate for velocity (derivative=1)
    vx, vy = splev(u_fine, tck, der=1)
    
    # Evaluate for acceleration (derivative=2)
    ax, ay = splev(u_fine, tck, der=2)

    # 4. --- Calculate Kinematic States ---
    theta = np.arctan2(vy, vx)
    
    # Linear velocity (speed). Note this is NOT constant.
    # The speed is determined by the spline parameterization.
    v = np.sqrt(vx**2 + vy**2)
    
    # Angular velocity (omega)
    # The formula is: omega = (vx*ay - vy*ax) / (vx^2 + vy^2)
    # Add a small epsilon to avoid division by zero if the robot stops.
    omega = (vx * ay - vy * ax) / (vx**2 + vy**2 + 1e-6)
    
    return {
        "x": x_fine, "y": y_fine, "theta": theta,
        "v": v, "omega": omega,
        "waypoints": waypoints
    }


def plot_bspline_results(results):
    """Helper function to visualize the B-spline results."""
    waypoints = results["waypoints"]
    x, y, theta = results["x"], results["y"], results["theta"]
    
    plt.figure(figsize=(10, 8))
    plt.plot(waypoints[:, 0], waypoints[:, 1], 'ro', markersize=10, label='Waypoints')
    plt.plot(x, y, 'b-', linewidth=2, label='B-Spline Trajectory')

    # Add arrows to show heading
    for i in range(0, len(x), 20):
        plt.arrow(x[i], y[i],
                  0.5 * np.cos(theta[i]), 0.5 * np.sin(theta[i]),
                  head_width=0.15, fc='b', ec='b')

    plt.title("B-Spline Trajectory Generation")
    plt.xlabel("X Position (m)")
    plt.ylabel("Y Position (m)")
    plt.legend()
    plt.grid(True)
    plt.axis('equal')
    plt.show()





class Evader:
    """
    Represents the state and path-following logic for an evading agent.
    It uses a look-ahead controller to follow a given path.
    """
    def __init__(self, x: float, y: float, theta: float, v: float, path: np.ndarray = None, lookahead_distance: float = 5.0):
        # --- State Attributes ---
        self.x = x
        self.y = y
        self.theta = theta
        self.v = v
        
        # --- Path-Following Attributes ---
        self.path = path
        self.lookahead_distance = lookahead_distance
        self.finished = False

        if self.path is not None:
            self._preprocess_path()

    def _preprocess_path(self):
        """Pre-calculates lengths for efficient path tracking."""
        if not isinstance(self.path, np.ndarray) or self.path.ndim != 2 or self.path.shape[1] != 2:
            raise ValueError("Path must be a NumPy array of shape (N, 2)")
        
        segment_vectors = np.diff(self.path, axis=0)
        self.segment_lengths = np.linalg.norm(segment_vectors, axis=1)
        self.cumulative_lengths = np.insert(np.cumsum(self.segment_lengths), 0, 0)
        self.total_path_length = self.cumulative_lengths[-1]

    @property
    def vec(self) -> np.ndarray:
        return np.array([self.x, self.y, self.theta, self.v])

    @property
    def pos(self) -> np.ndarray:
        return np.array([self.x, self.y])

    def update(self, dt: float):
        if self.path is None or self.finished:
            return

        target_point = self._get_lookahead_point()
        
        if target_point is None:
            self.finished = True
            return

        dx = target_point[0] - self.x
        dy = target_point[1] - self.y
        target_theta = math.atan2(dy, dx)
        
        self.theta = target_theta
        self.x += self.v * math.cos(self.theta) * dt
        self.y += self.v * math.sin(self.theta) * dt

    def _get_lookahead_point(self) -> np.ndarray:
        # ### FIX STARTS HERE ###
        # This whole block is rewritten to be correct and more robust.
        
        min_dist_to_path = float('inf')
        closest_point_on_path = None
        closest_segment_index = 0

        # 1. Find the point on the path polyline closest to the evader
        for i in range(len(self.path) - 1):
            p1 = self.path[i]
            p2 = self.path[i+1]
            
            line_vec = p2 - p1
            point_vec = self.pos - p1
            line_len_sq = np.dot(line_vec, line_vec)
            
            t = 0.0
            if line_len_sq > 0:
                t = max(0, min(1, np.dot(point_vec, line_vec) / line_len_sq))
                
            closest_point_on_segment = p1 + t * line_vec
            dist = np.linalg.norm(self.pos - closest_point_on_segment)
            
            if dist < min_dist_to_path:
                min_dist_to_path = dist
                closest_point_on_path = closest_point_on_segment
                closest_segment_index = i

        # 2. Calculate how far along the path this closest point is
        dist_to_closest_point = self.cumulative_lengths[closest_segment_index] + \
                                np.linalg.norm(closest_point_on_path - self.path[closest_segment_index])

        # 3. Find the target distance by adding the lookahead distance
        target_dist = dist_to_closest_point + self.lookahead_distance
        
        if target_dist >= self.total_path_length:
            return None

        # 4. Find which segment the look-ahead point is on
        target_segment_index = np.searchsorted(self.cumulative_lengths, target_dist, side='right') - 1
        
        # 5. Interpolate to find the exact look-ahead point
        dist_into_segment = target_dist - self.cumulative_lengths[target_segment_index]
        start_point = self.path[target_segment_index]
        
        # Handle cases where segment length might be zero
        segment_len = self.segment_lengths[target_segment_index]
        if segment_len <= 1e-6:
             return start_point

        end_point = self.path[target_segment_index + 1]
        proportion = dist_into_segment / segment_len
        lookahead_point = start_point + proportion * (end_point - start_point)
        # ### FIX ENDS HERE ###
        
        return lookahead_point
    
    def get_predicted_trajectory(self, N, dt):
        """
        Predicts the evader's future path by simulating its movement.
        """
        preds = np.zeros((2, N))

        temp_evader = Evader(
            x=self.x, 
            y=self.y, 
            theta=self.theta, 
            v=self.v, 
            path=self.path, 
            lookahead_distance=self.lookahead_distance
        )
        
        if temp_evader.path is None:
            for i in range(N):
                temp_evader.update(dt)
                preds[:, i] = temp_evader.pos
            return preds

        for i in range(N):
            temp_evader.update(dt)
            preds[:, i] = temp_evader.pos

        return preds
        
def forward(current_state: Evader, delta_t: float) -> Evader:
    """
    FIXED: Move evader forward with correct simple Euler integration.
    This model assumes constant velocity and heading over the small time step delta_t.
    """
    # Calculate the change in position based on current heading and speed
    dx = current_state.v * np.cos(current_state.theta) * delta_t
    dy = current_state.v * np.sin(current_state.theta) * delta_t
    
    # Return a new Evader state object with the updated position
    return Evader(
        x=current_state.x + dx,
        y=current_state.y + dy,
        theta=current_state.theta, # Heading remains constant for a straight-away model
        v=current_state.v          # Speed remains constant for a straight-away model
    )

def get_straight_away_trajectories(initial_evader_state: Evader,
                                   num_trajectories: int,
                                   num_time_steps: int,
                                   delta_t: float,
                                   speed_variation_std: float = 0.1,
                                   heading_variation_std: float = 0.01) -> np.ndarray:
    """
    FIXED: Generates M (num_trajectories) plausible straight-line trajectories.
    Each trajectory is a sequence of T (num_time_steps) positions.
    Returns a 3D numpy array of shape (M, T, 2).
    """
    all_trajectories_list = []

    for _ in range(num_trajectories):
        # Create a perturbed initial state for this specific trajectory
        # This simulates uncertainty in the evader's current speed and heading
        perturbed_speed = max(0, initial_evader_state.v + np.random.normal(0, speed_variation_std))
        perturbed_heading = initial_evader_state.theta + np.random.normal(0, heading_variation_std)
        
        # This is the starting state for this one simulated future
        current_state = Evader(x=initial_evader_state.x,
                               y=initial_evader_state.y,
                               theta=perturbed_heading,
                               v=perturbed_speed)
        
        trajectory_points_list = []
        for _ in range(num_time_steps):
            # Add the current (x, y) position to our list of points
            trajectory_points_list.append(current_state.pos)
            # Propagate the state forward to the next time step
            current_state = forward(current_state, delta_t)
            
        all_trajectories_list.append(np.array(trajectory_points_list))
        
    return np.array(all_trajectories_list)




def generate_goal_directed_trajectories(
    initial_evader_state: Evader,
    target_goals: list,  # List of [x, y] target coordinates
    num_samples_per_goal: int,
    num_time_steps: int, # Desired fixed length of output trajectories
    delta_t: float,
    speed_variation_std: float = 0.1,
    heading_noise_std: float = np.deg2rad(15.0), # Not used by RRT directly, RRT has its own randomness
    momentum_factor: float = 0.2,          # Not used by RRT directly
    koz_list: list = None, # List of KOZ vertex arrays, passed to RRT as obstacle_list
    koz_avoidance_radius: float = 3.0, # Not used by RRT directly (RRT uses robot_radius)
    koz_steer_strength: float = 0.5    # Not used by RRT directly
) -> np.ndarray:
    """
    Generates time-discretized trajectories of fixed length (num_time_steps)
    by finding a path with RRT to target_goals and then resampling it based on velocity.
    """
    
    # RRT parameters (as hardcoded in your snippet, ideally these would be params too)
    rand_area = [-20, 20, -12, 12] # [min_x, max_x, min_y, max_y] for RRT sampling
    rrt_expand_dis = 2.0
    rrt_path_resolution = 0.2  # RRT's internal step resolution
    rrt_max_iter = 1000        # Reduced for potentially faster prediction generation
    rrt_robot_radius = 0.1     # Small radius for evader (point mass assumption for collision)
    rrt_goal_sample_rate = 10  # Bias RRT towards goal more often

    # This list will store the final time-discretized trajectories
    time_discretized_plans = [] 

    # distance based trajctory optim

    distances_to_goals = [np.linalg.norm(np.array(g) - initial_evader_state.pos) for g in target_goals]
    epsilon = 1e-3 
    attractiveness_scores = [1.0 / (d + epsilon) for d in distances_to_goals]
    sum_scores = sum(attractiveness_scores)
    goal_selection_probabilities = []
    if sum_scores < 1e-6: # Avoid division by zero if all goals are extremely far or scores are zero
        # Fallback to uniform probability
        if target_goals:
             goal_selection_probabilities = [1.0 / len(target_goals)] * len(target_goals)
    else:
        goal_selection_probabilities = [score / sum_scores for score in attractiveness_scores]

    total_trajectories_to_generate = len(target_goals) * num_samples_per_goal

    for _ in range(total_trajectories_to_generate):
        # --- MODIFICATION: Select a target goal based on calculated probabilities ---
        if not goal_selection_probabilities: # Should not happen if target_goals is not empty
            selected_target_goal_np = np.array(target_goals[0]) # Fallback
        else:
            chosen_goal_index = np.random.choice(len(target_goals), p=goal_selection_probabilities)
            selected_target_goal_np = np.array(target_goals[chosen_goal_index])
        # --- END MODIFICATION ---
        perturbed_speed = max(0.1, initial_evader_state.v + np.random.normal(0, speed_variation_std))
        
        rrt = RRT(
            start=[initial_evader_state.x, initial_evader_state.y],
            goal=[selected_target_goal_np[0], selected_target_goal_np[1]],
            rand_area=rand_area,
            obstacle_list=koz_list if koz_list is not None else [], # Pass KOZs to RRT
            expand_dis=rrt_expand_dis,
            path_resolution=rrt_path_resolution,
            max_iter=rrt_max_iter,
            robot_radius=rrt_robot_radius,
            goal_sample_rate=rrt_goal_sample_rate
        )

        # rrt_path is a list of [x,y] waypoints from RRT, e.g., [[x_start,y_start], ..., [x_goal,y_goal]]
        rrt_path_waypoints = rrt.planning(animation=False) 
        
        current_time_discretized_trajectory = np.zeros((num_time_steps, 2))

        if rrt_path_waypoints is None or len(rrt_path_waypoints) < 2:
            # Fallback: RRT failed or path is too short (e.g. start is goal).
            # Generate a simple straight path towards the goal using evader's kinematics.
            # print(f"RRT failed or path too short for goal {g}, sample {i}. Using kinematic fallback.")
            temp_state = Evader(x=initial_evader_state.x, y=initial_evader_state.y, 
                                theta=initial_evader_state.theta, v=perturbed_speed)
            for k_ts in range(num_time_steps):
                current_time_discretized_trajectory[k_ts, :] = temp_state.pos
                dir_vec_to_goal = selected_target_goal_np - temp_state.pos
                if np.linalg.norm(dir_vec_to_goal) > 0.1: # Only update theta if not at goal
                        temp_state = Evader(x=temp_state.x, y=temp_state.y, 
                                            theta=np.arctan2(dir_vec_to_goal[1], dir_vec_to_goal[0]), 
                                            v=temp_state.v)
                temp_state = forward(temp_state, delta_t)
        else:
            # Valid RRT path found, now resample it based on time and velocity
            rrt_nodes_np = np.array(rrt_path_waypoints) # Shape (N_rrt_nodes, 2)

            # Calculate cumulative distances along the RRT path segments
            segment_vectors = np.diff(rrt_nodes_np, axis=0)
            segment_lengths = np.linalg.norm(segment_vectors, axis=1)
            
            if len(segment_lengths) == 0: # Path is a single point (start might be very close/at goal)
                cumulative_rrt_path_lengths = np.array([0.0])
            else:
                cumulative_rrt_path_lengths = np.concatenate(([0.0], np.cumsum(segment_lengths)))
            
            total_rrt_path_length = cumulative_rrt_path_lengths[-1]

            for k_time_step in range(num_time_steps):
                # Distance evader should have traveled along the path by this time step
                dist_evader_should_travel = perturbed_speed * (k_time_step * delta_t)

                if dist_evader_should_travel >= total_rrt_path_length and total_rrt_path_length > 1e-6 :
                    # Evader has reached or passed the end of the RRT path, stays at RRT goal
                    current_time_discretized_trajectory[k_time_step, :] = rrt_nodes_np[-1, :]
                elif total_rrt_path_length <= 1e-6: # RRT path is effectively a single point
                    current_time_discretized_trajectory[k_time_step, :] = rrt_nodes_np[0, :]
                else:
                    # Find which RRT path segment the evader is on at this distance
                    # 'right' means if dist is equal to a cum_length, it picks the segment *after*
                    segment_idx = np.searchsorted(cumulative_rrt_path_lengths, 
                                                    dist_evader_should_travel, side='right') - 1
                    segment_idx = max(0, segment_idx) 
                    # Ensure segment_idx is valid for segment_lengths and rrt_nodes_np
                    # It should not exceed len(segment_lengths) - 1
                    segment_idx = min(segment_idx, len(segment_lengths) - 1 if len(segment_lengths) > 0 else 0)


                    p1 = rrt_nodes_np[segment_idx]
                    # Guard against segment_idx+1 being out of bounds if path was very short
                    p2_idx = min(segment_idx + 1, len(rrt_nodes_np) - 1)
                    p2 = rrt_nodes_np[p2_idx]
                    
                    dist_covered_at_segment_start = cumulative_rrt_path_lengths[segment_idx]
                    dist_into_current_segment = dist_evader_should_travel - dist_covered_at_segment_start
                    
                    current_segment_actual_length = segment_lengths[segment_idx] if segment_idx < len(segment_lengths) else 0


                    if current_segment_actual_length < 1e-6: # Segment is essentially a point, or past end
                        fraction_along_segment = 0.0 if dist_into_current_segment <=0 else 1.0
                    else:
                        fraction_along_segment = dist_into_current_segment / current_segment_actual_length
                    
                    fraction_along_segment = np.clip(fraction_along_segment, 0.0, 1.0)

                    interpolated_point = p1 + fraction_along_segment * (p2 - p1)
                    current_time_discretized_trajectory[k_time_step, :] = interpolated_point
        
        time_discretized_plans.append(current_time_discretized_trajectory)
            
    if not time_discretized_plans:
        # Fallback if target_goals was empty or num_samples_per_goal was 0
        dummy_traj = np.tile(initial_evader_state.pos, (num_time_steps, 1))
        num_to_generate = num_samples_per_goal * len(target_goals) if (target_goals and num_samples_per_goal > 0) else 1
        if num_to_generate == 0: num_to_generate = 1 # Ensure at least one trajectory if called
        return np.array([dummy_traj] * num_to_generate)

    return np.array(time_discretized_plans)

# --- Visualization and Demonstration (using the function from your previous code) ---



def plot_results(results):
    """Helper function to visualize the optimization results."""
    state = results["state"]
    waypoints = results["waypoints"]
    ref_path = results["reference_path"]
    
    plt.figure(figsize=(10, 8))
    
    # Plot original sparse waypoints
    plt.plot(waypoints[:, 0], waypoints[:, 1], 'ro', markersize=10, label='Waypoints')
    
    # Plot the dense reference trajectory
    plt.plot(ref_path[0, :], ref_path[1, :], 'g--', label='Reference Path')

    # Plot the final optimized trajectory
    plt.plot(state[0, :], state[1, :], 'b-', linewidth=2, label='Optimized Trajectory')

    # Add arrows to show heading
    for i in range(0, state.shape[1], 10):
        plt.arrow(state[0, i], state[1, i],
                  0.5 * np.cos(state[2, i]), 0.5 * np.sin(state[2, i]),
                  head_width=0.15, fc='b', ec='b')

    plt.title("Unicycle Trajectory Optimization")
    plt.xlabel("X Position (m)")
    plt.ylabel("Y Position (m)")
    plt.legend()
    plt.grid(True)
    plt.axis('equal')
    plt.show()


def visualize(ground_truth: np.ndarray, predicted_trajectories: np.ndarray,koz_list:typing.Union[list,None]=None):
    """Simple visualization function to plot the results."""
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.set_aspect('equal', adjustable='box')
    
    # Plot Ground Truth
    ax.plot(ground_truth[:, 0], ground_truth[:, 1], 'r-', label='Ground Truth', linewidth=2.5)
    ax.plot(ground_truth[0, 0], ground_truth[0, 1], 'ro', markersize=8, label='Start')
    
    # Plot Predicted Trajectories
    for i, trajectory in enumerate(predicted_trajectories):
        label = 'Predicted' if i == 0 else None
        ax.plot(trajectory[:, 0], trajectory[:, 1], 'k--', alpha=0.1, label=label)

    if koz_list is not None:
        for koz in koz_list:
            poly = Polygon(koz, closed=True, color='orange', alpha=0.4, label='KOZ')
            ax.add_patch(poly)

    ax.set_xlim(-20,20)
    ax.set_ylim(-20,20)

    ax.set_xlabel('X Position')
    ax.set_ylabel('Y Position')
    ax.set_title('Straight-Away Trajectory Predictions')
    ax.legend()
    ax.grid(True)
    plt.show()

if __name__ == '__main__':
    waypoints = np.array([
        [0, 0],
        [5, 2],
        [6, 8],
        [2, 7],
        [-2, 4],
        [0, 0]
    ])

    # Generate and solve the trajectory

    v = 10


    
    # Plot the outcome

    start = time.time()
    fine_waypoints_by_time = interpolate_by_time(waypoints, dt=0.1, avg_velocity=v)
    print("get traj time", time.time() - start)


    fig, ax2 = plt.subplots()
    ax2.plot(waypoints[:, 0], waypoints[:, 1], 'ro-', markersize=10, linewidth=2, label='Coarse Path')
    ax2.plot(fine_waypoints_by_time[:, 0], fine_waypoints_by_time[:, 1], 'gx', markersize=5, label=f'Fine Points ({len(fine_waypoints_by_time)})')
    ax2.set_title(f"Interpolation by Time (dt={0.1}s, v={v}m/s)")
    ax2.set_xlabel("X Position (m)")
    ax2.set_ylabel("Y Position (m)")
    ax2.legend()
    ax2.grid(True)
    ax2.axis('equal')
    
    plt.suptitle("Waypoint Interpolation Methods")
    plt.show()
    
    start = time.time()
    results = generate_bspline_trajectory(waypoints)
    print("get traj time", time.time() - start)
    
    # Plot the outcome
    plot_bspline_results(results)



