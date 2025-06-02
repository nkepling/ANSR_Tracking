import numpy as np
from dataclasses import dataclass
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
import math
import typing
from rrt import RRT

# --- Corrected Evader Class and Kinematic Functions ---

@dataclass
class Evader:
    """Represents the state of the evading agent."""
    x: float
    y: float
    theta: float  # Heading in radians
    v: float      # Speed in units/sec

    @property
    def vec(self) -> np.ndarray:
        """Returns the full state vector [x, y, theta, v]."""
        return np.array([self.x, self.y, self.theta, self.v])

    @property
    def pos(self) -> np.ndarray:
        """Returns the position vector [x, y]."""
        return np.array([self.x, self.y])

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


# def generate_goal_directed_trajectories(
#     initial_evader_state: Evader,
#     target_goals: list,  # List of [x, y] target coordinates
#     num_samples_per_goal: int,
#     num_time_steps: int,
#     delta_t: float,
#     speed_variation_std: float = 0.1,
#     heading_noise_std: float = np.deg2rad(15.0),
#     momentum_factor: float = 0.2,
#     koz_list: list = None, # List of KOZ vertex arrays
#     koz_avoidance_radius: float = 3.0, # How close to a KOZ center to trigger avoidance
#     koz_steer_strength: float = 0.5    # How strongly to steer away from KOZs (0-1)
# ) -> np.ndarray:
     
#     rand_area = [-20, 20, -12, 12] 

    
#     plans  = [ ]
#     for g in target_goals:
#         for i in range(num_samples_per_goal):
#             rrt = RRT(
#                 start=[initial_evader_state.x, initial_evader_state.y], # Start position from your simulate_with_jax.py
#                 goal=[g[0], g[1]],  # Goal (e.g., end of the corridor before koz3)
#                 rand_area=rand_area,
#                 obstacle_list=koz_list,
#                 expand_dis=2.0,       # Reduced expand distance for tighter spaces
#                 path_resolution=0.2,  # Finer path resolution
#                 max_iter=2000,        # Increased iterations for complex environments
#                 robot_radius=0.5      # Example robot radius
#             )

#             path = rrt.planning(animation=False)
#             plans.append(path)

#     truncated_plans = []
#     for p in plans:
#         if len(p) < num_time_steps:
#             diff = num_time_steps - len(p)
#             goal = p[-1]
#             buff = [goal for x in range(diff)]
#             truncated_plans.append(p + buff)
#         else:
#             truncated_plans.append(p[:num_time_steps])
        

#     return np.array(truncated_plans)


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
    # --- Simulation Parameters ---
    initial_state = Evader(x=0.0, y=0.0, theta=np.deg2rad(0), v=10.0)
    
    M = 20  # Number of predicted trajectories
    T = 10  # Number of time steps in each trajectory
    dt = 0.1 # Time delta

    koz1 = np.array([(5,2.5),(-15,2.5),(-15,1.5),(5,1.5)])
    koz2 = np.array([(4,10),(4,2.5),(5,2.5),(5,10)])
    koz3 = np.array([(5,-2.5),(-15,-2.5),(-15,-1.5),(5,-1.5)])
    koz4 = np.array([(4,-2.5),(4,-10),(5,-10),(5,-2.5)])
    koz5 = np.array([(8,10),(8,-10),(9,-10),(9,10)])
    # koz2 = np.array([(5,)])
    koz_list = [koz1,koz2,koz3,koz4,koz5]

    predicted_evader_trajectories = generate_goal_directed_trajectories(
                initial_evader_state = initial_state,
                target_goals =[(6,10),(6,-10)],  # List of [x, y] target coordinates
                num_samples_per_goal = 5,
                num_time_steps = T,
                delta_t = dt,
                speed_variation_std = 0.1,
                heading_noise_std = np.deg2rad(15.0),
                momentum_factor = 0.2,
                koz_list =koz_list, # List of KOZ vertex arrays
                koz_avoidance_radius  = 3.0, # How close to a KOZ center to trigger avoidance
                koz_steer_strength  = 0.5    # How strongly to steer away from KOZs (0-1)
            )

    # --- Generate a single Ground Truth trajectory for comparison (no noise) ---
    gt_trajectory = []
    current_gt_state = initial_state
    for _ in range(T):
        gt_trajectory.append(current_gt_state.pos)
        current_gt_state = forward(current_gt_state, dt)
    ground_truth_np = np.array(gt_trajectory)

    # --- Visualize the results ---

    # koz_verts = np.array([(1,1),(2,1),(2,-1),(1,-1),(1,1)])

  
    visualize(ground_truth=ground_truth_np, predicted_trajectories=predicted_evader_trajectories,koz_list=koz_list)