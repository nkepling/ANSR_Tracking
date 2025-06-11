# It's assumed your fov_solver.py and utils.py are in the same directory
# so they can be imported.

import casadi as ca
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Ellipse
import matplotlib.animation as animation
import time
import math



# --- Main Simulation Script ---
if __name__ == "__main__":

    # --- 1. SETUP ENVIRONMENT AND EVADER PATH ---
    segmap_file = "city_1000_1000_seg_segids.npz"
    obstacle_map_file = "city_1000_1000.npz"
    
    roads, _ = load_roads(segmap_file, visualize=False)
    roads = np.rot90(roads.T)
    
    obstacle_map, _, _ = load_obstacle_map(obstacle_map_file, depth=10)
    obstacle_map = np.rot90(obstacle_map.T)

    evader_gt_path = get_ground_truth_evader_path(roads)

    # --- 2. SETUP MPC SIMULATION ---
    DT = 0.2  # Using a slightly larger timestep for the large map
    SIM_STEPS = 500
    N_horizon = 20

    # Agent Initialization
    pursuer_start_pos = np.array([380.0, 500.0]) # Start near the evader
    pursuer_initial_heading = np.deg2rad(-90)
    uav_state = np.array([pursuer_start_pos[0], pursuer_start_pos[1], pursuer_initial_heading])
    
    evader = (path=evader_gt_path, speed=10.0, lookahead_distance=15.0)

    uav_max_velocity = 35.0 # Pursuer is faster
    uav_max_angular_velocity = np.deg2rad(90)
    fov_ellipse_params = {'a': 80.0, 'b': 40.0} # Larger FOV for the larger map
    STANDOFF_DISTANCE = 50.0

    # --- IMPORTANT: Ignoring obstacles in the MPC solver for now ---
    mpc_obstacles = [] 
    obstacle_weight = 0.0
    fov_weight = 50.0

    solver_options = {"print_level": 0, "acceptable_tol": 1e-2, "max_cpu_time": 0.1}

    # Data Logging
    uav_history = [uav_state.copy()]
    evader_history = [evader.pos.copy()]
    planned_trajectories = []
    prev_state_sol, prev_control_sol = None, None

    # --- 3. RUN MPC SIMULATION LOOP ---
    print("\nRunning MPC simulation on real map (tracking only)...")
    for i in range(SIM_STEPS):
        # Create a "perfect" prediction of the evader's path for the MPC
        current_evader_idx = evader.path_index
        prediction_end_idx = min(len(evader_gt_path), current_evader_idx + N_horizon + 1)
        evader_prediction = evader_gt_path[current_evader_idx:prediction_end_idx].T
        
        if evader_prediction.shape[1] < N_horizon + 1:
            print("Evader nearing end of path, stopping simulation.")
            break

        # Warm start guess
        state_guess, control_guess = None, None
        if prev_state_sol is not None:
            state_guess = np.roll(prev_state_sol, -1, axis=1)
            state_guess[:, -1] = state_guess[:, -2]
        if prev_control_sol is not None:
            control_guess = np.roll(prev_control_sol, -1, axis=1)
            control_guess[:, -1] = control_guess[:, -2]

        # Solve for pursuer's optimal move
        optimal_controls, planned_state = solve_uav_tracking_with_fov(
            uav_state, evader_prediction, uav_max_velocity, uav_max_angular_velocity,
            mpc_obstacles, obstacle_weight, fov_ellipse_params, fov_weight, STANDOFF_DISTANCE,
            solver_options, state_guess, control_guess, N_horizon, DT
        )
        
        # Update states
        prev_state_sol, prev_control_sol = planned_state, optimal_controls
        v, omega = optimal_controls[:, 0]
        uav_state[0] += v * np.cos(uav_state[2]) * DT
        uav_state[1] += v * np.sin(uav_state[2]) * DT
        uav_state[2] += omega * DT
        evader.update(DT)
        
        # Log data
        uav_history.append(uav_state.copy())
        evader_history.append(evader.pos.copy())
        planned_trajectories.append(planned_state)

        if evader.finished:
            print("Evader reached end of path.")
            break
            
    print("Simulation complete. Creating animation...")
    
    # --- 4. CREATE ANIMATION ---
    fig, ax = plt.subplots(figsize=(12, 12))
    uav_path = np.array(uav_history)
    evader_path = np.array(evader_history)

    def update(frame):
        ax.clear()
        # Display the map as the background
        ax.imshow(obstacle_map, cmap='gray', origin='lower', alpha=0.6)
        
        # Plot full historical paths
        ax.plot(evader_gt_path[:, 0], evader_gt_path[:, 1], ':', color='gold', lw=2, label="Evader's Full Path")
        ax.plot(evader_path[:frame+1, 0], evader_path[:frame+1, 1], 'r--', label='Evader History')
        ax.plot(uav_path[:frame+1, 0], uav_path[:frame+1, 1], 'b-', label='UAV History')
        
        # Plot current agent positions
        ax.plot(evader_path[frame, 0], evader_path[frame, 1], 'ro', markersize=10, label="Evader")
        uav_current_pos = uav_path[frame, :2]
        uav_current_theta = uav_path[frame, 2]
        ax.plot(uav_current_pos[0], uav_current_pos[1], 'bo', markersize=10, label="UAV")
        
        # Plot the UAV's plan
        if frame < len(planned_trajectories):
            plan = planned_trajectories[frame]
            ax.plot(plan[0, :], plan[1, :], 'g-+', alpha=0.7, label='UAV Plan')
        
        # Plot the FOV ellipse
        a,b = fov_ellipse_params['a'], fov_ellipse_params['b']
        heading_vec = np.array([np.cos(uav_current_theta), np.sin(uav_current_theta)])
        ellipse_center = uav_current_pos + a * heading_vec
        fov_ellipse = Ellipse(xy=ellipse_center, width=2*a, height=2*b, angle=np.rad2deg(uav_current_theta),
                              edgecolor='b', facecolor='blue', alpha=0.15)
        ax.add_patch(fov_ellipse)

        # Formatting
        ax.set_title(f"MPC Tracking on Real Map | Time: {frame*DT:.1f}s")
        ax.legend(loc='upper right')
        ax.invert_yaxis() # Match the image coordinate system

    ani = animation.FuncAnimation(fig, update, frames=len(uav_history), repeat=False, interval=int(1000*DT))
    ani.save('real_map_tracking_simulation.gif', writer='pillow', fps=int(1/DT))
    print("Animation saved to real_map_tracking_simulation.gif")