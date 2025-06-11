# In file: simulate_with_jax.py

import numpy as np
from dummy_pwm import Evader, forward, generate_goal_directed_trajectories
import animator
# --- MODIFICATION: Import your new CasADi-based solver ---
from opti_lib import solve_uav_tracking
import time

# Note: JAX is no longer used for the solver, but dummy_pwm might use it.
import jax
jax.config.update("jax_enable_x64", True)


def run_simulation():
    """
    Runs a multi-step pursuit-evasion simulation using the CasADi-based solver.
    """
    # --- 1. Simulation Setup ---
    T = 15      # Planning horizon
    dt = 0.5    # Time step duration
    M = 5       # Number of evader trajectories to sample per goal
    sim_steps = 50

    target_goals = [(6, -11), (6, 11)]

    # Agent parameters
    pursuer_max_velo = 8.0
    evader_velo = 8.0
    
    # --- MODIFICATION: Set penalty weights for the new solver ---
    obstacle_penalty_weight = 10000.0
    tracking_weight = 1.0 # Default tracking weight
    # NOTE: The new solver doesn't handle polygonal KOZs, only circular obstacles.
    # The pursuer will not actively avoid the KOZs defined below in its planning.
    
    # Evader ground truth turning maneuver parameters
    EVADER_TURN_X_THRESHOLD = 6
    EVADER_TURN_DEGREES = 90

    # --- 2. Initialize Agent States ---
    pursuer_current_pos = np.array([-18.0, 0.0])
    evader_current_state = Evader(x=-15.0, y=0.0, theta=np.deg2rad(0), v=evader_velo)
    
    # --- 3. Log History for Final Visualization ---
    pursuer_history = [pursuer_current_pos]
    evader_history = [evader_current_state.pos]
    pursuer_plans_history = []
    evader_predictions_history = []
    solve_time_history = []

    # --- Define Obstacles and KOZs ---
    # The new solver uses circular obstacles in the format: [center_x, center_y, radius]
    obstacles = [
        [0.0, 0.0, 1.0],
        [3.0, 0.0, 1.0]
    ]

    # KOZ definitions (used for evader prediction, not pursuer planning with this solver)
    koz1 = np.array([(5, 2.5), (-15, 2.5), (-15, 1.5), (5, 1.5)])
    koz2 = np.array([(4, 10), (4, 2.5), (5, 2.5), (5, 10)])
    koz3 = np.array([(5, -2.5), (-15, -2.5), (-15, -1.5), (5, -1.5)])
    koz4 = np.array([(4, -2.5), (4, -10), (5, -10), (5, -2.5)])
    koz5 = np.array([(8, 10), (8, -10), (9, -10), (9, 10)])
    koz_list = [koz1, koz2, koz3, koz4, koz5]
    
    print("Starting simulation with CasADi-based solver...")
    # --- 4. Main Simulation Loop ---
    evader_has_made_ground_truth_turn = False
    for i in range(sim_steps):
        print(f"Running step {i+1}/{sim_steps}...")

        # --- a. PREDICT ---
        # Generate multiple possible evader trajectories
        predicted_evader_trajectories = generate_goal_directed_trajectories(
            initial_evader_state=evader_current_state,
            target_goals=target_goals,
            num_samples_per_goal=M,
            num_time_steps=T,
            delta_t=dt,
            speed_variation_std=0.1,
            heading_noise_std=np.deg2rad(15.0),
            momentum_factor=0.2,
            koz_list=koz_list,
            koz_avoidance_radius=3.0,
            koz_steer_strength=0.5
        )
        evader_predictions_history.append(predicted_evader_trajectories)

        # --- b. PLAN (using the new CasADi solver) ---
        
        # --- MODIFICATION: Adapt data for the new solver ---
        # Since the new solver takes a single reference trajectory, we average the predictions
        # to get a more robust "mean" expectation of the evader's path.
        mean_evader_trajectory = np.mean(predicted_evader_trajectories, axis=0)
        # The solver expects shape (2, T), so we transpose the trajectory.
        reference_trajectory_for_solver = mean_evader_trajectory.T

        start_time = time.perf_counter()
        
        # Call the new solver with the correct arguments
        control_inputs, pursuer_plan_T,  = solve_uav_tracking(
            initial_pos=pursuer_current_pos,
            reference_trajectory=reference_trajectory_for_solver,
            max_speed=pursuer_max_velo,
            obstacles=obstacles,
            obstacle_weight=obstacle_penalty_weight,
            N=T, # Horizon
            dt=dt
        )

        end_time = time.perf_counter()
        solve_time = end_time - start_time
        solve_time_history.append(solve_time)

        if pursuer_plan_T is None:
            print("Solver failed to find a plan. Ending simulation.")
            break
        
        # The solver returns shape (2, T+1). Transpose to (T+1, 2) for consistency.
        pursuer_plan = pursuer_plan_T.T 
        pursuer_plans_history.append(pursuer_plan)

        # --- c. EVADER MANEUVER (Ground Truth Evader) ---
        if not evader_has_made_ground_truth_turn and evader_current_state.x >= EVADER_TURN_X_THRESHOLD:
            print(f"--- EVADER (Ground Truth) MAKING A TURN AT STEP {i}, x-pos: {evader_current_state.x:.2f} ---")
            turn_direction = np.random.choice([-1, 1])
            evader_current_state = Evader(
                x=evader_current_state.x,
                y=evader_current_state.y,
                theta=evader_current_state.theta + turn_direction * np.deg2rad(EVADER_TURN_DEGREES),
                v=evader_current_state.v
            )
            evader_has_made_ground_truth_turn = True

        # --- d. ACT & UPDATE STATE ---
        # Apply the FIRST step of the optimal plan (MPC logic)
        # The plan is for T+1 states (0 to T). We move to state 1.
        # pursuer_current_pos = pursuer_plan[1, :]
        pursuer_current_pos = pursuer_current_pos + dt * control_inputs
        evader_current_state = forward(evader_current_state, dt)

        pursuer_history.append(pursuer_current_pos)
        evader_history.append(evader_current_state.pos)

        # Check for termination condition
        if min([np.linalg.norm(np.array([evader_current_state.x, evader_current_state.y]) - np.array(g)) for g in target_goals]) < 1:
            print("Evader reached a target goal. Ending simulation.")
            break

    print("Simulation finished.")
    return {
        "pursuer_history": np.array(pursuer_history),
        "evader_history": np.array(evader_history),
        "pursuer_plans": pursuer_plans_history,
        "evader_predictions": evader_predictions_history,
        "solve_times": solve_time_history,
        "keep_out_zones": koz_list,
        "obstacles": obstacles # Pass circular obstacles for animation
    }


if __name__ == "__main__":
    simulation_data = run_simulation()

    solve_times_data = simulation_data.get("solve_times")
    if solve_times_data and len(solve_times_data) > 0:
        print("\n--- Solver Performance ---")
        print(f"Average solve time: {np.mean(solve_times_data):.4f} seconds")
        print(f"Median solve time:  {np.median(solve_times_data):.4f} seconds")
        print(f"Max solve time:     {np.max(solve_times_data):.4f} seconds")
        print(f"Min solve time:     {np.min(solve_times_data):.4f} seconds")
        print(f"Standard deviation: {np.std(solve_times_data):.4f} seconds")

    animator.create_animation(simulation_data, "casadi_simulation.gif")