# In file: simulate_with_jax.py

import numpy as np
from dummy_pwm import Evader, forward, get_straight_away_trajectories
import animator
# --- MODIFICATION: Import the new class-based JAX solver ---
import track_ipotp as tracker

# --- Use JAX for 64-bit precision, which is good practice when working with IPOPT ---
import jax
jax.config.update("jax_enable_x64", True)
import time


def run_simulation():
    """
    Runs a multi-step pursuit-evasion simulation using the advanced JAX+IPOPT solver.
    """
    # --- 1. Simulation Setup ---
    T = 10    # Planning horizon (can be longer with a fast solver)
    dt = 0.1    # Time step duration
    M = 5      # Number of evader trajectories to predict
    sim_steps = 30 # Total number of steps in the simulation

    # Agent parameters
    pursuer_max_velo = 12.0
    evader_velo = 12.0
    min_separation_dist = 1.0

    # Evader turning maneuver parameters
    TURN_INTERVAL = 20 #
    TURN_DEGREES = 90 #

    # --- 2. Initialize Agent States ---
    pursuer_current_pos = np.array([-15.0, 0.0]) #
    evader_current_state = Evader(x=-10.0, y=0.0, theta=np.deg2rad(0), v=evader_velo) #

    # --- 3. Log History for Final Visualization ---
    pursuer_history = [pursuer_current_pos] #
    evader_history = [evader_current_state.pos] #
    pursuer_plans_history = [] #
    evader_predictions_history = [] #
    solve_time_history = []

    koz1 = np.array([
        [-4, 1], [-5, 1], [-5, -1], [-4, -1]
    ])

    koz2 = np.array([
        [1, 1], [-1, 1], [-1, -1], [1, -1]
    ])


    print("Starting simulation with JAX + Class-based IPOPT solver...")
    # --- 4. Main Simulation Loop ---
    for i in range(sim_steps): #
        print(f"Running step {i+1}/{sim_steps}...") #

        # --- a. PREDICT ---
        # Generate M possible future trajectories for the evader
        predicted_evader_trajectories = get_straight_away_trajectories( #
            initial_evader_state=evader_current_state, #
            num_trajectories=M, #
            num_time_steps=T, #
            delta_t=dt, #
            speed_variation_std=0.5, #
            heading_variation_std=np.deg2rad(10.0) #
        )
        evader_predictions_history.append(predicted_evader_trajectories) #

        # --- b. PLAN (using the new JAX solver) ---
        # --- MODIFICATION: This entire block is new ---

        # 1. Collate the static problem data into a dictionary
        problem_data = {
            "T": T,
            "dt": dt,
            "max_velo": pursuer_max_velo,
            "start_pos": pursuer_current_pos,
            "evader_trajectories": predicted_evader_trajectories,
            "evader_penalty_weight": 100.0,
            "min_evader_dist": min_separation_dist,
            "keep_out_zones": [koz1,koz2]
        }
    
  
        # 2. Define the bounds for the solver at this time step
        n_vars = 2 * T
        lb = -np.inf * np.ones(n_vars)
        ub = np.inf * np.ones(n_vars)

        n_koz_cons = T * len(problem_data["keep_out_zones"])

    # Constraint bounds
    # 2 (start) + (T-1) (motion) + T (avoidance)
        n_cons = 2 + (T-1) + T + n_koz_cons
      # cl <= g(x) <= cu
        cl = np.concatenate([
            np.zeros(2),             # Start pos constraint == 0
            -np.inf * np.ones(T - 1),# Motion constraint <= 0
            np.zeros(n_koz_cons)
        ])
        cu = np.concatenate([
            np.zeros(2),             # Start pos constraint == 0
            np.zeros(T - 1),         # Motion constraint <= 0
            np.inf * np.ones(n_koz_cons)
        ])

        
        # 3. Create an initial guess for the solver
        avg_end_pos = np.mean(predicted_evader_trajectories[:, -1, :], axis=0)
        x0 = np.linspace(pursuer_current_pos, avg_end_pos, T).flatten()

        # 4. Call the new solver
        start_time = time.perf_counter()
        pursuer_plan, info = tracker.solve(x0, problem_data, lb, ub, cl, cu)
        end_time = time.perf_counter()
        
        solve_time = end_time - start_time
        solve_time_history.append(solve_time)
        # --- End of MODIFICATION ---
        
        if pursuer_plan is None: #
            print("Solver failed to find a plan. Ending simulation.") #
            break #

        pursuer_plans_history.append(pursuer_plan) #

        # --- c. EVADER MANEUVER ---
        if i > 0 and (i % TURN_INTERVAL) == 0 and evader_current_state.x >= 0: #
            print(f"--- EVADER MAKING A {TURN_DEGREES}-DEGREE TURN AT STEP {i} ---") #
            evader_current_state = Evader( #
                x=evader_current_state.x, #
                y=evader_current_state.y, #
                theta=evader_current_state.theta + np.deg2rad(TURN_DEGREES), #
                v=evader_current_state.v #
            )
        
        # --- d. ACT & UPDATE STATE ---
        # Move the pursuer to the second waypoint of its new plan
        pursuer_current_pos = pursuer_plan[1, :] #
        # Move the evader forward according to its kinematic model
        evader_current_state = forward(evader_current_state, dt) #

        # Append the new ground-truth positions to the history
        pursuer_history.append(pursuer_current_pos) #
        evader_history.append(evader_current_state.pos) #

    print("Simulation finished.") #
    # --- 5. Return all collected data for animation ---
    return {
        "pursuer_history": np.array(pursuer_history), #
        "evader_history": np.array(evader_history), #
        "pursuer_plans": pursuer_plans_history, #
        "evader_predictions": evader_predictions_history, #
        "solve_time": solve_time_history,
        "keep_out_zones": [koz1,koz2]
    }


if __name__ == "__main__":
    # Run the simulation to get the data
    simulation_data = run_simulation()

    solve_time = simulation_data.get("solve_time")
    if solve_time is not None and len(solve_time) > 0:
        print("\n--- Solver Performance ---")
        print(f"Average solve time: {np.mean(solve_time):.4f} seconds")
        print(f"Median solve time:  {np.median(solve_time):.4f} seconds")
        print(f"Max solve time:     {np.max(solve_time):.4f} seconds")
        print(f"Min solve time:     {np.min(solve_time):.4f} seconds")
        print(f"Standard deviation: {np.std(solve_time):.4f} seconds")

    # Pass the data to the animator and save to a new file
    # --- MODIFICATION: Changed output filename ---
    animator.create_animation(simulation_data, "fast_pursuit.gif") #