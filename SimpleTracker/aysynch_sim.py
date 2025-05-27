# In file: simulate_asynchronous.py

import numpy as np
from dummy_pwm import Evader, forward, get_straight_away_trajectories
import animator
import track_ipotp as tracker
import time

# Use JAX for 64-bit precision, which is good practice when working with IPOPT
import jax
jax.config.update("jax_enable_x64", True)

def run_asynchronous_simulation():
    """
    Runs a simulation where the evader and pursuer move continuously, even while
    a new plan is being computed.
    """
    # --- 1. Simulation Setup ---
    T = 15
    dt = 0.1
    M = 10
    total_sim_time = 20.0

    # Agent parameters
    pursuer_max_velo = 3.0
    evader_velo = 1.5
    min_separation_dist = 2.0

    # Define the Keep Out Zone
    koz_verts = np.array([[-5, 2], [5, 2], [5, -2], [-5, -2]])
    keep_out_zones = [koz_verts]
    
    # --- 2. Initialize Agents and State ---
    # --- MODIFICATION: Pursuer is now a stateful object, just like the evader ---
    pursuer_state = Evader(x=-15.0, y=0.0, theta=0.0, v=pursuer_max_velo)
    evader_state = Evader(x=0.0, y=0.0, theta=np.deg2rad(0), v=evader_velo)

    # --- 3. Initialize History and Timing ---
    current_time = 0.0
    pursuer_history = [pursuer_state.pos]
    evader_history = [evader_state.pos]
    
    pursuer_plans_history = []
    evader_predictions_history = []
    
    # Initialize with a dummy plan (e.g., stay still at the start)
    pursuer_plan = np.tile(pursuer_state.pos, (T, 1))

    print("Starting Corrected ASYNCHRONOUS simulation...")
    # --- 4. Main Simulation Loop (Time-based) ---
    while current_time < total_sim_time:
        
        print(f"\n--- Starting Planning Cycle at t={current_time:.2f}s ---")
        
        # --- a. PREDICT (based on the current moment) ---
        predicted_evader_trajectories = get_straight_away_trajectories(
            initial_evader_state=evader_state,
            num_trajectories=M, num_time_steps=T, delta_t=dt,
            speed_variation_std=0.5, heading_variation_std=np.deg2rad(10.0)
        )

        # --- b. PLAN (This is a long, blocking call) ---
        problem_data = {
            "T": T, "dt": dt, "max_velo": pursuer_max_velo,
            "start_pos": pursuer_state.pos, # Plan starts from the pursuer's current true position
            "evader_trajectories": predicted_evader_trajectories,
            "min_evader_dist": min_separation_dist, "keep_out_zones": keep_out_zones
        }
        # Define bounds and initial guess for the solver
        n_vars = 2 * T
        lb, ub = -np.inf * np.ones(n_vars), np.inf * np.ones(n_vars)
        n_koz_cons = T * len(keep_out_zones)
        n_cons = 2 + (T - 1) + T + n_koz_cons
        cl = np.concatenate([np.zeros(2), -np.inf * np.ones(T - 1), np.zeros(T), np.zeros(n_koz_cons)])
        cu = np.concatenate([np.zeros(2), np.zeros(T - 1), np.inf * np.ones(T), np.inf * np.ones(n_koz_cons)])
        avg_end_pos = np.mean(predicted_evader_trajectories[:, -1, :], axis=0)
        x0 = np.linspace(pursuer_state.pos, avg_end_pos, T).flatten()

        # Measure how long the solver takes
        planning_start_time = time.perf_counter()
        new_pursuer_plan, info = tracker.solve(x0, problem_data, lb, ub, cl, cu)
        planning_end_time = time.perf_counter()
        solve_time = planning_end_time - planning_start_time
        print(f"Solver finished in {solve_time:.4f} seconds.")

        if new_pursuer_plan is None:
            new_pursuer_plan = pursuer_plan

        # --- c. WORLD CATCH-UP LOOP ---
        num_steps_to_catch_up = int(np.floor(solve_time / dt))
        print(f"World is catching up for {num_steps_to_catch_up} steps...")
        
        for step in range(num_steps_to_catch_up):
            if current_time >= total_sim_time: break
            
            # --- MODIFICATION: Move the pursuer incrementally ---
            # 1. Determine the target waypoint from the OLD active plan
            waypoint_to_follow = min(step + 1, T - 1)
            target_pos = pursuer_plan[waypoint_to_follow, :]
            
            # 2. Calculate heading towards the target
            direction_vector = target_pos - pursuer_state.pos
            pursuer_heading = np.arctan2(direction_vector[1], direction_vector[0])
            
            # 3. Create a state object with the new heading and move forward
            pursuer_state_to_move = Evader(x=pursuer_state.x, y=pursuer_state.y, theta=pursuer_heading, v=pursuer_state.v)
            pursuer_state = forward(pursuer_state_to_move, dt)
            
            # The evader also moves continuously
            evader_state = forward(evader_state, dt)
            
            # Advance the world clock
            current_time += dt
            
            # Log history for each `dt` step
            pursuer_history.append(pursuer_state.pos)
            evader_history.append(evader_state.pos)
            pursuer_plans_history.append(pursuer_plan)
            evader_predictions_history.append(predicted_evader_trajectories)

        # --- d. ADOPT NEW PLAN ---
        pursuer_plan = new_pursuer_plan

    print("Simulation finished.")
    return {
        "pursuer_history": np.array(pursuer_history),
        "evader_history": np.array(evader_history),
        "pursuer_plans": pursuer_plans_history,
        "evader_predictions": evader_predictions_history,
        "keep_out_zones": keep_out_zones
    }


if __name__ == "__main__":
    simulation_data = run_asynchronous_simulation()
    
    print("\nCreating animation...")
    animator.create_animation(simulation_data, "pursuit_asynchronous_corrected.gif")