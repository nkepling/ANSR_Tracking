# In file: simulate_with_jax.py

import numpy as np
# --- MODIFICATION: Import the new trajectory generation function ---
from dummy_pwm import Evader, forward, get_straight_away_trajectories, generate_goal_directed_trajectories
import animator
import track_ipotp as tracker # Assuming your solver is in track_ipotp.py
from opti_lib import solve_uav_tracking
import jax
jax.config.update("jax_enable_x64", True)
import time


def run_simulation():
    """
    Runs a multi-step pursuit-evasion simulation using the advanced JAX+IPOPT solver.
    """
    # --- 1. Simulation Setup ---
    T = 15    # Planning horizon
    dt = 0.5    # Time step duration
    M = 2       # Number of evader trajectories to predict (e.g., 1 straight, 2 up, 2 down)
    sim_steps = 50

    target_goals =[(6,-11),(6,11)]

    # Agent parameters
    pursuer_max_velo = 8.0
    evader_velo = 8.0       # Matched speed
    min_separation_dist = 1.0

    # Evader ground truth turning maneuver parameters
    EVADER_TURN_X_THRESHOLD = 6 # When the actual evader makes a turn decision
    EVADER_TURN_DEGREES = 90    # Degrees for the actual evader's turn

    # --- 2. Initialize Agent States ---
    pursuer_current_pos = np.array([-18.0, 0.0]) 
    evader_current_state = Evader(x=-15.0, y=0.0, theta=np.deg2rad(0), v=evader_velo) 

    prev_pursuer_plan = None

    # --- 3. Log History for Final Visualization ---
    pursuer_history = [pursuer_current_pos] 
    evader_history = [evader_current_state.pos] 
    pursuer_plans_history = [] 
    evader_predictions_history = [] 
    solve_time_history = []

    # KOZ definitions from your script
    koz1 = np.array([(4,2.5),(-15,2.5),(-15,1.5),(5,1.5),(5,10),(4,10)])
    koz1 = np.array([(5,2.5),(-15,2.5),(-15,1.5),(5,1.5)])
    koz2 = np.array([(4,10),(4,2.5),(5,2.5),(5,10)])
    koz3 = np.array([(5,-2.5),(-15,-2.5),(-15,-1.5),(5,-1.5)])
    koz4 = np.array([(4,-2.5),(4,-10),(5,-10),(5,-2.5)])
    koz5 = np.array([(8,10),(8,-10),(9,-10),(9,10)])


    
    # # koz2 = np.array([(5,)])
    koz_list = [koz1,koz2,koz3,koz4,koz5]

    obstacles = [
        (np.array([0.0, 0.0]), 1.0),
        (np.array([3.0, 0.0]), 1.0)
    ]
    


    print("Starting simulation with JAX + Class-based IPOPT solver...")
    # --- 4. Main Simulation Loop ---
    evader_has_made_ground_truth_turn = False # For the actual evader's behavior
    for i in range(sim_steps): 
        print(f"Running step {i+1}/{sim_steps}...") 

        # --- a. PREDICT ---
        # --- MODIFICATION: Call the new trajectory generation function ---

        # predicted_evader_trajectories = get_straight_away_trajectories(initial_evader_state=evader_current_state,num_trajectories=M,num_time_steps=T,delta_t=dt)

        predicted_evader_trajectories = generate_goal_directed_trajectories(
                initial_evader_state = evader_current_state,
                target_goals =  target_goals,  # List of [x, y] target coordinates
                num_samples_per_goal = M,
                num_time_steps = T,
                delta_t = dt,
                speed_variation_std = 0.1,
                heading_noise_std = np.deg2rad(15.0),
                momentum_factor = 0.2,
                koz_list =koz_list, # List of KOZ vertex arrays
                koz_avoidance_radius  = 3.0, # How close to a KOZ center to trigger avoidance
                koz_steer_strength  = 0.5    # How strongly to steer away from KOZs (0-1)
            )
        evader_predictions_history.append(predicted_evader_trajectories) 

        # --- b. PLAN (using the new JAX solver) ---
        problem_data = {
            "T": T, "dt": dt, "max_velo": pursuer_max_velo,
            "start_pos": pursuer_current_pos,
            "evader_trajectories": predicted_evader_trajectories,
            "evader_penalty_weight": 00.0, # Make sure this matches what track_ipotp.py expects
            "min_evader_dist": min_separation_dist,
            "obstacles": obstacles,
            "obstacle_penalty_weight": 600
            # If track_ipotp.py expects koz_penalty_weight, add it here.
            # "koz_penalty_weight": 2000.0 
        }
    
        n_vars = 2 * T
        lb = -np.inf * np.ones(n_vars)
        ub = np.inf * np.ones(n_vars)

        # Assuming track_ipotp.py uses hard KOZ constraints
        
        # Assuming track_ipotp.py still has evader min distance as penalty, not hard constraint
        n_cons = 2 + (T - 1) # start_pos + motion + KOZ
      
        cl = np.concatenate([
            np.zeros(2),             
            -np.inf * np.ones(T - 1),
        ])
        cu = np.concatenate([
            np.zeros(2),             
            np.zeros(T - 1),         
        ])
        

        if prev_pursuer_plan is None:
            avg_end_pos = np.mean(predicted_evader_trajectories[:, -1, :], axis=0)
            x0 = np.linspace(pursuer_current_pos, avg_end_pos, T,dtype=np.float64).flatten()
        else:
            x0 = prev_pursuer_plan

        start_time = time.perf_counter()
        # pursuer_plan, info = tracker.solve(x0, problem_data, lb, ub, cl, cu)
        pursuer_plan,u = solve_uav_tracking(predicted_evader_trajectories[0], pursuer_max_velo, N=T, dt=dt)
        end_time = time.perf_counter()

        
        solve_time = end_time - start_time
        solve_time_history.append(solve_time)
        
        if pursuer_plan is None: 
            print("Solver failed to find a plan. Ending simulation.") 
            break 

        pursuer_plans_history.append(pursuer_plan) 
        prev_pursuer_plan = pursuer_plan.flatten()

        # --- c. EVADER MANEUVER (Ground Truth Evader) ---
        if not evader_has_made_ground_truth_turn and evader_current_state.x >= EVADER_TURN_X_THRESHOLD: 
            print(f"--- EVADER (Ground Truth) MAKING A TURN AT STEP {i}, x-pos: {evader_current_state.x:.2f} ---") 
            # Randomly pick a turn direction for the actual evader
            turn_direction = np.random.choice([-1, 1]) # -1 for down, 1 for up
            evader_current_state = Evader( 
                x=evader_current_state.x, 
                y=evader_current_state.y, 
                theta=evader_current_state.theta + turn_direction * np.deg2rad(EVADER_TURN_DEGREES), 
                v=evader_current_state.v 
            )
            evader_has_made_ground_truth_turn = True # Turn only once
        
        # --- d. ACT & UPDATE STATE ---
        pursuer_current_pos = pursuer_plan[1, :] 
        evader_current_state = forward(evader_current_state, dt) 

        pursuer_history.append(pursuer_current_pos) 
        evader_history.append(evader_current_state.pos) 

        if min([np.linalg.norm(np.array([evader_current_state.x,evader_current_state.y])-np.array(g)) for g in target_goals]) < 1:
            break



      

    print("Simulation finished.") 
    return {
        "pursuer_history": np.array(pursuer_history), 
        "evader_history": np.array(evader_history), 
        "pursuer_plans": pursuer_plans_history, 
        "evader_predictions": evader_predictions_history, 
        "solve_times": solve_time_history, # Corrected key name
        "keep_out_zones": koz_list,
        "obstacles": obstacles
    }


if __name__ == "__main__":
    simulation_data = run_simulation()

    solve_times_data = simulation_data.get("solve_times") # Corrected key name
    if solve_times_data is not None and len(solve_times_data) > 0:
        print("\n--- Solver Performance ---")
        print(f"Average solve time: {np.mean(solve_times_data):.4f} seconds")
        print(f"Median solve time:  {np.median(solve_times_data):.4f} seconds")
        print(f"Max solve time:     {np.max(solve_times_data):.4f} seconds")
        print(f"Min solve time:     {np.min(solve_times_data):.4f} seconds")
        print(f"Standard deviation: {np.std(solve_times_data):.4f} seconds")

    animator.create_animation(simulation_data, "max_wall_time.gif")