import casadi as ca
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle # For plotting circles

def solve(x0, problem_data, lb, ub, cl, cu):
    
    x_flat, total_objective, g = problem_setup(**problem_data)
    nlp_prob = {
    'f': total_objective,
    'x': x_flat,
    'g': g
}

    opts = {
        'ipopt': {
            'print_level': 0,
        },
        'print_time': 0
    }

    solver = ca.nlpsol('solver', 'ipopt', nlp_prob, opts)
     # --- Solve the problem ---
    print("--- Starting IPOPT with CasADi backend ---")
    sol = solver(x0=x0, lbx=lb, ubx=ub, lbg=cl, ubg=cu) # Pass all bounds to the solver
    print("--- Solver Finished ---")

    # --- Extract results ---
    x_sol_flat = sol['x'].full().flatten()
    obj_val = sol['f'].full().item()
    status_msg = solver.stats()['return_status']

    return x_sol_flat, {'obj_val': obj_val, 'status_msg': status_msg}





def problem_setup(T, dt, max_velo, start_pos, evader_trajectories,
                          min_evader_dist, evader_penalty_weight,
                          obstacles, obstacle_penalty_weight, x0=None):
    #declare variables
    x = ca.MX.sym('x',T, 2)  # Decision variable: pursuer path
    x_flat = ca.vec(x)
    avg_evader_path = ca.MX(np.mean(evader_trajectories, axis=0))
    p = ca.MX(np.ones(evader_trajectories.shape[0]) / evader_trajectories.shape[0])
    ws = ca.MX(np.ones(T))
    max_dist_sq = (max_velo * dt)**2
    min_evader_dist_sq = min_evader_dist**2


    #base objective: Min distance between evader and puruser paths

    base_objective_terms = []
    for i in range(evader_trajectories.shape[0]): # Iterate through each evader
        evader_path_i = ca.MX(evader_trajectories[i, :, :]) # Convert *each 2D slice* to MX
        
        diff_i = x - evader_path_i # (T, 2) - (T, 2) -> (T, 2)
        sq_norms_i = ca.sum2(diff_i**2) # (T, 1) representing squared distance at each time step
        weighted_sq_norms_i = ca.MX(ws) * sq_norms_i 
        base_objective_terms.append(ca.MX(p[i]) * ca.sum1(weighted_sq_norms_i))

    base_objective = ca.sum1(ca.vertcat(*base_objective_terms))
    dist_to_evader_sq = ca.sum2((x - avg_evader_path)**2) # (T, 1)
    evader_violation = min_evader_dist_sq - dist_to_evader_sq
    evader_penalty = ca.sum1(ca.fmax(0, evader_violation)**2)


    # process obstacles
    obstacle_penalty_terms = []
    # FIX: Initialize total_obstacle_penalty to 0 here
    total_obstacle_penalty = ca.MX(0) 

    if obstacles: # Check if the list is not empty
        for obs in obstacles:
            obs_center_np = np.array(obs[0]) 
            obs_radius = obs[1]
            repeated_obs_center = ca.MX(np.tile(obs_center_np, (T, 1))) 
            diff_to_obs = x - repeated_obs_center 
            
            # FIX: Change sumsqr to sum2( ... **2)
            dist_sq_to_obs = ca.sum2(diff_to_obs**2) 
            
            violation_per_timestep = obs_radius**2 - dist_sq_to_obs
            obstacle_penalty_terms.append(ca.sum1(ca.fmax(0, violation_per_timestep)**2))

        # This line will only be executed if obstacles is not empty.
        # If it is empty, total_obstacle_penalty remains ca.MX(0) from initialization.
        total_obstacle_penalty = ca.sum1(ca.vertcat(*obstacle_penalty_terms))


    # This line has the comments REMOVED to show the full objective
    total_objective = base_objective + \
                      evader_penalty_weight * evader_penalty + \
                      obstacle_penalty_weight * total_obstacle_penalty 

    # --- Constraints ---
    constraints = []
    
    # Start position constraint
    start_pos_violation = x[0, :].T - ca.MX(start_pos) # (2,1)
    constraints.append(start_pos_violation)

    # Motion constraint
    motion_vectors = x[1:, :] - x[:-1, :] # (T-1, 2)
    motion_violation = ca.sum2(motion_vectors**2) - max_dist_sq # (T-1, 1)
    constraints.append(motion_violation)

    g = ca.vertcat(*constraints) # This should now work

    return x_flat, total_objective, g

    
if __name__ == "__main__":
    T = 100 
    dt = 0.1
    M = 10 # Number of evader trajectories
    
    # Initial pursuer state (current position)
    start_pos = np.array([0.0, 0.0])

    # Generate M evader trajectories (these are fixed for this solve)
    initial_evader_state = np.array([10.0, 2.0])
    evader_headings = np.linspace(np.deg2rad(-10), np.deg2rad(10), M)
    evader_trajectories = np.zeros((M, T, 2))
    for i in range(M):
        for k in range(T):
            evader_trajectories[i, k, :] = initial_evader_state + k * dt * 5.0 * np.array([np.cos(evader_headings[i]), np.sin(evader_headings[i])])
    
    # Define obstacles (list of ([center_x, center_y], radius))
    obstacles = [
        (np.array([-3.0, 0.0]), 1.5),
        (np.array([3.0, 0.0]), 1.0)
    ]
    
    # --- Assemble problem_data for the 'solve' function ---
    # This dictionary holds ALL numerical values required for this specific solve
    problem_data = {
        "T": T,
        "dt": dt,
        "max_velo": 2.0,
        "start_pos": start_pos, 
        "evader_trajectories": evader_trajectories,
        "min_evader_dist": 2.0,
        "evader_penalty_weight": 500.0,
        "obstacles": obstacles,
        "obstacle_penalty_weight": 100.0,
    }
    
    # --- Define bounds for the decision variables (pursuer path) ---
    n_vars = 2 * T # 2 (x,y) coordinates for each of T time steps
    lb = -np.inf * np.ones(n_vars) # No lower bound on x or y
    ub = np.inf * np.ones(n_vars)  # No upper bound on x or y

    # --- Define bounds for the hard constraints ---
    # Total number of constraints:
    # 2 (for start_pos constraint) + (T - 1) (for motion constraints)
    n_cons = 2 + (T - 1) 
    
    cl = np.concatenate([
        np.zeros(2),                  # Start pos constraint (equality): lower bound = 0
        -np.inf * np.ones(T - 1),     # Motion constraint (inequality <= 0): lower bound = -inf
    ])
    cu = np.concatenate([
        np.zeros(2),                  # Start pos constraint (equality): upper bound = 0
        np.zeros(T - 1),              # Motion constraint (inequality <= 0): upper bound = 0
    ])

    # --- Initial guess for the pursuer path ---
    # A simple straight line towards the average evader end position
    avg_end_pos = np.mean(evader_trajectories[:, -1, :], axis=0)
    x0 = np.linspace(start_pos, avg_end_pos, T).flatten()

    # --- Call the 'solve' function ---
    x_sol_flat, info = solve(x0, problem_data, lb, ub, cl, cu)

    print("\n--- Results ---")
    print(f"Status: {info['status_msg']}")
    print(f"Final Objective: {info['obj_val']}")

    # --- Plotting the results ---
    x_sol = x_sol_flat.reshape(T, 2)
    
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.plot(x_sol[:, 0], x_sol[:, 1], 'b-o', label='Pursuer Path')
    
    # Plot evader trajectories
    for i in range(M):
        ax.plot(evader_trajectories[i, :, 0], evader_trajectories[i, :, 1], 'r--', alpha=0.5, label='Evader Path' if i == 0 else "")
    
    # Plot obstacles
    for obs_center, obs_radius in obstacles:
        circle = Circle(obs_center, obs_radius, color='grey', alpha=0.6, label='Obstacle')
        ax.add_patch(circle)

    ax.set_title("Pursuit Problem (Non-Parameterized CasADi)")
    ax.legend()
    ax.grid(True)
    ax.set_aspect('equal', adjustable='box')
    plt.xlabel('X position')
    plt.ylabel('Y position')
    plt.show()