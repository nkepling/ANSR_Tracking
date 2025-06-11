import casadi as ca
import numpy as np

def get_half_planes_vectorized(vertices):
    """
    Computes the half-plane representation (Ax <= b) for a convex polygon
    defined by vertices.

    Args:
        vertices (np.ndarray): A NumPy array of shape (N, 2) where N is the
                               number of vertices. Assumed to be in either
                               clockwise or counter-clockwise order.

    Returns:
        normals (np.ndarray): An array of shape (N, 2) containing the normal vectors.
                              These normals point OUTWARD from the polygon (into the obstacle).
        offsets (np.ndarray): An array of shape (N,) containing the scalar offsets.
    """
    if vertices.shape[0] < 3:
        raise ValueError("A polygon must have at least 3 vertices.")
    
    v1_array = vertices
    v2_array = np.roll(vertices, -1, axis=0)
    edge_vectors = v2_array - v1_array
    raw_normals = np.c_[edge_vectors[:, 1], -edge_vectors[:, 0]]
    winding_sum = np.sum(v1_array[:, 0] * v2_array[:, 1] - v2_array[:, 0] * v1_array[:, 1])
    if winding_sum > 0: # Vertices are in Counter-Clockwise (CCW) order
        # raw_normals already point outward
        normals =raw_normals 
    else: # Vertices are in Clockwise (CW) order
        # raw_normals point inward, so we negate them to point outward
        normals = -raw_normals 

    norms = np.linalg.norm(normals, axis=1)
    normals = normals / norms[:, np.newaxis]
    
    offsets = np.einsum('ij,ij->i', normals, v1_array)
    
    return normals, offsets

def solve_uav_tracking_with_fov(
    initial_state,
    reference_trajectory,
    max_velocity,
    max_angular_velocity,
    obstacles,
    polygonal_obstacles,
    obstacle_weight,
    fov_params,
    fov_weight,
    standoff_distance,
    solver_opts,
    initial_state_guess=None, # <-- NEW: Initial guess for state trajectory
    initial_control_guess=None, # <-- NEW: Initial guess for control trajectory
    N=20,
    dt=0.1,
    collision_safety_margin=0.1
):
    """
    Solves the UAV tracking problem, accepting an initial guess to warm-start the solver.
    """
    opti = ca.Opti()

    # ---- State and Control Variables ----
    state = opti.variable(3, N + 1)
    control = opti.variable(2, N)
    
    # (The rest of the problem definition is unchanged)
    pos = state[:2, :]
    theta = state[2, :]
    v = control[0, :]
    omega = control[1, :]
    opti.subject_to(state[:, 0] == initial_state)
    for k in range(N):
        opti.subject_to(pos[:, k+1] == pos[:, k] + dt * ca.horzcat(v[k] * ca.cos(theta[k]), v[k] * ca.sin(theta[k])).T)
        opti.subject_to(theta[k+1] == theta[k] + dt * omega[k])
        opti.subject_to(opti.bounded(0, v[k], max_velocity))
        opti.subject_to(opti.bounded(-max_angular_velocity, omega[k], max_angular_velocity))


    tracking_objective = 0
    standoff_dist_sq = standoff_distance**2
    epsilon = 1e-5 
    for k in range(N + 1):
        ref_index = min(k, reference_trajectory.shape[1] - 1)
        dist_sq = ca.sumsqr(pos[:, k] - reference_trajectory[:, ref_index]) + epsilon
        # Penalize the error from the ideal standoff distance
        tracking_objective += (dist_sq - standoff_dist_sq)**2


    obstacle_penalty = 0
    for obs in obstacles:
        for k in range(N + 1):
            obstacle_penalty += ca.fmax(0, obs[2]**2 - ca.sumsqr(pos[:, k] - obs[:2]))**2


    # for poly_obs in polygonal_obstacles:
    #         # Loop over each time step in the prediction horizon
    #         for k in range(N + 1):
    #             # For each half-plane (edge) defining the convex polygon
    #             for j in range(poly_obs['normals'].shape[0]):
    #                 normal_vec = ca.DM(poly_obs['normals'][j, :])
    #                 offset_val = poly_obs['offsets'][j]
                    
    #                 # penetration = d - n·p
    #                 # This is positive only when the point p is 'inside' this specific half-plane.
    #                 penetration = offset_val - (normal_vec.T @ pos[:, k])
                    
    #                 # The penalty is now applied only if the penetration is positive.
    #                 # We have removed the `collision_safety_margin` term.
    #                 obstacle_penalty += ca.fmax(0, penetration)**2



    alpha = 1.0 # Smoothness parameter

    for poly_obs in polygonal_obstacles:
        # Loop over each time step in the prediction horizon
        for k in range(1, N + 1):
            
            # 1. Collect all violation terms for this polygon at this time step.
            violations = []
            for j in range(poly_obs['normals'].shape[0]):
                normal_vec = ca.DM(poly_obs['normals'][j, :])
                offset_val = poly_obs['offsets'][j]
                
                # --- THIS IS THE CORRECTED LINE ---
                # Violation (n·p - d) is positive OUTSIDE the half-plane.
                violation_j = (normal_vec.T @ pos[:, k]) - offset_val
                violations.append(violation_j)
            
            # Use vcat to turn the list into a CasADi column vector
            violations_vec = ca.vcat(violations)

            # 2. Compute the numerically stable LogSumExp (Smooth Maximum).
            # This now approximates the "worst" violation, which is positive outside.
            z_terms = violations_vec / alpha
            z_max = ca.mmax(z_terms) 
            logsumexp_val = alpha * (z_max + ca.log(ca.sum1(ca.exp(z_terms - z_max)) + 1e-10))

            # 3. Add to the total penalty ONLY IF the smooth max is positive.
            # This creates a penalty for being outside the polygon (the "Keep Out" behavior).
            obstacle_penalty += ca.fmax(0, logsumexp_val)**2


    fov_penalty = 0
    a = fov_params['a']; b = fov_params['b']
    for k in range(N + 1):
        ref_index = min(k, reference_trajectory.shape[1] - 1)
        evader_pos = reference_trajectory[:, ref_index]
        vec_world = evader_pos - pos[:, k]
        cos_th = ca.cos(theta[k]); sin_th = ca.sin(theta[k])
        x_local = vec_world[0] * cos_th + vec_world[1] * sin_th
        y_local = -vec_world[0] * sin_th + vec_world[1] * cos_th
        violation = ((x_local - a) / a)**2 + (y_local / b)**2 - 1
        fov_penalty += ca.fmax(0, violation)
        
    objective = tracking_objective + obstacle_weight * obstacle_penalty + fov_weight * fov_penalty
    opti.minimize(objective)

    # --- MODIFICATION START: Provide Initial Guess to Solver ---
    if initial_state_guess is not None:
        opti.set_initial(state, initial_state_guess)
    if initial_control_guess is not None:
        opti.set_initial(control, initial_control_guess)
    # --- MODIFICATION END ---

    # ---- Solve ----
    p_opts = {"expand": True}
    s_opts = solver_opts
    opti.solver('ipopt', p_opts, s_opts)

    try:
        sol = opti.solve()
        return sol.value(control), sol.value(state)
    except RuntimeError:
        print("Solver failed at this step! Returning zero control.")
        if initial_state_guess is None:
            fallback_state = np.tile(initial_state.reshape(3, 1), (1, N + 1))
            fallback_control = np.zeros((2, N))
            return fallback_control, fallback_state
        else:
            return initial_control_guess, initial_state_guess


class Evader:
    def __init__(self, x, y, v, theta):
        self.pos = np.array([x, y], dtype=float)
        self.v = v; self.theta = theta
    def move(self, dt):
        if self.pos[0] > 10.0: self.theta += np.deg2rad(10) * dt * 10
        self.pos[0] += self.v * np.cos(self.theta) * dt
        self.pos[1] += self.v * np.sin(self.theta) * dt
        return self.pos.copy()
    def get_predicted_trajectory(self, N, dt):
        preds = np.zeros((2, N))
        temp_evader = Evader(self.pos[0], self.pos[1], self.v, self.theta)
        for i in range(N): preds[:, i] = temp_evader.move(dt)
        return preds

# --- Main Simulation and Animation Setup ---
if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle, Ellipse
    import matplotlib.animation as animation
    import time

    # ---- Simulation Parameters ----
    DT = 0.1; SIM_STEPS = 50; N_horizon = 15

    # ---- Agent and FOV Initialization ----
    uav_state = np.array([0.0, -2.0, np.deg2rad(90)])
    evader = Evader(x=0.0, y=0.0, v=8.0, theta=np.deg2rad(45))
    uav_max_velocity = 12.0
    uav_max_angular_velocity = np.deg2rad(120)
    fov_ellipse_params = {'a': 8.0, 'b': 2.0}
    
    obstacle_weight = 500.0; fov_weight = 500.0; standoff_distance = 10.0
    # obstacles = [[5.0, 4.0, 1.0]]
    obstacles = []

    solver_options = {
        "print_level": 0, "acceptable_tol": 1e-3,
        "acceptable_iter": 5, "max_iter":500,"mu_strategy":"adaptive"
    }

    # ---- Data Logging & Initial Guess Variables ----
    uav_history = [uav_state.copy()]
    evader_history = [evader.pos.copy()]
    planned_trajectories = []
    solve_times = []
    
    # --- MODIFICATION: Initialize variables to store the previous solution ---
    prev_state_sol = None
    prev_control_sol = None

    print("Running MPC simulation with warm-starting...")
    for i in range(SIM_STEPS):
        evader_prediction = evader.get_predicted_trajectory(N_horizon, DT)
        
        # --- MODIFICATION START: Create the initial guess for this step ---
        state_guess = None
        control_guess = None
        if prev_state_sol is not None:
            # "Shift" the previous solution by one time step
            state_guess = np.roll(prev_state_sol, -1, axis=1)
            # The last state can be repeated or extrapolated
            state_guess[:, -1] = state_guess[:, -2]

        if prev_control_sol is not None:
            control_guess = np.roll(prev_control_sol, -1, axis=1)
            control_guess[:, -1] = control_guess[:, -2]
        # --- MODIFICATION END ---
        
        start_time = time.perf_counter()
        
        # The solver now returns the FULL control and state solutions
        optimal_controls, planned_state = solve_uav_tracking_with_fov(
            uav_state, evader_prediction, uav_max_velocity, uav_max_angular_velocity,
            obstacles, obstacle_weight, fov_ellipse_params, fov_weight,standoff_distance,
            solver_opts=solver_options,
            initial_state_guess=state_guess,       # Pass the guess
            initial_control_guess=control_guess,   # Pass the guess
            N=N_horizon, dt=DT
        )
        
        end_time = time.perf_counter()
        solve_times.append(end_time - start_time)

        # --- MODIFICATION: Store the full solution for the next iteration's guess ---
        prev_state_sol = planned_state
        prev_control_sol = optimal_controls
        
        # Use the first control input to move the UAV
        v, omega = optimal_controls[:, 0]
        uav_state[0] += v * np.cos(uav_state[2]) * DT
        uav_state[1] += v * np.sin(uav_state[2]) * DT
        uav_state[2] += omega * DT
        
        evader.move(DT)
        
        uav_history.append(uav_state.copy())
        evader_history.append(evader.pos.copy())
        planned_trajectories.append(planned_state)

    print("Simulation complete.")


    # --- NEW: Report on Solver Performance ---
    if solve_times:
        print("\n--- Solver Performance Statistics ---")
        print(f"Average solve time: {np.mean(solve_times):.3f} s")
        print(f"Median solve time:  {np.median(solve_times):.3f} s")
        print(f"Max solve time:     {np.max(solve_times):.3f} s")
        print(f"Min solve time:     {np.min(solve_times):.3f} s")
        print(f"Std dev:            {np.std(solve_times):.3f} s")
        print("-------------------------------------\n")
    
    # --- Create Animation ---
    print("Creating animation...")
    fig, ax = plt.subplots(figsize=(10, 10))
    uav_path = np.array(uav_history)
    evader_path = np.array(evader_history)

    def update(frame):
        # (Animation update function is unchanged)
        ax.clear()
        for obs in obstacles: ax.add_patch(Circle((obs[0], obs[1]), obs[2], color='k', fill=True, alpha=0.4))
        ax.plot(evader_path[:frame+1, 0], evader_path[:frame+1, 1], 'r--', label='Evader Path')
        ax.plot(uav_path[:frame+1, 0], uav_path[:frame+1, 1], 'b-', label='UAV Path')
        ax.plot(evader_path[frame, 0], evader_path[frame, 1], 'ro', markersize=10)
        uav_current_pos = uav_path[frame, :2]
        uav_current_theta = uav_path[frame, 2]
        ax.plot(uav_current_pos[0], uav_current_pos[1], 'bo', markersize=10)
        
        if frame < len(planned_trajectories):
            plan = planned_trajectories[frame]
            ax.plot(plan[0, :], plan[1, :], 'g-+', alpha=0.6, label='UAV Plan')
        
        a = fov_ellipse_params['a']; b = fov_ellipse_params['b']
        heading_vec = np.array([np.cos(uav_current_theta), np.sin(uav_current_theta)])
        ellipse_center = uav_current_pos + a * heading_vec
        fov_ellipse = Ellipse(
            xy=ellipse_center, width=2 * a, height=2 * b, angle=np.rad2deg(uav_current_theta),
            edgecolor='b', facecolor='blue', alpha=0.15
        )
        ax.add_patch(fov_ellipse)

        ax.set_title(f"UAV Pursuit with Adaptive Solver - Time: {frame*DT:.1f}s")
        ax.set_xlabel("X Position (m)"); ax.set_ylabel("Y Position (m)")
        ax.legend(loc='upper left'); ax.grid(True); ax.axis('equal')
        all_x = np.concatenate([uav_path[:, 0], evader_path[:, 0]])
        all_y = np.concatenate([uav_path[:, 1], evader_path[:, 1]])
        ax.set_xlim(all_x.min() - 2, all_x.max() + 2); ax.set_ylim(all_y.min() - 2, all_y.max() + 2)

    ani = animation.FuncAnimation(fig, update, frames=len(uav_history), repeat=False, interval=int(1000*DT))
    ani.save('uav_adaptive_simulation_3.gif', writer='pillow', fps=int(1/DT))
    print("Animation saved to uav_adaptive_simulation.gif")