import casadi as ca
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Ellipse
import matplotlib.animation as animation
import time

# The solver function `solve_uav_tracking_with_los` remains unchanged.
# (I've omitted it here for brevity, but it's the same as the previous response)
def solve_uav_tracking_with_los(
    initial_state, reference_trajectory, max_velocity, max_angular_velocity,
    obstacles, obstacle_weight, fov_params, fov_weight, los_weight,
    solver_opts, N=20, dt=0.1 ):
    opti = ca.Opti()
    state = opti.variable(3, N + 1); pos = state[:2, :]; theta = state[2, :]
    control = opti.variable(2, N); v = control[0, :]; omega = control[1, :]
    opti.subject_to(state[:, 0] == initial_state)
    for k in range(N):
        opti.subject_to(pos[:, k+1] == pos[:, k] + dt * ca.horzcat(v[k] * ca.cos(theta[k]), v[k] * ca.sin(theta[k])).T)
        opti.subject_to(theta[k+1] == theta[k] + dt * omega[k])
        opti.subject_to(opti.bounded(0, v[k], max_velocity))
        opti.subject_to(opti.bounded(-max_angular_velocity, omega[k], max_angular_velocity))
    tracking_objective = 0; obstacle_penalty = 0
    for k in range(N + 1):
        ref_index = min(k, reference_trajectory.shape[1] - 1)
        tracking_objective += ca.sumsqr(pos[:, k] - reference_trajectory[:, ref_index])
        for obs in obstacles:
            obstacle_penalty += ca.fmax(0, obs[2]**2 - ca.sumsqr(pos[:, k] - obs[:2]))**2
    fov_penalty = 0; a = fov_params['a']; b = fov_params['b']
    for k in range(N + 1):
        ref_index = min(k, reference_trajectory.shape[1] - 1)
        evader_pos = reference_trajectory[:, ref_index]
        vec_world = evader_pos - pos[:, k]
        cos_th = ca.cos(theta[k]); sin_th = ca.sin(theta[k])
        x_local = vec_world[0] * cos_th + vec_world[1] * sin_th
        y_local = -vec_world[0] * sin_th + vec_world[1] * cos_th
        violation = ((x_local - a) / a)**2 + (y_local / b)**2 - 1
        fov_penalty += ca.fmax(0, violation)
    los_penalty = 0
    for k in range(N + 1):
        pursuer_pos = pos[:, k]
        ref_index = min(k, reference_trajectory.shape[1] - 1)
        evader_pos = reference_trajectory[:, ref_index]
        line_vec = evader_pos - pursuer_pos
        line_len_sq = ca.sumsqr(line_vec) + 1e-9
        for obs in obstacles:
            obs_pos = obs[:2]; obs_rad_sq = obs[2]**2
            vec_to_obs = obs_pos - pursuer_pos
            t = ca.dot(vec_to_obs, line_vec) / line_len_sq
            t_clamped = ca.fmax(0, ca.fmin(1, t))
            closest_point_on_segment = pursuer_pos + t_clamped * line_vec
            dist_sq_to_line = ca.sumsqr(obs_pos - closest_point_on_segment)
            violation = obs_rad_sq - dist_sq_to_line
            los_penalty += ca.fmax(0, violation)**2
    objective = (tracking_objective + obstacle_weight * obstacle_penalty +
                 fov_weight * fov_penalty + los_weight * los_penalty)
    opti.minimize(objective)
    p_opts = {"expand": True}; s_opts = solver_opts
    opti.solver('ipopt', p_opts, s_opts)
    try:
        sol = opti.solve()
        return sol.value(control[:, 0]), sol.value(state)
    except RuntimeError:
        return np.zeros(2), np.tile(initial_state, (N + 1, 1)).T


class Evader:
    def __init__(self, x, y, v, theta):
        self.pos = np.array([x, y], dtype=float); self.v = v; self.theta = theta
    def move(self, dt):
        self.pos[0] += self.v * np.cos(self.theta) * dt
        self.pos[1] += self.v * np.sin(self.theta) * dt
        return self.pos.copy()
    def get_predicted_trajectory(self, N, dt):
        preds = np.zeros((2, N))
        temp_evader = Evader(self.pos[0], self.pos[1], self.v, self.theta)
        for i in range(N): preds[:, i] = temp_evader.move(dt)
        return preds


if __name__ == '__main__':
    # ---- Simulation Parameters ----
    DT = 0.1; SIM_STEPS = 150; N_horizon = 20
    CAPTURE_RADIUS = 0.5 # <-- NEW: Success condition

    # ---- Agent and FOV Initialization ----
    uav_state = np.array([0.0, -2.0, np.deg2rad(90)])
    evader = Evader(x=0.0, y=0.0, v=10.0, theta=np.deg2rad(45))
    uav_max_velocity = 10.0
    uav_max_angular_velocity = np.deg2rad(120)
    fov_ellipse_params = {'a': 8.0, 'b': 4.0}
    
    # --- Tuning Weights ---

    obstacles = [[5.0, 4.0, 1.0],[10,10,2]]

    # --- Tuning Weights ---
    obstacle_weight = 500.0
    fov_weight = 50.0 # Increased weight to prioritize keeping view
    los_weight = 1000.0

    solver_options = {"print_level": 0, "acceptable_tol": 1e-3, "max_cpu_time": 0.1}

    # ---- Data Logging ----
    uav_history = [uav_state.copy()]; evader_history = [evader.pos.copy()]
    planned_trajectories = []

    # --- Run Simulation to Gather Data ---
    print("Running MPC simulation with hard FOV and LOS rules...")
    for i in range(SIM_STEPS):
        evader_prediction = evader.get_predicted_trajectory(N_horizon, DT)
        optimal_control, planned_state = solve_uav_tracking_with_los(
            uav_state, evader_prediction, uav_max_velocity, uav_max_angular_velocity,
            obstacles, obstacle_weight, fov_ellipse_params, fov_weight,
            los_weight, solver_options, N=N_horizon, dt=DT
        )
        
        # --- ACT ---
        v, omega = optimal_control
        uav_state[0] += v * np.cos(uav_state[2]) * DT
        uav_state[1] += v * np.sin(uav_state[2]) * DT
        uav_state[2] += omega * DT
        evader.move(DT)
        
        # --- LOG ---
        uav_history.append(uav_state.copy()); evader_history.append(evader.pos.copy())
        planned_trajectories.append(planned_state)
        
        # --- MODIFICATION START: Check Simulation End Conditions ---
        
        # 1. Check for SUCCESS (Capture)
        pursuer_pos = uav_state[:2]
        evader_pos = evader.pos
        distance = np.linalg.norm(pursuer_pos - evader_pos)
        if distance < CAPTURE_RADIUS:
            print(f"\n--- SUCCESS: Evader captured at step {i+1}! ---")
            break

        # 2. Check for FAILURE (Loss of Sight)
        # Use the same logic as the solver to check if evader is in the ellipse
        a, b = fov_ellipse_params['a'], fov_ellipse_params['b']
        vec_world = evader_pos - pursuer_pos
        cos_th = np.cos(uav_state[2]); sin_th = np.sin(uav_state[2])
        x_local = vec_world[0] * cos_th + vec_world[1] * sin_th
        y_local = -vec_world[0] * sin_th + vec_world[1] * cos_th
        
        # The evader is "out of view" if this value is > 1
        ellipse_check_value = ((x_local - a) / a)**2 + (y_local / b)**2
        
        if ellipse_check_value > 1:
            print(f"\n--- FAILURE: Evader lost from field of view at step {i+1}! ---")
            break
            
        # --- MODIFICATION END ---

    if i == SIM_STEPS - 1:
        print("\n--- SIMULATION ENDED: Time limit reached. ---")


    # --- Animation (Code is unchanged, it will now correctly show the shortened simulation) ---
    print("Creating animation...")
    fig, ax = plt.subplots(figsize=(10, 10))
    uav_path = np.array(uav_history); evader_path = np.array(evader_history)

    # (check_los_for_plot function is the same, omitted for brevity)
    def check_los_for_plot(p_p, p_e, obs_list):
        line_vec = p_e - p_p; line_len_sq = np.sum(line_vec**2) + 1e-9
        for obs in obs_list:
            obs_pos = np.array(obs[:2]); obs_rad_sq = obs[2]**2
            vec_to_obs = obs_pos - p_p; t = np.dot(vec_to_obs, line_vec) / line_len_sq
            t_clamped = np.clip(t, 0, 1); closest_point = p_p + t_clamped * line_vec
            dist_sq = np.sum((obs_pos - closest_point)**2)
            if dist_sq < obs_rad_sq: return False
        return True

    def update(frame):
        ax.clear()
        ax.plot(evader_path[:, 0], evader_path[:, 1], 'r--', alpha=0.5)
        ax.plot(uav_path[:, 0], uav_path[:, 1], 'b-', alpha=0.5)
        uav_current_pos = uav_path[frame, :2]; uav_current_theta = uav_path[frame, 2]
        evader_current_pos = evader_path[frame, :]
        is_los_clear = check_los_for_plot(uav_current_pos, evader_current_pos, obstacles)
        los_color = 'lime' if is_los_clear else 'tomato'; los_style = '-' if is_los_clear else ':'
        ax.plot([uav_current_pos[0], evader_current_pos[0]], [uav_current_pos[1], evader_current_pos[1]],
                color=los_color, linestyle=los_style, lw=2, label="Line of Sight")
        ax.plot(evader_current_pos[0], evader_current_pos[1], 'ro', markersize=10, label="Evader")
        ax.plot(uav_current_pos[0], uav_current_pos[1], 'bo', markersize=10, label="UAV")
        if frame < len(planned_trajectories):
            plan = planned_trajectories[frame]
            ax.plot(plan[0, :], plan[1, :], 'g-+', alpha=0.6, label='UAV Plan')
        a, b = fov_ellipse_params['a'], fov_ellipse_params['b']
        heading_vec = np.array([np.cos(uav_current_theta), np.sin(uav_current_theta)])
        ellipse_center = uav_current_pos + a * heading_vec
        fov_ellipse = Ellipse(xy=ellipse_center, width=2*a, height=2*b, angle=np.rad2deg(uav_current_theta), edgecolor='b', facecolor='blue', alpha=0.1)
        ax.add_patch(fov_ellipse)
        for obs in obstacles: ax.add_patch(Circle((obs[0], obs[1]), obs[2], color='k', fill=True, alpha=0.7))
        ax.set_title(f"MPC with Hard FOV Rule - Time: {frame*DT:.1f}s")
        ax.legend(loc='upper left'); ax.grid(True); ax.axis('equal');
        ax.set_xlim(-8, 8); ax.set_ylim(-8, 8)

    ani = animation.FuncAnimation(fig, update, frames=len(uav_history), repeat=False, interval=int(1000*DT))
    ani.save('uav_hard_fov_simulation.gif', writer='pillow', fps=int(1/DT))
    print("Animation saved to uav_hard_fov_simulation.gif")