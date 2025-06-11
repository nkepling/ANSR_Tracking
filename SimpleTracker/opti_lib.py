import casadi as ca
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import matplotlib.animation as animation

# (Your solve_uav_tracking function remains unchanged here)
def solve_uav_tracking(initial_pos, reference_trajectory, max_speed, obstacles, obstacle_weight, N=20, dt=0.1):
    """
    Solves the UAV trajectory tracking problem for one MPC step.
    """
    opti = ca.Opti()

    x = opti.variable(2, N + 1)
    u = opti.variable(2, N)

    opti.subject_to(x[:, 0] == initial_pos)
    for k in range(N):
        opti.subject_to(x[:, k+1] == x[:, k] + dt * u[:, k])
        opti.subject_to(ca.sumsqr(u[:, k]) <= max_speed**2)

    tracking_objective = 0
    for k in range(N + 1):
        ref_index = min(k, reference_trajectory.shape[1] - 1)
        tracking_objective += ca.sumsqr(x[:, k] - reference_trajectory[:, ref_index])

    obstacle_penalty = 0
    for obs in obstacles:
        obs_center = np.array([obs[0], obs[1]])
        obs_radius_sq = obs[2]**2
        for k in range(N + 1):
            dist_sq = ca.sumsqr(x[:, k] - obs_center)
            violation = obs_radius_sq - dist_sq
            obstacle_penalty += ca.fmax(0, violation)

    objective = tracking_objective + obstacle_weight * obstacle_penalty
    opti.minimize(objective)

    p_opts = {"expand": True}
    s_opts = {"print_level": 0} # Suppress solver output
    opti.solver('ipopt', p_opts, s_opts)

    try:
        sol = opti.solve()
        return sol.value(u[:, 0]), sol.value(x)
    except RuntimeError:
        print("Solver failed at this step!")
        return np.zeros(2), np.tile(initial_pos, (N + 1, 1)).T


# --- NEW: A simple class to manage the evader's state ---
class Evader:
    def __init__(self, x, y, v, theta):
        self.pos = np.array([x, y], dtype=float)
        self.v = v
        self.theta = theta # angle in radians

    def move(self, dt):
        """Move the evader for one time step."""
        # Add some turning logic for a more interesting path
        if self.pos[0] > 4.0:
             self.theta += np.deg2rad(15) * dt * 10
        
        self.pos[0] += self.v * np.cos(self.theta) * dt
        self.pos[1] += self.v * np.sin(self.theta) * dt
        return self.pos.copy()

    def get_predicted_trajectory(self, N, dt):
        """Predict the evader's future path based on its current state."""
        preds = np.zeros((2, N))
        temp_evader = Evader(self.pos[0], self.pos[1], self.v, self.theta)
        for i in range(N):
            preds[:, i] = temp_evader.move(dt)
        return preds

# --- Main Simulation and Animation Setup ---
if __name__ == '__main__':
    # ---- Simulation Parameters ----
    DT = 0.1
    SIM_STEPS = 100
    N_horizon = 20

    # ---- Agent Initialization ----
    uav_pos = np.array([0.0, -1.0])
    evader = Evader(x=0.0, y=0.0, v=1.5, theta=np.deg2rad(20))
    
    uav_max_speed = 2.0
    obstacles = [[3.5, 2.0, 0.5], [6.0, -1.0, 0.7]]
    obstacle_weight = 250.0

    # ---- Data Logging ----
    uav_history = [uav_pos.copy()]
    evader_history = [evader.pos.copy()]
    planned_trajectories = []

    # --- Run the full simulation first to gather all data ---
    print("Running MPC simulation to gather data for animation...")
    for i in range(SIM_STEPS):
        # 1. PLAN: Evader predicts its future path
        evader_prediction = evader.get_predicted_trajectory(N_horizon, DT)
        
        # 2. PLAN: UAV solves for an optimal trajectory to intercept
        optimal_control, planned_trajectory = solve_uav_tracking(
            uav_pos,
            evader_prediction,
            uav_max_speed,
            obstacles,
            obstacle_weight=obstacle_weight,
            N=N_horizon - 1,
            dt=DT
        )
        
        # 3. ACT: Update positions based on the plan and evader's movement
        uav_pos += optimal_control * DT
        evader.move(DT)
        
        # 4. LOG: Store data for animation
        uav_history.append(uav_pos.copy())
        evader_history.append(evader.pos.copy())
        planned_trajectories.append(planned_trajectory)

    print("Simulation complete. Starting animation...")

    # --- Create the Animation ---
    fig, ax = plt.subplots(figsize=(10, 8))
    uav_path = np.array(uav_history)
    evader_path = np.array(evader_history)

    def update(frame):
        ax.clear()

        # Plot obstacles
        for obs in obstacles:
            ax.add_patch(Circle((obs[0], obs[1]), obs[2], color='k', fill=True, alpha=0.5))

        # Plot the full historical paths up to the current frame
        ax.plot(evader_path[:frame+1, 0], evader_path[:frame+1, 1], 'r--', label='Evader Path')
        ax.plot(uav_path[:frame+1, 0], uav_path[:frame+1, 1], 'b-', label='UAV Path')

        # Plot the current positions of the agents
        ax.plot(evader_path[frame, 0], evader_path[frame, 1], 'ro', markersize=10)
        ax.plot(uav_path[frame, 0], uav_path[frame, 1], 'bo', markersize=10)

        # Plot the UAV's planned trajectory at the current frame
        if frame < len(planned_trajectories):
            plan = planned_trajectories[frame]
            ax.plot(plan[0, :], plan[1, :], 'g-+', markersize=5, alpha=0.7, label='UAV Plan')

        # Formatting
        ax.set_title(f"UAV Pursuit Simulation - Time: {frame*DT:.1f}s")
        ax.set_xlabel("X Position (m)")
        ax.set_ylabel("Y Position (m)")
        ax.legend(loc='upper left')
        ax.grid(True)
        ax.axis('equal')
        # Dynamically adjust plot limits
        all_x = np.concatenate([uav_path[:, 0], evader_path[:, 0]])
        all_y = np.concatenate([uav_path[:, 1], evader_path[:, 1]])
        ax.set_xlim(all_x.min() - 1, all_x.max() + 1)
        ax.set_ylim(all_y.min() - 1, all_y.max() + 1)


    # Create and save the animation
    ani = animation.FuncAnimation(fig, update, frames=len(uav_history), repeat=False, interval=1000*DT)
    
    try:
        # To save as a GIF, you may need to install imagemagick or pillow
        ani.save('uav_evader_simulation.gif', writer='pillow', fps=1/DT)
        print("Animation saved to uav_evader_simulation.gif")
    except Exception as e:
        print(f"Could not save animation. Displaying instead. Error: {e}")
        plt.show()