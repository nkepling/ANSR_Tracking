import jax
import jax.numpy as jnp
import numpy as np
import cyipopt

# Use JAX for 64-bit precision, which IPOPT expects.
jax.config.update("jax_enable_x64", True)

def get_half_planes(vertices):
    """
    Computes the half-plane representation (Ax <= b) for a convex polygon
    defined by vertices in clockwise order.
    """
    A = []
    b = []
    for i in range(len(vertices)):
        v1 = vertices[i]
        v2 = vertices[(i + 1) % len(vertices)] # Wrap around for the last edge
        normal = np.array([v2[1] - v1[1], v1[0] - v2[0]])
        offset = np.dot(normal, v1)
        A.append(normal)
        b.append(offset)
    return np.array(A, dtype=np.float64), np.array(b, dtype=np.float64)

class PursuitProblem:
    """
    Implements the pursuit optimization problem for cyipopt using the class-based API.
    This class calculates the objective, constraints, and their exact first and second
    derivatives (Jacobian and Hessian) using JAX.
    """
    # --- MODIFICATION: Removed koz_penalty_weight from constructor ---
    def __init__(self, T, dt, max_velo, start_pos, evader_trajectories, 
                 min_evader_dist, evader_penalty_weight,
                 obstacles,
                 obstacle_penalty_weight,
                 logsumexp_alpha=0.1):
        # --- Store static problem data ---
        self.T = T
        self.start_pos = jnp.array(start_pos, dtype=jnp.float64)
        self.evader_trajectories = jnp.array(evader_trajectories, dtype=jnp.float64)
        
        # Pre-calculate values for use in JAX functions
        self.avg_evader_path = jnp.mean(self.evader_trajectories, axis=0)
        self.p = jnp.ones(self.evader_trajectories.shape[0]) / self.evader_trajectories.shape[0]
        self.ws = jnp.ones(self.T)
        self.max_dist_sq = (max_velo * dt)**2
        self.min_evader_dist_sq = min_evader_dist**2
        self.alpha = logsumexp_alpha
        self.evader_penalty_weight = evader_penalty_weight
        self.obstacle_penalty_weight = obstacle_penalty_weight 

        # self.kozs = []

        if obstacles:
            obstacle_centers = jnp.array([obs[0] for obs in obstacles], dtype=jnp.float64)
            obstacle_radii = jnp.array([obs[1] for obs in obstacles], dtype=jnp.float64)
        else:
            obstacle_centers = jnp.array([], dtype=jnp.float64).reshape(0, 2) # Empty 0x2 array
            obstacle_radii = jnp.array([], dtype=jnp.float64)
            
        self.obstacle_centers = obstacle_centers
        self.obstacle_radii = obstacle_radii
        self.num_obstacles = len(obstacles)



        # if keep_out_zones:
        #     for vertices in keep_out_zones:
        #         A, b = get_half_planes(vertices)
        #         self.kozs.append({'A': jnp.array(A), 'b': jnp.array(b)})

        # --- Define JAX functions ---
        def _objective_jax(x):
            pursuer_path = x.reshape((self.T, 2))
            
            # --- Base Objective ---
            diffs = pursuer_path[jnp.newaxis, :, :] - self.evader_trajectories
            sq_norms = jnp.sum(diffs**2, axis=-1)
            weighted_sq_norms = self.ws * sq_norms
            base_objective = jnp.sum(self.p * jnp.sum(weighted_sq_norms, axis=-1))
            
            # --- MODIFICATION: Only the evader distance penalty remains ---
            dist_to_evader_sq = jnp.sum((pursuer_path - self.avg_evader_path)**2, axis=1)
            evader_violation = self.min_evader_dist_sq - dist_to_evader_sq
            evader_penalty = jnp.sum(jnp.maximum(0, evader_violation))


            total_obstacle_penalty = 0.0 # Initialize to float
            
            if self.num_obstacles > 0:
                diffs_to_obstacles = pursuer_path[:, jnp.newaxis, :] - self.obstacle_centers[jnp.newaxis, :, :]
                
                dist_sq_to_obstacles = jnp.sum(diffs_to_obstacles**2, axis=-1)
                violation_per_step_per_obs = self.obstacle_radii[jnp.newaxis, :]**2 - dist_sq_to_obstacles
                total_obstacle_penalty = jnp.sum(jnp.maximum(0, violation_per_step_per_obs)**2)

            # --- Final Combined Objective ---
            return base_objective + self.evader_penalty_weight * evader_penalty + self.obstacle_penalty_weight * total_obstacle_penalty

        def _constraints_jax(x):
            pursuer_path = x.reshape((self.T, 2))
            
            # --- Start position and motion are hard constraints ---
            start_pos_violation = pursuer_path[0] - self.start_pos
            motion_vectors = pursuer_path[1:] - pursuer_path[:-1]
            motion_violation = jnp.sum(motion_vectors**2, axis=1) - self.max_dist_sq
            
            # --- MODIFICATION: KOZ avoidance is now a hard constraint again ---
            # koz_violations = []
            # if self.kozs:
            #     for k in range(self.T):
            #         for koz in self.kozs:
            #             z_terms = (koz['A'] @ pursuer_path[k] - koz['b']) / self.alpha
            #             z_max = jnp.max(z_terms)
            #             # Subtract z_max before exp to prevent overflow, add it back (scaled by alpha) after log
            #             logsumexp = self.alpha * (z_max + jnp.log(jnp.sum(jnp.exp(z_terms - z_max)) + 1e-10)) # Add small epsilon for log(0)
            #             koz_violations.append(logsumexp)
            
            return jnp.concatenate([start_pos_violation, motion_violation])

        # --- JIT-compile all necessary functions ---
        self.objective_jit = jax.jit(_objective_jax)
        self.constraints_jit = jax.jit(_constraints_jax)
        self.gradient_jit = jax.jit(jax.grad(_objective_jax))
        self.jacobian_jit = jax.jit(jax.jacfwd(_constraints_jax))
        
        def _lagrangian_jax(x, lagrange, obj_factor):
            return obj_factor * _objective_jax(x) + jnp.dot(lagrange, _constraints_jax(x))

        self.hessian_lag_jit = jax.jit(jax.hessian(_lagrangian_jax))
        
        # --- Pre-calculate Hessian Sparsity Structure ---
        # --- MODIFICATION: Update number of constraints to include KOZ ---
        n_cons = 2 + (self.T - 1)
        
        x_dummy = np.zeros(self.T * 2, dtype=np.float64)
        lagrange_dummy = np.zeros(n_cons, dtype=np.float64)
        obj_factor_dummy = 1.0
        
        hessian_structure = self.hessian_lag_jit(x_dummy, lagrange_dummy, obj_factor_dummy)
        self.hess_rows, self.hess_cols = jnp.nonzero(jnp.tril(hessian_structure))

    # --- Methods that will be called by IPOPT (unchanged) ---
    def objective(self, x): return self.objective_jit(x)
    def gradient(self, x): return self.gradient_jit(x)
    def constraints(self, x): return self.constraints_jit(x)
    def jacobian(self, x): return self.jacobian_jit(x)
    def hessianstructure(self): return (self.hess_rows, self.hess_cols)
    def hessian(self, x, lagrange, obj_factor):
        H = self.hessian_lag_jit(x, lagrange, obj_factor)
        return H[self.hess_rows, self.hess_cols]
    def intermediate(self, *args): pass


def solve(x0, problem_data, lb, ub, cl, cu):
    problem_obj = PursuitProblem(**problem_data)
    
    nlp = cyipopt.Problem(
        n=len(x0), m=len(cl), problem_obj=problem_obj,
        lb=lb, ub=ub, cl=cl, cu=cu
    )
    # --- MODIFICATION: Corrected IPOPT option name ---
    # time_budget_seconds = 0.45
    # nlp.add_option('max_wall_time', time_budget_seconds)
    nlp.add_option('warm_start_init_point', 'yes')
    nlp.add_option('print_level', 5)

    print("--- Starting IPOPT with JAX backend (Hybrid Constraints) ---")
    x, info = nlp.solve(x0)
    print("--- Solver Finished ---")
    # --- MODIFICATION: Return flat array, let caller reshape ---
    return x.reshape(problem_data["T"],2), info

if __name__ == "__main__":
    T = 20
    dt = 0.1
    M = 10
    
    start_pos = np.array([0.0, 0.0])
    initial_evader_state = np.array([10.0, 2.0])
    evader_headings = np.linspace(np.deg2rad(-10), np.deg2rad(10), M)
    evader_trajectories = np.zeros((M, T, 2))
    for i in range(M):
        for k in range(T):
            evader_trajectories[i, k, :] = initial_evader_state + k * dt * 5.0 * np.array([np.cos(evader_headings[i]), np.sin(evader_headings[i])])
    
    koz_verts = np.array([[5, 2], [-5, 2], [-5, -2], [5, -2]])
    
    # --- MODIFICATION: Remove koz_penalty_weight from problem_data ---
    problem_data = {
        "T": T, "dt": dt, "max_velo": 8.0,
        "start_pos": start_pos, "evader_trajectories": evader_trajectories,
        "min_evader_dist": 2.0, "evader_penalty_weight": 500.0,
        "keep_out_zones": [koz_verts]
    }
    
    n_vars = 2 * T
    lb = -np.inf * np.ones(n_vars)
    ub = np.inf * np.ones(n_vars)

    # --- MODIFICATION: Update constraint bounds for the hybrid formulation ---
    n_koz_cons = T * len(problem_data["keep_out_zones"])
    n_cons = 2 + (T - 1) + n_koz_cons
    
    cl = np.concatenate([
        np.zeros(2),                  # Start pos constraint == 0
        -np.inf * np.ones(T - 1),     # Motion constraint <= 0
        np.zeros(n_koz_cons)          # KOZ constraint >= 0
    ])
    cu = np.concatenate([
        np.zeros(2),                  # Start pos constraint == 0
        np.zeros(T - 1),              # Motion constraint <= 0
        np.inf * np.ones(n_koz_cons)  # KOZ constraint >= 0
    ])

    avg_end_pos = np.mean(evader_trajectories[:, -1, :], axis=0)
    x0 = np.linspace(start_pos, avg_end_pos, T).flatten()

    x_sol_flat, info = solve(x0, problem_data, lb, ub, cl, cu )

    print("\n--- Results ---")
    print(f"Status: {info['status_msg']}")
    print(f"Final Objective: {info['obj_val']}")

    # --- MODIFICATION: Reshape solution here for plotting ---
    x_sol = x_sol_flat.reshape(T, 2)
    
    import matplotlib.pyplot as plt
    from matplotlib.patches import Polygon

    fig, ax = plt.subplots(figsize=(10, 10))
    ax.plot(x_sol[:, 0], x_sol[:, 1], 'b-o', label='Pursuer Path')
    ax.plot(evader_trajectories[0, :, 0], evader_trajectories[0, :, 1], 'r--', label='Evader Path (Example)')
    ax.add_patch(Polygon(koz_verts, closed=True, color='red', alpha=0.4, label='Keep-Out Zone'))
    ax.set_title("KOZ Avoidance Trajectory Plan (Hybrid Constraints)")
    ax.legend()
    ax.grid(True)
    ax.set_aspect('equal', adjustable='box')
    plt.show()