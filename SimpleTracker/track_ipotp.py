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
        # Normal vector pointing inwards for a clockwise polygon
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
    def __init__(self, T, dt, max_velo, start_pos, evader_trajectories, min_evader_dist,keep_out_zones,logsumexp_alpha=0.1):
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

        self.kozs = []
        if keep_out_zones:
            for vertices in keep_out_zones:
                A, b = get_half_planes(vertices)
                self.kozs.append({'A': jnp.array(A), 'b': jnp.array(b)})

        # --- Define JAX functions ---
        # Define pure functions for objective and constraints to be differentiated by JAX.
        def _objective_jax(x):
            pursuer_path = x.reshape((self.T, 2))
            diffs = pursuer_path - self.evader_trajectories # Shape (M, T, 2)
            sq_norms = jnp.sum(diffs**2, axis=-1) # Shape (M, T)
            weighted_sq_norms = self.ws * sq_norms # Shape (M, T)
            trajectory_costs = jnp.sum(weighted_sq_norms, axis=-1) # Shape (M,)
            expected_cost = jnp.sum(self.p * trajectory_costs)
            return expected_cost

        def _constraints_jax(x):
            pursuer_path = x.reshape((self.T, 2))
            
            start_pos_violation = pursuer_path[0] - self.start_pos
            
            motion_vectors = pursuer_path[1:] - pursuer_path[:-1]
            motion_violation = jnp.sum(motion_vectors**2, axis=1) - self.max_dist_sq
            
            dist_to_evader_sq = jnp.sum((pursuer_path - self.avg_evader_path)**2, axis=1)
            evader_avoid_violation = dist_to_evader_sq - self.min_evader_dist_sq

            koz_violations = []
            if self.kozs:
                for k in range(self.T): # For each point in the trajectory
                    for koz in self.kozs: # For each keep-out-zone
                        # Condition for being outside is max(A*x - b) > 0
                        # We use LogSumExp to approximate max smoothly.
                        # We want LogSumExp(z) >= 0, which is our constraint g(x) >= 0.
                        z = (koz['A'] @ pursuer_path[k] - koz['b']) / self.alpha
                        logsumexp = self.alpha * jnp.log(jnp.sum(jnp.exp(z)))
                        koz_violations.append(logsumexp)

            
            return jnp.concatenate([start_pos_violation, motion_violation, evader_avoid_violation,jnp.array(koz_violations)])

        # --- JIT-compile all necessary functions ---
        self.objective_jit = jax.jit(_objective_jax)
        self.constraints_jit = jax.jit(_constraints_jax)
        self.gradient_jit = jax.jit(jax.grad(_objective_jax))
        self.jacobian_jit = jax.jit(jax.jacfwd(_constraints_jax))
        
        # --- Hessian Calculation ---
        # IPOPT requires the Hessian of the Lagrangian: L = obj_factor * f(x) + lagrange^T * g(x)
        def _lagrangian_jax(x, lagrange, obj_factor):
            return obj_factor * _objective_jax(x) + jnp.dot(lagrange, _constraints_jax(x))

        # Create the Hessian function from the Lagrangian
        self.hessian_lag_jit = jax.jit(jax.hessian(_lagrangian_jax))
        
        # --- Pre-calculate Hessian Sparsity Structure ---
        # This is a crucial optimization. We tell IPOPT which entries are non-zero.
        # We can get this structure from the Jacobian of the gradient of the Lagrangian.
        # We use a dummy x, lagrange, and obj_factor to trace the structure.
        n_koz_cons = self.T * len(self.kozs)
        n_cons = 2 + (self.T - 1) + self.T + n_koz_cons
        
        x_dummy = np.zeros(self.T * 2, dtype=np.float64)
        # --- FIX: Use the correct n_cons for the dummy vector ---
        lagrange_dummy = np.zeros(n_cons, dtype=np.float64)
        obj_factor_dummy = 1.0
        
        hessian_structure = self.hessian_lag_jit(x_dummy, lagrange_dummy, obj_factor_dummy)
        self.hess_rows, self.hess_cols = jnp.nonzero(jnp.tril(hessian_structure))

    # --- Methods that will be called by IPOPT ---

    def objective(self, x):
        return self.objective_jit(x)

    def gradient(self, x):
        return self.gradient_jit(x)

    def constraints(self, x):
        return self.constraints_jit(x)

    def jacobian(self, x):
        # IPOPT needs a dense numpy array for the jacobian
        return self.jacobian_jit(x)

    def hessianstructure(self):
        # Return the pre-computed sparsity structure of the Hessian
        return (self.hess_rows, self.hess_cols)

    def hessian(self, x, lagrange, obj_factor):
        # Compute the Hessian of the Lagrangian at the given point
        H = self.hessian_lag_jit(x, lagrange, obj_factor)
        # Return only the non-zero values corresponding to the structure
        return H[self.hess_rows, self.hess_cols]

    def intermediate(self, alg_mod, iter_count, obj_value, inf_pr, inf_du, mu,
                     d_norm, regularization_size, alpha_du, alpha_pr, ls_trials):
        """ Callback function for monitoring optimization progress. """
        # print(f"Iter: {iter_count:3d}  Obj: {obj_value:10.4e}  Inf_pr: {inf_pr:10.4e}  Inf_du: {inf_du:10.4e}")
        pass


def solve(x0, problem_data, lb, ub, cl, cu):
    """
    Instantiates and solves the optimization problem.
    """
    # --- Create the Problem ---
    problem_obj = PursuitProblem(**problem_data)
    
    nlp = cyipopt.Problem(
        n=len(x0),
        m=len(cl),
        problem_obj=problem_obj,
        lb=lb,
        ub=ub,
        cl=cl,
        cu=cu
    )

    time_budget_seconds = 0.45
    nlp.add_option('max_wall_time', time_budget_seconds)

    # Optional: You can also limit the number of iterations
    # nlp.add_option('max_iter', 100)
    
    # Suppress most of the solver output for cleaner logs
    nlp.add_option('print_level', 0)

    # --- Solve the Problem ---
    print("--- Starting IPOPT with JAX backend (Hessian enabled) ---")
    x, info = nlp.solve(x0)
    print("--- Solver Finished ---")
    return x.reshape(problem_data["T"],2), info


if __name__ == "__main__":
    # --- 1. Define Problem Parameters ---
    T = 20  # Time horizon
    dt = 0.1
    M = 10  # Number of evader trajectories
    
    # --- 2. Create Dummy Data ---
    start_pos = np.array([0.0, 0.0])
    # Create M plausible straight-line trajectories for the evader
    initial_evader_state = np.array([10.0, 2.0])
    evader_headings = np.linspace(np.deg2rad(-10), np.deg2rad(10), M)
    evader_trajectories = np.zeros((M, T, 2))
    for i in range(M):
        for k in range(T):
            evader_trajectories[i, k, :] = initial_evader_state + k * dt * 5.0 * np.array([np.cos(evader_headings[i]), np.sin(evader_headings[i])])
    
    koz_verts = np.array([
        [5, 2], [-5, 2], [-5, -2], [5, -2]
    ])
    problem_data = {
        "T": T,
        "dt": dt,
        "max_velo": 8.0,
        "start_pos": start_pos,
        "evader_trajectories": evader_trajectories,
        "min_evader_dist": 2.0,
        "keep_out_zones": [koz_verts]
    }

   
    
    # --- 3. Define Bounds ---
    # Variable bounds (position can be anywhere)
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
        np.zeros(T),
        np.zeros(n_koz_cons)                # Avoidance constraint >= 0
    ])
    cu = np.concatenate([
        np.zeros(2),             # Start pos constraint == 0
        np.zeros(T - 1),         # Motion constraint <= 0
        np.inf * np.ones(T),
        np.inf * np.ones(n_koz_cons)      # Avoidance constraint >= 0
    ])

    # --- 4. Initial Guess ---
    # A straight line from the start towards the average endpoint of evader trajectories
    avg_end_pos = np.mean(evader_trajectories[:, -1, :], axis=0)
    x0 = np.linspace(start_pos, avg_end_pos, T).flatten()

    # --- 5. Solve the problem ---
    x_sol, info = solve(x0, problem_data, lb, ub, cl, cu)

    # --- 6. Print Results ---
    print("\n--- Results ---")
    print(f"Status: {info['status_msg']}")
    print(f"Final Objective: {info['obj_val']}")


    print(x_sol.reshape(T,2))
    # You can now plot or analyze x_sol.reshape(T, 2)

    import matplotlib.pyplot as plt
    from matplotlib.patches import Polygon

    fig, ax = plt.subplots(figsize=(10, 10))
    ax.plot(x_sol[:, 0], x_sol[:, 1], 'b-o', label='Pursuer Path')
    ax.plot(evader_trajectories[0, :, 0], evader_trajectories[0, :, 1], 'r--', label='Evader Path (Example)')
    
    ax.add_patch(Polygon(koz_verts, closed=True, color='red', alpha=0.4, label='Keep-Out Zone'))
    
    ax.set_title("KOZ Avoidance Trajectory Plan")
    ax.legend()
    ax.grid(True)
    ax.set_aspect('equal', adjustable='box')
    plt.show()