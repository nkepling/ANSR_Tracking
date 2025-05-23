import numpy as np
from scipy.optimize import minimize,Bounds

def get_half_plane_representation(verts):
    """
    Computes the half-plane representation (Ax <= b) for a convex polygon
    defined by clockwise vertices.
    """
    A = []
    b = []
    for i in range(len(verts) - 1):
        v1 = verts[i]
        v2 = verts[i+1]
        # Normal vector pointing inwards for a clockwise polygon
        normal = np.array([v2[1] - v1[1], v1[0] - v2[0]])
        offset = np.dot(normal, v1)
        A.append(normal)
        b.append(offset)
    return np.array(A), np.array(b)

def get_constraints(T, start_pos, max_velo, dt,evader_path,min_dist):
    """
    Creates the constraint dictionaries and bounds for the SLSQP solver.
    
    Args:
        T: Number of time steps.
        start_pos: The (x, y) fixed starting location.
        swx_poly_verts: Vertices of the "Safe Walk Zone" convex polygon.
        max_velo: Maximum velocity of the pursuer.
        dt: Time step duration.

    Returns:
        A tuple of (constraints_list, bounds_object).
    """
    # --- Constraint 1: Fixed Starting Location ---
    def start_pos_constraint(x_flat):
        return x_flat[0:2] - start_pos

    # --- Constraint 2: Motion Constraints (Max Velocity) ---
    def motion_constraint(x_flat):
        path = x_flat.reshape(T, 2)
        vectors = path[1:] - path[:-1]
        squared_distances = np.sum(vectors**2, axis=1)
        # Constraint is g(x) >= 0, so (v_max*dt)^2 - dist^2 >= 0
        return (max_velo * dt)**2 - squared_distances

    # --- Constraint 3: Stay Within Safe Walk Zone (SWX) ---
    # This is more robust for SLSQP than a Keep-Out Zone. We ensure the path
    # stays INSIDE a convex polygon.
    # A, b = get_half_plane_representation(swx_poly_verts)
    # def swx_constraint(x_flat):
    #     path = x_flat.reshape(T, 2)
    #     # For each point on the path, calculate Ax - b.
    #     # The constraint is satisfied if Ax - b <= 0, so we return -(Ax - b)
    #     # to fit the g(x) >= 0 requirement.
    #     return b - path @ A.T

    def evader_avoidance_constraint(x_flat):
        pursuer_path = x_flat.reshape(T, 2)
        # Calculate squared distance at each time step
        squared_dist = np.sum((pursuer_path - evader_path)**2, axis=1)
        # Constraint is satisfied if dist^2 - min_dist^2 >= 0
        return squared_dist - min_dist**2

 
    constraints = [
        {'type': 'eq', 'fun': start_pos_constraint},
        {'type': 'ineq', 'fun': motion_constraint},
    ]

    # constraints.append({'type': 'ineq', 'fun': evader_avoidance_constraint})

    # --- Create Bounds Object ---
    # A simple box boundary as a fallback. The SWX is more precise.
    bounds = Bounds([-np.inf] * (2 * T), [np.inf] * (2 * T))

    return constraints, bounds

def solve(evader_trajectories: np.ndarray, T: int, dt: float, start_pos: np.ndarray, max_velo: float, p=None, w=None):
    """
    Sets up and solves the optimization problem to find the optimal pursuer path.
    """
    M = evader_trajectories.shape[0]

    MIN_EVADER_DISTANCE = 2.5**2
    EVADER_PENALTY_WEIGHT = 500
    avg_evader_path = np.mean(evader_trajectories, axis=0)
   

    # --- Define Problem Weights and Parameters ---
    # Time-decaying weights: prioritize matching earlier parts of trajectories

    if w is None:
        ws = np.ones(T)
    else:
        ws = np.array([w**k for k in range(1,T+1)])

    # Probabilities for each evader trajectory (here, uniform)
    if p is None:
        p = np.ones(M) / M

    # --- 1. Get Constraints and Bounds ---
    constraints, bounds = get_constraints(T, start_pos, max_velo, dt,avg_evader_path,MIN_EVADER_DISTANCE)

    # --- 2. Create the Objective Function for the Solver ---
    # The solver's objective function can only take the decision variable (x)
    # as its first argument. We use a lambda to pass in the other fixed data.
    

    # --- 3. Create an Initial Guess (x0) ---
    # A straight line from the start towards the average endpoint of evader trajectories
    avg_end_pos = np.mean(evader_trajectories[:, -1, :], axis=0)
    x0 = np.linspace(start_pos, avg_end_pos, T).flatten()


    def objective_with_penalty(x_flat):
        # Reshape pursuer path
        pursuer_path = x_flat.reshape(T, 2)
        X_reshaped = pursuer_path.reshape(1, T, 2)

        # 1. Calculate the original objective (get close to predictions)
        obj_original = objective(X_reshaped, evader_trajectories, ws, p)

        # 2. Calculate the penalty for being too close to the evader
        # We penalize based on the distance to the average predicted path
        avg_evader_path = np.mean(evader_trajectories, axis=0)
        squared_dist_to_evader = np.sum((pursuer_path - avg_evader_path)**2, axis=1)
        
        # Hinge loss: penalty is zero if constraint is met (dist^2 > min_dist^2)
        violation = MIN_EVADER_DISTANCE - squared_dist_to_evader
        penalty = np.sum(np.maximum(0, violation)) # Use maximum to ignore non-violations

        # 3. Combine them
        return obj_original + EVADER_PENALTY_WEIGHT * penalty


    # fun_for_solver = lambda x_flat: objective(x_flat.reshape(1, T, 2), evader_trajectories, ws, p)
    fun_for_solver = objective_with_penalty
    # --- 4. Run the Optimizer ---
    # print("Running SLSQP optimizer...")
    res = minimize(
        fun_for_solver,
        x0,
        method='SLSQP',
        bounds=bounds,
        constraints=constraints,
        options={'disp':False, 'maxiter': 500}
    )

    if res.success:
        # print("Optimization successful!")
        # Reshape the flat result back into a path
        return res.x.reshape(T, 2)
    else:
        print("Optimization failed:", res.message)
        print("Returning Initial Guess")
        return x0.reshape(T,2)

def objective(X,Y,w,p ):
    """Compute objective

    X: (1,T,2) numpy tensor
    Y: (M,T,2) numpy tensor
    w: (T,) numpy array 
    p: (M,) numpy array
    """

    return np.sum(p * (np.sum(w * (np.linalg.norm(X - Y, axis=-1)**2), axis=-1))) 



if __name__ == "__main__":
    from dummy_pwm import *
    from vis import visualize_plan

    T = 10
    dt = 0.1
    M = 5  # Number of predicted trajectories
    w = 1

    X = np.random.random((1,T,2))
    pursuer_start_pos = np.array([-10.0,0.0])
    initial_state = Evader(x=2.0, y=0.0, theta=np.deg2rad(0), v=10.0)
    
    # --- Generate Predictions ---
    Y = get_straight_away_trajectories(
        initial_evader_state=initial_state,
        num_trajectories=M,
        num_time_steps=T,
        delta_t=dt,
        speed_variation_std=2.0,       # Standard deviation for speed
        heading_variation_std=np.deg2rad(5.0) # Standard deviation for heading
    )

    # --- Generate a single Ground Truth trajectory for comparison (no noise) ---
    gt_trajectory = []
    current_gt_state = initial_state
    for _ in range(T):
        gt_trajectory.append(current_gt_state.pos)
        current_gt_state = forward(current_gt_state, dt)
    ground_truth_np = np.array(gt_trajectory)

    # koz polygon verts going clockwise
    koz_verts = np.array([(1,1),(2,1),(2,-1),(1,-1),(1,1)])

    ws = np.array([w**k for k in range(1,T+1)])
    p = np.ones((M))


    pursuer_plan = solve(Y, T, dt, pursuer_start_pos,max_velo=10)

    visualize_plan(pursuer_plan,ground_truth_np,Y)





    



