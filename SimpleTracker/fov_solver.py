import casadi as ca
import numpy as np


def get_half_planes_vectorized(vertices):
    """
    Computes the half-plane representation (Ax <= b) for a convex polygon
    defined by vertices.
    """
    if vertices.shape[0] < 3:
        raise ValueError("A polygon must have at least 3 vertices.")
    
    v1_array = vertices
    v2_array = np.roll(vertices, -1, axis=0)
    edge_vectors = v2_array - v1_array
    raw_normals = np.c_[edge_vectors[:, 1], -edge_vectors[:, 0]]
    winding_sum = np.sum(v1_array[:, 0] * v2_array[:, 1] - v2_array[:, 0] * v1_array[:, 1])
    if winding_sum > 0: 
        normals =raw_normals 
    else: 
        normals = -raw_normals 

    norms = np.linalg.norm(normals, axis=1)
    normals = normals / norms[:, np.newaxis]
    
    offsets = np.einsum('ij,ij->i', normals, v1_array)
    
    return normals, offsets

def closest_point_on_segment_casadi(p, a, b):
    """
    Computes the closest point on a line segment [a, b] to a point p.
    All inputs (p, a, b) are CasADi variables or expressions representing 2D points.
    Returns the closest point on the segment.
    """
    ab = b - a
    ap = p - a
    t = ca.dot(ap, ab) / ca.dot(ab, ab)
    t_clamped = ca.fmax(0, ca.fmin(1, t)) 
    closest_pt = a + t_clamped * ab
    return closest_pt

def compute_min_dist_sq_to_polygon_casadi(p, vertices):
    """
    Computes the minimum squared distance from a CasADi point p to a convex
    polygon defined by its numpy vertices.

    Args:
        p: A 2x1 CasADi variable representing the point (e.g., UAV position).
        vertices: A numpy array of shape (num_vertices, 2) defining the
                  convex polygon's vertices in order.

    Returns:
        A CasADi expression for the minimum squared distance.
    """
    num_vertices = vertices.shape[0]
    all_dist_sq = []

    for i in range(num_vertices):
        v1 = vertices[i, :]
        v2 = vertices[(i + 1) % num_vertices, :]
        closest_pt_on_edge = closest_point_on_segment_casadi(p, v1, v2)
        dist_sq = ca.sumsqr(p - closest_pt_on_edge)
        all_dist_sq.append(dist_sq)

    min_dist_sq = ca.mmin(ca.vertcat(*all_dist_sq))
    return min_dist_sq

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
    saftey_radius=2,
    slack_weight=1e6 
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


    # saftey bubble constraint

    num_polygons = len(polygonal_obstacles)
    if num_polygons > 0:
        slack_poly = opti.variable(num_polygons, N + 1)


    safety_radius_sq = saftey_radius**2

    # for k in range(N + 1):
    #         # Iterate through each polygonal obstacle
    #         for poly_verts in polygonal_obstacles:
    #             # Calculate the minimum squared distance from the UAV to the polygon
    #             min_dist_sq_to_poly = compute_min_dist_sq_to_polygon_casadi(pos[:, k], poly_verts)
                
    #             # Add the non-penetration constraint to the optimizer
    #             opti.subject_to(min_dist_sq_to_poly >= safety_radius_sq)
            


    polygon_slack_penalty = 0  # Initialize penalty term for slack
    for k in range(N + 1):
        for i, poly_verts in enumerate(polygonal_obstacles):
            s_ik = slack_poly[i, k]
            opti.subject_to(s_ik >= 0)

            min_dist_sq_to_poly = compute_min_dist_sq_to_polygon_casadi(pos[:, k], poly_verts)
            opti.subject_to(min_dist_sq_to_poly >= safety_radius_sq - s_ik)
            polygon_slack_penalty += s_ik**2

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
        
    objective = tracking_objective + obstacle_weight * obstacle_penalty + fov_weight * fov_penalty + slack_weight * polygon_slack_penalty
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

