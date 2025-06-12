import casadi as ca
import numpy as np
import time
import cv2


def grab_obstacles(uav_state, obstacle_map, avoidance_region, visualize=False, get_ellipse=True):

    map_height, map_width = obstacle_map.shape
    center_x, center_y = int(uav_state[0]), int(uav_state[1])
    half = avoidance_region // 2
    slice_x_start, slice_x_end = max(center_x - half, 0), min(center_x + half, map_width)
    slice_y_start, slice_y_end = max(center_y - half, 0), min(center_y + half, map_height)
    map_slice = obstacle_map[slice_y_start:slice_y_end, slice_x_start:slice_x_end]
    
    binary_image = (map_slice * 255).astype(np.uint8)
    kernel = np.ones((3,3), np.uint8)

    closed_image = cv2.morphologyEx(binary_image, cv2.MORPH_CLOSE, kernel)

    padded_image = cv2.copyMakeBorder(closed_image, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=0)
    contours, _ = cv2.findContours(padded_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    transformed_coords = []
    offset = np.array([slice_x_start, slice_y_start])

    for c in contours:
        if c.shape[0] < 3:  # Need at least 3 points for a polygon
            continue
        c = c.squeeze(1)

        hull = cv2.convexHull(c).squeeze(1)
        c_global = hull + offset
        transformed_coords.append(c_global)

    return transformed_coords


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
    tracking_weight,
    predicted_trajectories,
    probabilities,
    max_velocity,
    max_angular_velocity,
    polygonal_obstacles,
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
    start = time.perf_counter()
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


 # --- MODIFICATION START: Probabilistic Tracking Objective ---
    total_tracking_objective = 0
    total_fov_penalty = 0
    polygon_slack_penalty = 0
    standoff_dist_sq = standoff_distance**2
    epsilon = 1e-5
    a = fov_params['a']
    b = fov_params['b']
    num_polygons = len(polygonal_obstacles)
    safety_radius_sq = saftey_radius**2
    slack_poly = opti.variable(num_polygons, N + 1)

    for k in range(N + 1):
        
        # --- 1. Polygonal Obstacle Logic (for time step 'k') ---
        if num_polygons > 0:
            for i, poly_verts in enumerate(polygonal_obstacles):
                s_ik = slack_poly[i, k]
                opti.subject_to(s_ik >= 0)

                min_dist_sq_to_poly = compute_min_dist_sq_to_polygon_casadi(pos[:, k], poly_verts)
                opti.subject_to(min_dist_sq_to_poly >= safety_radius_sq - s_ik)
                polygon_slack_penalty += s_ik**2
                

        tracking_penalty_k = 0
        fov_penalty_k = 0
        
        for j, p_j in enumerate(probabilities):
            # Get the j-th reference trajectory
            ref_traj_j = predicted_trajectories[j]
            ref_index = min(k, ref_traj_j.shape[1] - 1)
            evader_pos = ref_traj_j[:, ref_index]
            
            # --- Tracking Calculation ---
            dist_sq = ca.sumsqr(pos[:, k] - evader_pos) + epsilon

            tracking_penalty_k += p_j * (dist_sq - standoff_dist_sq)**2
            
            # --- FOV Calculation ---
            vec_world = evader_pos - pos[:, k]
            cos_th = ca.cos(theta[k])
            sin_th = ca.sin(theta[k])
            x_local = vec_world[0] * cos_th + vec_world[1] * sin_th
            y_local = -vec_world[0] * sin_th + vec_world[1] * cos_th
            violation = ((x_local - a) / a)**2 + (y_local / b)**2 - 1
            fov_penalty_k += p_j * ca.fmax(0, violation)

        # Add the accumulated penalties for this time step to the totals
        total_tracking_objective += tracking_penalty_k
        total_fov_penalty += fov_penalty_k


    objective = tracking_weight * total_tracking_objective +  fov_weight * total_fov_penalty + slack_weight * polygon_slack_penalty
    opti.minimize(objective)

    # --- MODIFICATION START: Provide Initial Guess to Solver ---
    if initial_state_guess is not None:
        opti.set_initial(state, initial_state_guess)
    if initial_control_guess is not None:
        opti.set_initial(control, initial_control_guess)

    p_opts = {"expand": True, "print_time": True}
    
    # Solver options: suppress IPOPT startup banner and iteration info
    s_opts = {"ipopt": {"print_level": 0, "sb": "yes"}}
    s_opts = solver_opts
    opti.solver('ipopt', p_opts, s_opts)

    end = time.perf_counter()
    construction_time = end -start

    print(f"Total Construction time:  {(end-start)*1000:.3f} ms")


    
    try:
        sol = opti.solve()
        return sol.value(control), sol.value(state),construction_time,True
    except RuntimeError:
        print("Solver failed at this step! Returning zero control.")
        if initial_state_guess is None:
            fallback_state = np.tile(initial_state.reshape(3, 1), (1, N + 1))
            fallback_control = np.zeros((2, N))
            return fallback_control, fallback_state,construction_time,False
        else:
            return initial_control_guess, initial_state_guess,construction_time,False
        


def track():
    pass