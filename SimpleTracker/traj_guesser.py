import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import splprep, splev

def walk_spline_projected_steps(tck, step_distance):
    """
    Generates waypoints by "walking" along a B-spline with fixed-size steps.

    At each point, it takes a step along the tangent and then finds the closest
    point on the spline to that new virtual position.

    Args:
        tck: The tuple (knots, coefficients, degree) returned by splprep.
        step_distance (float): The length of each virtual step (e.g., velo * dt).

    Returns:
        np.ndarray: An array of shape (N, 2) containing the generated waypoints.
    """
    if step_distance <= 0:
        raise ValueError("step_distance must be positive.")

    # --- Initialization ---
    current_u = 0.0
    # Evaluate the spline at u=0 to get the starting point
    start_point = splev(current_u, tck)
    path_points = [start_point]
    
    # Define how far ahead to search for the closest point at each step.
    # This needs to be larger than step_distance to be robust.
    search_u_window = 0.1 

    # --- Iterative Stepping ---
    while current_u < 1.0:
        # 1. Get current position and direction (tangent)
        current_pos = splev(current_u, tck)
        vx, vy = splev(current_u, tck, der=1)
        
        tangent_vector = np.array([vx, vy])
        tangent_norm = np.linalg.norm(tangent_vector)

        if tangent_norm < 1e-6: # Reached end or a stationary point
            break
            
        unit_tangent = tangent_vector / tangent_norm

        # 2. Take a "virtual step" along the tangent
        next_pos_guess = current_pos + unit_tangent * step_distance

        # 3. Find the closest point on the spline to this guess
        # Create a search space of 'u' values ahead of our current 'u'
        u_search_space = np.linspace(current_u, min(1.0, current_u + search_u_window), 500)
        
        # Evaluate the spline over this search space
        spline_search_points = np.array(splev(u_search_space, tck)).T
        
        # Find the index of the point closest to our guess
        distances_sq = np.sum((spline_search_points - next_pos_guess)**2, axis=1)
        closest_index = np.argmin(distances_sq)
        
        # This is our new point on the spline
        next_u = u_search_space[closest_index]
        
        # Check for termination: if we are stuck or at the end, break
        if abs(next_u - current_u) < 1e-7:
            break
            
        current_u = next_u
        path_points.append(splev(current_u, tck))

    return np.array(path_points)


### Example Usage

if __name__ == '__main__':
    # --- 1. Setup: Create a B-spline ---
    coarse_waypoints = np.array([
        [0, 0],
        [10, 5],
        [12, 15],
        [5, 12],
        [8, 2]
    ])
    distance = np.cumsum(np.sqrt(np.sum(np.diff(coarse_waypoints, axis=0)**2, axis=1)))
    distance = np.insert(distance, 0, 0)
    tck, u_param = splprep([coarse_waypoints[:, 0], coarse_waypoints[:, 1]], u=distance, k=3, s=0)

    # --- 2. Use the "walking" function ---
    velocity = 5.0  # m/s
    dt = 0.5       # seconds
    step_dist = velocity * dt # The length of each step is 2.5 meters
    
    walked_points = walk_spline_projected_steps(tck, step_distance=step_dist)

    # --- 3. Plotting for visualization ---
    # Plot the full, smooth B-spline for reference
    u_fine = np.linspace(0, 1, 500)
    x_fine, y_fine = splev(u_fine, tck)

    plt.figure(figsize=(12, 9))
    plt.plot(coarse_waypoints[:, 0], coarse_waypoints[:, 1], 'ro-', markersize=10, linewidth=2, label='Coarse Waypoints')
    plt.plot(x_fine, y_fine, 'g--', linewidth=1.5, label='Full B-Spline Path')
    plt.plot(walked_points[:, 0], walked_points[:, 1], 'bx-', markersize=8, mew=2, label=f'Walked Points (step_dist={step_dist}m)')

    plt.title("Walking Along a B-Spline with Projected Steps")
    plt.xlabel("X Position (m)")
    plt.ylabel("Y Position (m)")
    plt.legend()
    plt.grid(True)
    plt.axis('equal')
    plt.show()