import numpy as np
from fov_solver import get_half_planes_vectorized
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
# def get_half_planes_vectorized(vertices):
#     """
#     Computes the half-plane representation (Ax <= b) for a convex polygon
#     defined by vertices. This version produces OUTWARD-pointing normals.
#     """
#     if vertices.shape[0] < 3:
#         raise ValueError("A polygon must have at least 3 vertices.")
    
#     v1_array = vertices
#     v2_array = np.roll(vertices, -1, axis=0)
#     edge_vectors = v2_array - v1_array
#     raw_normals = np.c_[edge_vectors[:, 1], -edge_vectors[:, 0]]
    
#     # Calculate the winding order using the shoelace formula
#     winding_sum = np.sum(v1_array[:, 0] * v2_array[:, 1] - v2_array[:, 0] * v1_array[:, 1])

#     # For CCW winding, raw_normals point outward.
#     # For CW winding, they point inward, so they must be flipped.
#     if winding_sum > 0: # Vertices are in Counter-Clockwise (CCW) order
#         normals = raw_normals 
#     else: # Vertices are in Clockwise (CW) order
#         normals = -raw_normals
    
#     # Normalize the normals for consistent distance calculations
#     norms = np.linalg.norm(normals, axis=1)
#     normals = normals / norms[:, np.newaxis]
    
#     offsets = np.einsum('ij,ij->i', normals, v1_array)
    
#     return normals, offsets

def calculate_and_print_penetration(point_name, point_coords, normals, offsets):
    """
    Calculates and prints the penetration for a given point against a set of half-planes.
    """
    print(f"\n--- Testing Point: '{point_name}' at {point_coords} ---")
    
    penetrations = []
    for i in range(normals.shape[0]):
        normal_vec = normals[i]
        offset_val = offsets[i]
        
        # This is the core calculation: penetration = d - (n · p)
        # A positive value means the point is "inside" this specific half-plane.
        penetration = offset_val - np.dot(normal_vec, point_coords)
        penetrations.append(penetration)
        
        print(f"  Edge {i}: Normal = [{normal_vec[0]:6.2f}, {normal_vec[1]:6.2f}], Offset = {offset_val:6.2f}, Penetration = {penetration:8.4f}")

    print("-" * 50)
    # The rule: The point is INSIDE if and only if ALL penetration values are non-negative.
    # We use a small tolerance for floating point comparisons.
    if all(p >= -1e-9 for p in penetrations):
        print("Conclusion: ALL penetration values are non-negative.")
        print("          => The point is INSIDE the polygon.")
    else:
        print("Conclusion: AT LEAST ONE penetration value is negative.")
        print("          => The point is OUTSIDE the polygon.")
    print("-" * 50)

def plot_results(vertices, normals, inside_point, outside_point):
    """
    Generates a plot to visualize the obstacle, points, and normal vectors.
    """
    fig, ax = plt.subplots(figsize=(10, 10))

    # 1. Plot the obstacle polygon
    obstacle_patch = Polygon(vertices, closed=True, color='k', alpha=0.4, label='Obstacle')
    ax.add_patch(obstacle_patch)

    # 2. Plot the test points
    ax.plot(inside_point[0], inside_point[1], 'go', markersize=10, label='UAV Inside')
    ax.plot(outside_point[0], outside_point[1], 'ro', markersize=10, label='UAV Outside')
    ax.text(inside_point[0] + 0.1, inside_point[1] + 0.1, 'Inside', color='g', fontsize=12)
    ax.text(outside_point[0] + 0.1, outside_point[1] + 0.1, 'Outside', color='r', fontsize=12)


    # 3. Plot the normal vectors starting from the midpoint of each edge
    v1_array = vertices
    v2_array = np.roll(vertices, -1, axis=0)
    edge_midpoints = (v1_array + v2_array) / 2
    
    ax.quiver(edge_midpoints[:, 0], edge_midpoints[:, 1], 
              normals[:, 0], normals[:, 1], 
              color='b', scale=4, width=0.005, label='Outward Normals')

    # 4. Finalize the plot
    ax.set_title("Obstacle Collision Test Visualization", fontsize=16)
    ax.set_xlabel("X Position", fontsize=12)
    ax.set_ylabel("Y Position", fontsize=12)
    ax.legend()
    ax.grid(True, linestyle='--')
    
    # Critical: Ensure the aspect ratio is equal
    ax.axis('equal')
    
    # Set plot limits to have some padding
    all_x = np.concatenate([vertices[:,0], [inside_point[0], outside_point[0]]])
    all_y = np.concatenate([vertices[:,1], [inside_point[1], outside_point[1]]])
    ax.set_xlim(all_x.min() - 1, all_x.max() + 1)
    ax.set_ylim(all_y.min() - 1, all_y.max() + 1)

    plt.show()


# --- Main Test Execution ---
if __name__ == "__main__":
    # 1. Define the vertices of a 2x2 square centered at (3, 3)
    square_vertices = np.array([
        [2.0, 2.0], [4.0, 2.0], [4.0, 4.0], [2.0, 4.0] 
    ])

    print("=" * 50)
    print("Running penetration test for a square obstacle.")
    print(f"Square Vertices:\n{square_vertices}")
    print("=" * 50)

    # 2. Get the half-plane representation
    obstacle_normals, obstacle_offsets = get_half_planes_vectorized(square_vertices)

    # 3. Define the test points
    uav_inside = np.array([3.0, 3.0])
    uav_outside = np.array([5.0, 5.0])

    # 4. Run the text-based calculations
    calculate_and_print_penetration("UAV Inside", uav_inside, obstacle_normals, obstacle_offsets)
    calculate_and_print_penetration("UAV Outside", uav_outside, obstacle_normals, obstacle_offsets)
    
    # 5. Generate the visualization
    print("\nGenerating plot...")
    plot_results(square_vertices, obstacle_normals, uav_inside, uav_outside)
    print("Plot window opened. Close the window to exit the script.")
