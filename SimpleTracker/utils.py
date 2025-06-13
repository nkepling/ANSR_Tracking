import numpy as np
import matplotlib.pyplot as plt
import json
import matplotlib.patches as patches
from functools import partial
from matplotlib import transforms
from matplotlib.path import Path
from scipy.sparse import lil_matrix
from scipy.sparse.csgraph import connected_components, shortest_path,yen,breadth_first_tree
from skimage.morphology import skeletonize, thin
from scipy.spatial import KDTree
import cv2
import networkx as nx



def build_kdtree(G):
    nodes = list(G.nodes)
    positions = np.array([G.nodes[n]['pos'] for n in nodes])
    kdtree = KDTree(positions)
    print("k-d tree built successfully.")
    return kdtree, nodes

def find_closest_node_kdtree(kdtree, node_list, uav_coords):
    dist, idx = kdtree.query(uav_coords, k=1) # k=1 for the single nearest neighbor
    return node_list[idx],dist

def fill_road_gaps(road_map: np.ndarray, kernel_size: int = 5) -> np.ndarray:
    """
    Fills small holes and gaps in a binary road map using a morphological closing operation.

    This is ideal for fixing spurious "0" pixels within a larger road segment,
    improving the connectivity of the road network for graph creation.

    Args:
        road_map (np.ndarray): The binary road map (0s and 1s).
        kernel_size (int): The size of the square kernel used for the closing operation.
                           A larger kernel will fill larger holes. A good starting
                           point is 5. Must be an odd number.

    Returns:
        np.ndarray: A new binary road map with the gaps filled.
    """
    if kernel_size % 2 == 0:
        print(f"Warning: kernel_size should be odd. Incrementing to {kernel_size + 1}.")
        kernel_size += 1

    # Convert the input map from (0, 1) to (0, 255) and uint8 type for OpenCV
    binary_image = (road_map * 255).astype(np.uint8)

    # Define the structuring element (kernel) for the operation
    kernel = np.ones((kernel_size, kernel_size), np.uint8)

    # Perform the morphological closing operation
    closed_image = cv2.morphologyEx(binary_image, cv2.MORPH_CLOSE, kernel)

    # Convert the processed image back to a binary map of (0, 1)
    cleaned_road_map = (closed_image > 0).astype(int)

    return cleaned_road_map

def skeletonize_roads(road_binary_map: np.ndarray) -> np.ndarray:
    """
    Simplifies a binary road map by reducing road segments to a single-pixel
    width using skeletonization (thinning). This preserves the connectivity
    and overall shape of the road network.

    Args:
        road_binary_map (np.ndarray): A 2D binary NumPy array (0s and 1s)
                                      where 1s represent road segments.

    Returns:
        np.ndarray: A new 2D binary NumPy array representing the skeletonized
                    (thinned) road map.
    """
    # Ensure the input is boolean, which skeletonize prefers
    # True for foreground (roads), False for background
    binary_input = road_binary_map.astype(bool)
    
    # Perform skeletonization
    # You can also explore `skimage.morphology.thin` for slightly different results
    skeleton = skeletonize(binary_input)
    
    # Convert back to int (0s and 1s) if preferred
    return skeleton.astype(int)


def create_grid_csgraph(binary_array: np.ndarray) -> lil_matrix:
    """
    Creates a SciPy csgraph (sparse adjacency matrix) from a binary array
    where '1's represent nodes and have edges to all direct neighbors
    (horizontal, vertical, and diagonal).

    Args:
        binary_array (np.ndarray): A 2D binary NumPy array (0s and 1s).

    Returns:
        lil_matrix: A sparse adjacency matrix (LIL format) representing the graph.
                    The value at (i, j) is 1 if there's an edge between node i and node j.
    """
    rows, cols = binary_array.shape
    num_nodes = rows * cols
    graph = lil_matrix((num_nodes, num_nodes), dtype=int)

    # Iterate through each cell in the binary array
    for r in range(rows):
        for c in range(cols):
            # If the current cell is a '1' (a node in our graph)
            if binary_array[r, c] == 1:
                current_node_idx = r * cols + c

                # Define relative coordinates for all 8 direct neighbors (including diagonals)
                # dr: delta row, dc: delta column
                neighbors = [
                    (-1, -1), (-1, 0), (-1, 1),  # Top row
                    (0, -1),           (0, 1),   # Middle row (excluding self)
                    (1, -1), (1, 0), (1, 1)    # Bottom row
                ]

                for dr, dc in neighbors:
                    n_r, n_c = r + dr, c + dc

                    # Check if the neighbor is within bounds
                    if 0 <= n_r < rows and 0 <= n_c < cols:
                        # If the neighbor is also a '1' (a valid node)
                        if binary_array[n_r, n_c] == 1:
                            neighbor_node_idx = n_r * cols + n_c
                            # Add an edge between the current node and its neighbor
                            # Since it's undirected, add edges in both directions
                            graph[current_node_idx, neighbor_node_idx] = 1
                            graph[neighbor_node_idx, current_node_idx] = 1
    return graph


def discretize_obstacle_map(obstacle_map: np.ndarray, dim: tuple[int, int], obs_thresh: float) -> tuple[np.ndarray, np.ndarray]:
    """
    Downsamples a high-resolution obstacle map to a lower resolution and returns
    the centroid coordinates for each cell in the new grid.

    A cell in the new grid is marked as an obstacle if the percentage of
    obstacle cells in its corresponding high-resolution region exceeds a
    threshold.

    Args:
        obstacle_map (np.ndarray): The high-resolution 2D obstacle map.
                                   Assumes 1 for obstacle, 0 for free space.
        dim (tuple[int, int]): The new, smaller grid dimensions (rows, cols).
        obs_thresh (float): The threshold (0.0 to 1.0) for marking a new cell
                            as an obstacle. For example, 0.25 means a cell is
                            an obstacle if more than 25% of its area is blocked.

    Raises:
        ValueError: If the original map dimensions are not perfectly divisible
                    by the new dimensions.

    Returns:
        tuple[np.ndarray, np.ndarray]: A tuple containing:
            - low_res_map (np.ndarray): The new, lower-resolution obstacle map.
            - centroids (np.ndarray): A 2D array of shape (new_rows, new_cols, 2),
                                      where centroids[r, c] contains the (row, col)
                                      centroid coordinates (in high-resolution pixel space)
                                      for the cell at (r, c) in the low-resolution map.
    """
    orig_rows, orig_cols = obstacle_map.shape
    new_rows, new_cols = dim

    if orig_rows % new_rows != 0 or orig_cols % new_cols != 0:
        raise ValueError(
            f"Original map shape {obstacle_map.shape} must be divisible by the new dimensions {dim}."
        )

    block_rows = orig_rows // new_rows
    block_cols = orig_cols // new_cols

    reshaped_map = obstacle_map.reshape(new_rows, block_rows, new_cols, block_cols)
    transposed_map = reshaped_map.transpose(0, 2, 1, 3)
    block_means = transposed_map.mean(axis=(2, 3))

    low_res_map = np.where(block_means > obs_thresh, 1, 0)

    # Calculate centroids for each new low-resolution cell
    centroids = np.zeros((new_rows, new_cols, 2), dtype=float)
    for r_low in range(new_rows):
        for c_low in range(new_cols):
            # Calculate the starting (top-left) pixel coordinates of the block
            start_row = r_low * block_rows
            start_col = c_low * block_cols

            # Calculate the centroid by adding half the block dimensions to the start
            centroid_row = start_row + (block_rows - 1) / 2.0
            centroid_col = start_col + (block_cols - 1) / 2.0
            
            centroids[r_low, c_low] = [centroid_row, centroid_col]

    return low_res_map, centroids

def load_mission(mission_file):
    with open(mission_file, 'r') as file:
        d = json.load(file)

    mission = d["scenario_objective"]

    eoi = mission["entities_of_interest"]
    constaints = d["scenario_constraints"]
    
    return (mission, eoi,constaints)


def coordinate_to_pixel(coord, resolution, center):
    x_pixel = int((coord[0] - center[0]) / resolution)
    y_pixel = int((coord[1] - center[1]) / resolution)
    return x_pixel, y_pixel

def pixel_to_coordinate(pixel, resolution, center):
    x = center[0] + (pixel[0] - 0.5) * resolution
    y = center[1] + (pixel[1] - 0.5) * resolution
    return x, y

def rotate_pixel(pixel, shape):
    # shape: (height, width)
    x, y = pixel
    return y, shape[0] - 1 - x

def load_roads(filename,visualize=True):
    seg_map = np.load(filename)
    seg_ids = seg_map["data"]
    resolution = seg_map["resolution"]

    road_ids = np.where(
        np.any(
            np.all(SEGID_COLORS[:, None] == ROAD, axis=2),
            axis=1
        )
    )[0]

    print("Road IDs:", road_ids)    

    # Now, create a mask for roads in seg_ids
    roads = np.isin(seg_ids, road_ids).astype(int)
    if visualize:
        plt.imshow(roads, cmap='gray')
        plt.title("Roads in City Map")
        plt.gca().invert_yaxis()
        plt.show()

    return roads, resolution

def get_belief_map(eoi):
    for entity in eoi:
        entity_priors = entity.get("entity_priors", {})
        belief_map = entity_priors.get("location_belief_map", {})[0]     
    return belief_map

def load_obstacle_map(filename,depth=0):
    """
    Load the obstacle map from a .npz file. Depth is flying heigh of UAV
    """
    npz = np.load(filename)
    data = npz["data"]
    center = npz["center"]
    resolution = float(npz["resolution"][0])

    map_width = data.shape[1]
    map_height = data.shape[0]
    origin_x = center[0] - (map_width / 2) * resolution
    origin_y = center[1] - (map_height / 2) * resolution
    print(f"Obstacle Map Origin: ({origin_x}, {origin_y})")

    obstacle_map = np.where(data > depth, True, False).astype(int)

    return obstacle_map, resolution, (origin_x, origin_y)

def plot_belief_on_city_map(segmap_file, mission_description_file, obstacle_map_file, visualize=True,depth=0):
    """
    Plot the belief map on the city map.
    """
    roads, resolution = load_roads(segmap_file, visualize=False)
    roads = roads.T
    roads = np.rot90(roads) 

    mission, eoi, constraints = load_mission(mission_description_file)
    belief_map = get_belief_map(eoi)
    coords = belief_map["polygon_vertices"]  # Should be Nx2 array of (x, y) in world coordinates
    pixel_coords = []
    seg_map = np.load(segmap_file)
    center = seg_map["center"]

    im_width = roads.shape[1]
    im_height = roads.shape[0]

    im_center_x = im_width // 2
    im_center_y = im_height // 2

    origin_x = center[0] - 0.5 * im_width * resolution
    origin_y = center[1] - 0.5 * im_height * resolution

    ####### Belief coords
    origin_pixel = coordinate_to_pixel([0, 0], resolution, [origin_y, origin_x])
    partial_coordinate_to_pixel = partial(coordinate_to_pixel, resolution=resolution, center=[origin_x, origin_y])
    for coord in coords:
        pixel_coords.append(list(map(partial_coordinate_to_pixel,coord)))

    pixel_coords = np.array(pixel_coords)

    fig, ax = plt.subplots()
    ax.imshow(roads, cmap='bone')
    pixel_coords_rot = []

    for poly in pixel_coords:
        pixel_coords_rot.append([rotate_pixel(pt, roads.shape) for pt in poly])
    origin_pixel_rot = rotate_pixel(origin_pixel, roads.shape)

    ############ KOZ

    koz_coords = constraints["spatial_constraints"]["keep_out_zones"]

    for koz in koz_coords:
        koz_poly = koz["keep_out_polygon_vertices"]
        koz_pixel_coords = list(map(partial(coordinate_to_pixel, resolution=resolution, center=[origin_x, origin_y]), koz_poly))
        koz_pixel_coords_rot = [rotate_pixel(pt, roads.shape) for pt in koz_pixel_coords]
        koz_patch = patches.Polygon(koz_pixel_coords_rot, closed=True, edgecolor='red', facecolor='red', linewidth=2, alpha=0.5)
        ax.add_patch(koz_patch)

    stay_within_coords = constraints["spatial_constraints"]["stay_within_zones"]

    for stay_within in stay_within_coords:
        stay_within_poly = stay_within["stay_within_polygon_vertices"]
        stay_within_pixel_coords = list(map(partial(coordinate_to_pixel, resolution=resolution, center=[origin_x, origin_y]), stay_within_poly))
        stay_within_pixel_coords_rot = [rotate_pixel(pt, roads.shape) for pt in stay_within_pixel_coords]
        stay_within_patch = patches.Polygon(stay_within_pixel_coords_rot, closed=True, edgecolor='blue', facecolor='blue', linewidth=2, alpha=0.5)
        ax.add_patch(stay_within_patch)

    ######## add obstacle map
    obstacle_map, resolution, (origin_x, origin_y) = load_obstacle_map(obstacle_map_file, depth=depth)
    obstacle_map = obstacle_map.T
    obstacle_map = np.rot90(obstacle_map)  # Rotate the obstacle map to match the roads orientation
    
    ######## Plotting

    start = patches.Polygon(pixel_coords_rot[0], closed=True, edgecolor='g', facecolor='g', linewidth=2,alpha=0.5)
    ax.scatter(origin_pixel_rot[0], origin_pixel_rot[1], c='red', s=50, label='Origin (0,0)')
    ax.add_patch(start)
    end = patches.Polygon(pixel_coords_rot[1], closed=True, edgecolor='g', facecolor='g', linewidth=2,alpha=0.5)
    ax.add_patch(end)
    ax.set_title("Belief Polygon on Roads Map")

    xticks = ax.get_xticks()
    yticks = ax.get_yticks()
    # xlabels = [f"{origin_y + x * resolution}" for x in xticks]
    # ylabels = [f"{origin_x + y* resolution}" for y in yticks]
    # ax.set_xticklabels(xlabels)
    # ax.set_yticklabels(ylabels)
    ax.set_xlabel("World X")
    ax.set_ylabel("World Y")

    plt.show()
    fig, ax = plt.subplots()
    ax.imshow(roads, cmap='bone')

    ax.imshow(np.ma.masked_where(obstacle_map == 0, obstacle_map), cmap='Reds', alpha=0.5)

    koz_coords = constraints["spatial_constraints"]["keep_out_zones"]

    for koz in koz_coords:
        koz_poly = koz["keep_out_polygon_vertices"]
        koz_pixel_coords = list(map(partial(coordinate_to_pixel, resolution=resolution, center=[origin_x, origin_y]), koz_poly))
        koz_pixel_coords_rot = [rotate_pixel(pt, roads.shape) for pt in koz_pixel_coords]
        koz_patch = patches.Polygon(koz_pixel_coords_rot, closed=True, edgecolor='red', facecolor='red', linewidth=2, alpha=0.5)
        ax.add_patch(koz_patch)

    stay_within_coords = constraints["spatial_constraints"]["stay_within_zones"]

    for stay_within in stay_within_coords:
        stay_within_poly = stay_within["stay_within_polygon_vertices"]
        stay_within_pixel_coords = list(map(partial(coordinate_to_pixel, resolution=resolution, center=[origin_x, origin_y]), stay_within_poly))
        stay_within_pixel_coords_rot = [rotate_pixel(pt, roads.shape) for pt in stay_within_pixel_coords]
        stay_within_patch = patches.Polygon(stay_within_pixel_coords_rot, closed=True, edgecolor='blue', facecolor='none', linewidth=2, alpha=0.9)
        ax.add_patch(stay_within_patch)


    for poly in pixel_coords_rot:
        mask_shape = roads.shape
        belief_mask = np.zeros(mask_shape, dtype=bool)
        # Use the first polygon (adjust if you have multiple)
        poly_path = Path(poly)
        y_grid, x_grid = np.mgrid[0:mask_shape[0], 0:mask_shape[1]]
        points = np.vstack((x_grid.ravel(), y_grid.ravel())).T
        mask = poly_path.contains_points(points).reshape(mask_shape)
        belief_mask[mask] = True

        intersection = np.logical_and(belief_mask, roads > 0)

        ax.imshow(np.ma.masked_where(~intersection, intersection), cmap='spring', alpha=0.9)




    ax.scatter(origin_pixel_rot[0], origin_pixel_rot[1], c='red', s=50, label='Origin (0,0)')
    ax.set_title("Belief Polygon ∩ Roads")
    ax.set_xlabel("World X")
    ax.set_ylabel("World Y")
    plt.show()


def plot_map(city_map,save_file=None):

    fig,ax = plt.subplots()
    ax.imshow(city_map,origin="lower",cmap="binary")

    if save_file is not None:
        plt.savefig(save_file)

    plt.show()


def get_evader_path_from_file(file_name,obstacle_map,roads,resolution,center,visualize=False):
    """Plot evader path with roads
    """

    origin_x,origin_y = center
    data = np.loadtxt(file_name, delimiter=',', dtype=float)
    im_width = roads.shape[1]
    im_height = roads.shape[0]

    im_center_x = im_width // 2
    im_center_y = im_height // 2
    
    origin_pixel = coordinate_to_pixel([0, 0], resolution, [origin_x, origin_y])
    origin_pixel_rot = rotate_pixel(origin_pixel, roads.shape)

    partial_coordinate_to_pixel = partial(coordinate_to_pixel, 
                                          resolution=resolution, 
                                          center=[origin_x, origin_y]) # Corrected origin format if needed
    
    pixel_coords = np.apply_along_axis(partial_coordinate_to_pixel, 
                                     axis=1, 
                                     arr=data[:, :2])
    rotated_pixels = np.apply_along_axis(lambda x: rotate_pixel([x[0],x[1]],obstacle_map.shape),axis=1,arr=pixel_coords)
    start_pos = pixel_coords[0, :]
    end_pos = pixel_coords[-1, :]

    if visualize:

        print("Generating plot...")
        fig, ax = plt.subplots(figsize=(12, 12))
        ax.scatter(origin_pixel_rot[0], origin_pixel_rot[1], c='red', s=50, label='Origin (0,0)')
        ax.imshow(obstacle_map, cmap="binary", origin='lower') 
        ax.imshow(roads, cmap="grey", alpha=0.4)
        ax.plot(rotated_pixels[:, 0], rotated_pixels[:, 1], 'red', lw=2, label="Evader's Path")
        ax.scatter(rotated_pixels[0][0], rotated_pixels[0][1], c='lime', s=150, label='Start', zorder=5, marker='o', edgecolors='black')
        ax.scatter(rotated_pixels[-1][0], rotated_pixels[-1][1], c='red', s=150, label='End', zorder=5, marker='X')
        ax.set_title("Evader Path on City Map")
        ax.set_xlabel("X Pixel Coordinate")
        ax.set_ylabel("Y Pixel Coordinate")
        ax.legend()
        plt.show()

    return rotated_pixels


def visualize_graph_on_grid(
    high_res_map: np.ndarray,
    low_res_map: np.ndarray,
    centroids: np.ndarray,
    graph: lil_matrix,
    paths_to_highlight: list = None, # List of paths (list of node indices) to draw
    title: str = "Graph on High-Resolution Grid"
):
    """
    Visualizes the graph nodes (centroids) and edges on the original
    high-resolution grid, with an option to highlight specific paths.

    Args:
        high_res_map (np.ndarray): The original high-resolution 2D map.
        low_res_map (np.ndarray): The low-resolution obstacle map.
        centroids (np.ndarray): A 2D array of shape (new_rows, new_cols, 2)
                                containing the (row, col) centroid coordinates
                                for each cell in the low-resolution map.
        graph (lil_matrix): The sparse adjacency matrix representing the graph
                            of the low-resolution map.
        paths_to_highlight (list, optional): A list where each element is a list
                                             of node indices representing a path.
                                             These paths will be drawn in different colors.
                                             Defaults to None.
        title (str, optional): Title for the plot. Defaults to "Graph on High-Resolution Grid".
    """
    fig, ax = plt.subplots(figsize=(10, 10))

    ax.imshow(high_res_map, cmap='binary', origin='lower', alpha=0.6)
    
    rows, cols = high_res_map.shape
    new_rows, new_cols = low_res_map.shape
    
    block_rows = rows // new_rows
    block_cols = cols // new_cols

    for i in range(1, new_rows):
        ax.axhline(i * block_rows - 0.5, color='gray', linestyle='--', linewidth=0.5)
    for j in range(1, new_cols):
        ax.axvline(j * block_cols - 0.5, color='gray', linestyle='--', linewidth=0.5)

    # Plot nodes (centroids of valid low-res cells)
    node_x_coords = []
    node_y_coords = []
    
    # Create a reverse mapping from low-res (r,c) to flattened index
    low_res_idx_map = np.full(low_res_map.shape, -1, dtype=int)
    current_node_idx = 0
    for r_low in range(new_rows):
        for c_low in range(new_cols):
            if low_res_map[r_low, c_low] == 1:
                node_y_coords.append(centroids[r_low, c_low, 0])
                node_x_coords.append(centroids[r_low, c_low, 1])
                low_res_idx_map[r_low, c_low] = current_node_idx
                current_node_idx += 1

    ax.plot(node_x_coords, node_y_coords, 'ro', markersize=4, label='Graph Nodes (Centroids)')

    # Plot edges
    graph_coo = graph.tocoo()
    for i, j, _ in zip(graph_coo.row, graph_coo.col, graph_coo.data):
        if i >= j: # Draw each edge once
            continue
            
        r1_low, c1_low = np.unravel_index(i, low_res_map.shape)
        r2_low, c2_low = np.unravel_index(j, low_res_map.shape)
        
        x1_centroid, y1_centroid = centroids[r1_low, c1_low, 1], centroids[r1_low, c1_low, 0]
        x2_centroid, y2_centroid = centroids[r2_low, c2_low, 1], centroids[r2_low, c2_low, 0]
        
        ax.plot([x1_centroid, x2_centroid], [y1_centroid, y2_centroid], 'b-', linewidth=1, alpha=0.7)

    # Highlight K shortest paths
    if paths_to_highlight:
        colors = plt.cm.get_cmap('viridis', len(paths_to_highlight)) # Get a colormap for paths
        for k_idx, path_nodes in enumerate(paths_to_highlight):
            path_x = []
            path_y = []
            for node_idx in path_nodes:
                # Need to convert flattened node_idx back to low-res (r,c)
                r_low, c_low = np.unravel_index(node_idx, low_res_map.shape)
                path_x.append(centroids[r_low, c_low, 1])
                path_y.append(centroids[r_low, c_low, 0])
            ax.plot(path_x, path_y, marker='o', linestyle='-', linewidth=2, color=colors(k_idx), 
                    label=f'Path {k_idx+1} (Len: {len(path_nodes)-1})', markersize=7, alpha=0.9) # Length is number of edges

    ax.set_title(title)
    ax.set_xlabel("Original Grid Column (X)")
    ax.set_ylabel("Original Grid Row (Y)")
    ax.set_aspect('equal', adjustable='box')
    ax.legend()
    plt.grid(False)
    plt.show()

def visualize_bfs_on_skeleton(
    high_res_map: np.ndarray,
    low_res_map: np.ndarray,
    centroids: np.ndarray,
    start_node_idx: int,
    bfs_paths: dict,
    max_depth: int,
    title: str = "BFS Exploration Results"
):
    """
    Visualizes the results of a Breadth-First Search on the graph, with each
    path to the frontier color-coded for clarity.

    Args:
        high_res_map (np.ndarray): The original high-resolution 2D map for context.
        low_res_map (np.ndarray): The low-resolution map used to build the graph.
        centroids (np.ndarray): Centroid coordinates for the low-resolution grid cells.
        start_node_idx (int): The flattened index of the starting node for the BFS.
        bfs_paths (dict): The output from the BFS function, mapping destination nodes to their paths.
        max_depth (int): The maximum depth of the search, used for the title.
        title (str, optional): The title for the plot.
    """
    fig, ax = plt.subplots(figsize=(12, 12))

    # 1. Plot the high-resolution map as the background
    ax.imshow(high_res_map, cmap='binary', origin='lower', alpha=0.7)

    paths_to_visualize = list(bfs_paths.values())

    # 2. Plot all the paths found by BFS with unique colors
    if paths_to_visualize:
        # --- NEW: Set up a colormap ---
        # This creates a color generator that will give us a unique color for each path.
        # 'nipy_spectral' and 'gist_rainbow' are good choices for many distinct colors.
        num_paths = len(paths_to_visualize)
        colors = plt.cm.get_cmap('nipy_spectral', num_paths)

        # --- MODIFIED: Loop with enumerate to get an index for coloring ---
        for i, path_nodes in enumerate(paths_to_visualize):
            path_x = []
            path_y = []
            for node_idx in path_nodes:
                r_low, c_low = np.unravel_index(node_idx, low_res_map.shape)
                path_x.append(centroids[r_low, c_low, 1])
                path_y.append(centroids[r_low, c_low, 0])
            
            # --- MODIFIED: Use the generated color for this specific path ---
            ax.plot(path_x, path_y, color=colors(i), linewidth=2.5, alpha=0.9)

    # 3. Highlight all the explored nodes on the frontier
    frontier_nodes_indices = list(bfs_paths.keys())
    if frontier_nodes_indices:
        node_x_coords = []
        node_y_coords = []
        for node_idx in frontier_nodes_indices:
            r_low, c_low = np.unravel_index(node_idx, low_res_map.shape)
            node_x_coords.append(centroids[r_low, c_low, 1])
            node_y_coords.append(centroids[r_low, c_low, 0])
        ax.scatter(node_x_coords, node_y_coords, c='white', s=60, label='Frontier Nodes', zorder=5, edgecolors='black')

    # 4. Clearly mark the start node
    start_r, start_c = np.unravel_index(start_node_idx, low_res_map.shape)
    start_x = centroids[start_r, start_c, 1]
    start_y = centroids[start_r, start_c, 0]
    ax.scatter(start_x, start_y, c='lime', s=250, marker='*', label='Start Node', zorder=10, edgecolors='black')

    ax.set_title(f"{title} (from Node {start_node_idx} to Depth {max_depth})")
    ax.set_xlabel("Grid Column (X)")
    ax.set_ylabel("Grid Row (Y)")
    ax.set_aspect('equal', adjustable='box')
    ax.legend()
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.show()





    # segmentation colors for different categories
ROAD = [
    [43, 47, 206],   # road
    #[134, 57, 119],  # sidewalk
    [215, 4, 215],   # crossing
    [101, 109, 181],  # crossing
    #[0, 53, 65],     # parking lot
]
BUILDING = [
    [153, 108, 6],
    [184, 145, 182],
    [156, 198, 23],
    [146, 52, 70],
    [202, 97, 155],
    [1, 222, 192],
    [218, 124, 115],
]

# RGB values for segmentation ids
SEGID_COLORS = np.array([
    [0, 0, 0],
    [153, 108, 6],
    [112, 105, 191],
    [89, 121, 72],
    [190, 225, 64],
    [206, 190, 59],
    [81, 13, 36],
    [115, 176, 195],
    [161, 171, 27],
    [135, 169, 180],
    [29, 26, 199],
    [102, 16, 239],
    [242, 107, 146],
    [156, 198, 23],
    [49, 89, 160],
    [68, 218, 116],
    [11, 236, 9],
    [196, 30, 8],
    [121, 67, 28],
    [0, 53, 65],
    [146, 52, 70],
    [226, 149, 143],
    [151, 126, 171],
    [194, 39, 7],
    [205, 120, 161],
    [212, 51, 60],
    [211, 80, 208],
    [189, 135, 188],
    [54, 72, 205],
    [103, 252, 157],
    [124, 21, 123],
    [19, 132, 69],
    [195, 237, 132],
    [94, 253, 175],
    [182, 251, 87],
    [90, 162, 242],
    [199, 29, 1],
    [254, 12, 229],
    [35, 196, 244],
    [220, 163, 49],
    [86, 254, 214],
    [152, 3, 129],
    [92, 31, 106],
    [207, 229, 90],
    [125, 75, 48],
    [98, 55, 74],
    [126, 129, 238],
    [222, 153, 109],
    [85, 152, 34],
    [173, 69, 31],
    [37, 128, 125],
    [58, 19, 33],
    [134, 57, 119],
    [218, 124, 115],
    [120, 0, 200],
    [225, 131, 92],
    [246, 90, 16],
    [51, 155, 241],
    [202, 97, 155],
    [184, 145, 182],
    [96, 232, 44],
    [133, 244, 133],
    [180, 191, 29],
    [1, 222, 192],
    [99, 242, 104],
    [91, 168, 219],
    [65, 54, 217],
    [148, 66, 130],
    [203, 102, 204],
    [216, 78, 75],
    [234, 20, 250],
    [109, 206, 24],
    [164, 194, 17],
    [157, 23, 236],
    [158, 114, 88],
    [245, 22, 110],
    [67, 17, 35],
    [181, 213, 93],
    [170, 179, 42],
    [52, 187, 148],
    [247, 200, 111],
    [25, 62, 174],
    [100, 25, 240],
    [191, 195, 144],
    [252, 36, 67],
    [241, 77, 149],
    [237, 33, 141],
    [119, 230, 85],
    [28, 34, 108],
    [78, 98, 254],
    [114, 161, 30],
    [75, 50, 243],
    [66, 226, 253],
    [46, 104, 76],
    [8, 234, 216],
    [15, 241, 102],
    [93, 14, 71],
    [192, 255, 193],
    [253, 41, 164],
    [24, 175, 120],
    [185, 243, 231],
    [169, 233, 97],
    [243, 215, 145],
    [72, 137, 21],
    [160, 113, 101],
    [214, 92, 13],
    [167, 140, 147],
    [101, 109, 181],
    [53, 118, 126],
    [3, 177, 32],
    [40, 63, 99],
    [186, 139, 153],
    [88, 207, 100],
    [71, 146, 227],
    [236, 38, 187],
    [215, 4, 215],
    [18, 211, 66],
    [113, 49, 134],
    [47, 42, 63],
    [219, 103, 127],
    [57, 240, 137],
    [227, 133, 211],
    [145, 71, 201],
    [217, 173, 183],
    [250, 40, 113],
    [208, 125, 68],
    [224, 186, 249],
    [69, 148, 46],
    [239, 85, 20],
    [108, 116, 224],
    [56, 214, 26],
    [179, 147, 43],
    [48, 188, 172],
    [221, 83, 47],
    [155, 166, 218],
    [62, 217, 189],
    [198, 180, 122],
    [201, 144, 169],
    [132, 2, 14],
    [128, 189, 114],
    [163, 227, 112],
    [45, 157, 177],
    [64, 86, 142],
    [118, 193, 163],
    [14, 32, 79],
    [200, 45, 170],
    [74, 81, 2],
    [59, 37, 212],
    [73, 35, 225],
    [95, 224, 39],
    [84, 170, 220],
    [159, 58, 173],
    [17, 91, 237],
    [31, 95, 84],
    [34, 201, 248],
    [63, 73, 209],
    [129, 235, 107],
    [231, 115, 40],
    [36, 74, 95],
    [238, 228, 154],
    [61, 212, 54],
    [13, 94, 165],
    [141, 174, 0],
    [140, 167, 255],
    [117, 93, 91],
    [183, 10, 186],
    [165, 28, 61],
    [144, 238, 194],
    [12, 158, 41],
    [76, 110, 234],
    [150, 9, 121],
    [142, 1, 246],
    [230, 136, 198],
    [5, 60, 233],
    [232, 250, 80],
    [143, 112, 56],
    [187, 70, 156],
    [2, 185, 62],
    [138, 223, 226],
    [122, 183, 222],
    [166, 245, 3],
    [175, 6, 140],
    [240, 59, 210],
    [248, 44, 10],
    [83, 82, 52],
    [223, 248, 167],
    [87, 15, 150],
    [111, 178, 117],
    [197, 84, 22],
    [235, 208, 124],
    [9, 76, 45],
    [176, 24, 50],
    [154, 159, 251],
    [149, 111, 207],
    [168, 231, 15],
    [209, 247, 202],
    [80, 205, 152],
    [178, 221, 213],
    [27, 8, 38],
    [244, 117, 51],
    [107, 68, 190],
    [23, 199, 139],
    [171, 88, 168],
    [136, 202, 58],
    [6, 46, 86],
    [105, 127, 176],
    [174, 249, 197],
    [172, 172, 138],
    [228, 142, 81],
    [7, 204, 185],
    [22, 61, 247],
    [233, 100, 78],
    [127, 65, 105],
    [33, 87, 158],
    [139, 156, 252],
    [42, 7, 136],
    [20, 99, 179],
    [79, 150, 223],
    [131, 182, 184],
    [110, 123, 37],
    [60, 138, 96],
    [210, 96, 94],
    [123, 48, 18],
    [137, 197, 162],
    [188, 18, 5],
    [39, 219, 151],
    [204, 143, 135],
    [249, 79, 73],
    [77, 64, 178],
    [41, 246, 77],
    [16, 154, 4],
    [116, 134, 19],
    [4, 122, 235],
    [177, 106, 230],
    [21, 119, 12],
    [104, 5, 98],
    [50, 130, 53],
    [30, 192, 25],
    [26, 165, 166],
    [10, 160, 82],
    [106, 43, 131],
    [44, 216, 103],
    [255, 101, 221],
    [32, 151, 196],
    [213, 220, 89],
    [70, 209, 228],
    [97, 184, 83],
    [82, 239, 232],
    [251, 164, 128],
    [193, 11, 245],
    [38, 27, 159],
    [229, 141, 203],
    [130, 56, 55],
    [147, 210, 11],
    [162, 203, 118],
    [43, 47, 206],
], dtype=np.uint8)

def get_kdtree(roads):
    filled_road = fill_road_gaps(roads, kernel_size=11)
    skeletonized_roads = skeletonize_roads(filled_road)


    G = nx.Graph()
    rows, cols = skeletonized_roads.shape
    for r in range(rows):
        for c in range(cols):
            if skeletonized_roads[r, c] == 1:
                current_node = (r, c)
                G.add_node(current_node)

                for dr in [-1, 0, 1]:
                    for dc in [-1, 0, 1]:
                        if dr == 0 and dc == 0:
                            continue
                        neighbor_r, neighbor_c = r + dr, c + dc
                        if 0 <= neighbor_r < rows and 0 <= neighbor_c < cols and \
                        skeletonized_roads[neighbor_r, neighbor_c] == 1:
                            neighbor_node = (neighbor_r, neighbor_c)
                            G.add_edge(current_node, neighbor_node)

    
    positions = {node: (node[1], node[0]) for node in G.nodes()}

    nx.set_node_attributes(G, positions, name="pos")


    full_G = nx.Graph()
    rows, cols = filled_road.shape
    for r in range(rows):
        for c in range(cols):
            if filled_road[r, c] == 1:
                current_node = (r, c)
                full_G.add_node(current_node)

                for dr in [-1, 0, 1]:
                    for dc in [-1, 0, 1]:
                        if dr == 0 and dc == 0:
                            continue
                        neighbor_r, neighbor_c = r + dr, c + dc
                        if 0 <= neighbor_r < rows and 0 <= neighbor_c < cols and \
                        filled_road[neighbor_r, neighbor_c] == 1:
                            neighbor_node = (neighbor_r, neighbor_c)
                            full_G.add_edge(current_node, neighbor_node)



    positions = {node: (node[1], node[0]) for node in full_G.nodes()}

    nx.set_node_attributes(full_G, positions, name="pos")

    kdtree, nodes_list = build_kdtree(G)
    # kdtree, nodes_list = build_kdtree(full_G)

    return G,kdtree,nodes_list




if __name__ == "__main__":
    import time
    from matplotlib.patches import Circle
    from dummy_pwm import interpolate_by_time
    
    segmap_file = "city_1000_1000_seg_segids.npz"
    mission_description_file = "description.json"
    obstacle_map_file = "city_1000_1000.npz"    

    # --- 1. Data Loading and Preprocessing ---
    print("Loading maps and mission data...")
    roads, resolution = load_roads(segmap_file, visualize=False)
    roads = np.rot90(roads.T)
    dim = (1000, 1000)
    
    # --- 2. Graph Creation from Skeletonized Roads ---
    print("Skeletonizing road map and building graph...")
    filled_road = fill_road_gaps(roads, kernel_size=11)
    skeletonized_roads = skeletonize_roads(filled_road)
    
    skeletonized_roads = skeletonized_roads
    # fig,ax = plt.subplots(figsize=(10,10))
    # # ax.imshow(filled_road,cmap="binary")
    # ax.imshow(skeletonized_roads,cmap="RdGy")

    # plt.show()

    # --- 1. Build the graph (same as before) ---
    G = nx.Graph()
    rows, cols = skeletonized_roads.shape
    for r in range(rows):
        for c in range(cols):
            if skeletonized_roads[r, c] == 1:
                current_node = (r, c)
                G.add_node(current_node)

                for dr in [-1, 0, 1]:
                    for dc in [-1, 0, 1]:
                        if dr == 0 and dc == 0:
                            continue
                        neighbor_r, neighbor_c = r + dr, c + dc
                        if 0 <= neighbor_r < rows and 0 <= neighbor_c < cols and \
                        skeletonized_roads[neighbor_r, neighbor_c] == 1:
                            neighbor_node = (neighbor_r, neighbor_c)
                            G.add_edge(current_node, neighbor_node)

    print(f"Graph created with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges.")

    positions = {node: (node[1], node[0]) for node in G.nodes()}

    nx.set_node_attributes(G, positions, name="pos")

    print("\nSuccessfully added 'pos' attribute to all nodes.")

    kdtree, nodes_list = build_kdtree(G)
    uav_coords = (633, 291)
    start_time = time.perf_counter()
    center_node,dist = find_closest_node_kdtree(kdtree, nodes_list, uav_coords)
    center_coords = G.nodes[center_node]['pos']
    search_radius = 50.0 # e.g., 15 meters

    indices_in_radius = kdtree.query_ball_point(center_coords, r=search_radius)
    
    nodes_in_radius = [nodes_list[i] for i in indices_in_radius]
    subgraph = G.subgraph(nodes_in_radius)
    all_paths_from_center = nx.single_source_shortest_path(subgraph,center_node)
    print(len(all_paths_from_center))
    paths_to_frontier = {}
    for target_node, path in all_paths_from_center.items():
        if len(path) - 1 >= 200:
            paths_to_frontier[target_node] = path
    
    print(len(paths_to_frontier))
    # trajectories = []
    # for i in paths_to_frontier.values():
    #     new_path = interpolate_by_time(np.array(i),0.1,20)
    #     traj = new_path[:20]
    #     trajectories.append(traj)

    
    lookup_time_ms = (time.perf_counter() - start_time) * 1000

    # TODO: Filter based on UAV position vector
    # TODO: Filter start path to start at uav location 
    # TODO: add smoothing using spines
    
    print(f"Closest graph node found and subgraph grabbed : {center_node} in {lookup_time_ms:.3f} ms.")
    # print(f"Original graph has {G.number_of_nodes()} nodes.")
    # # print(f"Subgraph within {search_radius} units of node {center_node} has {subgraph_by_distance.number_of_nodes()} nodes.")

    # print(center_node)
    # print(dist)


    # fig,ax = plt.subplots()
    # ax.imshow(roads,cmap="binary")


    # print("\n--- Subgraph by Physical Distance ---")
  
    # # 6. Visualize
    # pos = nx.get_node_attributes(G, 'pos')
    # fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 7))

    # # Plot full graph
    # ax1.set_title("Full Graph")
    # nx.draw(G, pos, ax=ax1, node_size=30, node_color='lightgray')
    # nx.draw_networkx_nodes(G, pos, nodelist=[center_node], ax=ax1, node_size=100, node_color='red')

    # # Plot subgraph
    # ax2.set_title(f"Subgraph (Radius = {search_radius} units)")
    # nx.draw(G, pos, ax=ax2, node_size=30, node_color='lightgray')
    # nx.draw(subgraph_by_distance, pos, ax=ax2, node_size=50, node_color='skyblue')
    # nx.draw_networkx_nodes(G, pos, nodelist=[center_node], ax=ax2, node_size=100, node_color='red')
    # # Add a circle to show the search radius
    # circle = Circle(center_coords, search_radius, color='red', fill=False, linestyle='--', linewidth=2)
    # ax2.add_patch(circle)
    # ax2.set_aspect('equal')


        
    # ax.plot(uav_coords[0],uav_coords[1],"o",color="red")
    # ax.plot(center_node[1],center_node[0],"x",color="green")
    # plt.show()
