import numpy as np
import matplotlib.pyplot as plt
from load_mission import load_mission # Assuming this function is defined elsewhere and works
import matplotlib.patches as patches
from functools import partial

# --- (Your existing SEGMENTATION DATA, ROAD, BUILDING, SEGID_COLORS arrays remain unchanged) ---
# Segmentation map data

# segmentation colors for different categories
ROAD = [
    [43, 47, 206],  # road
    # [134, 57, 119],  # sidewalk
    [215, 4, 215],  # crossing
    [101, 109, 181],  # crossing
    # [0, 53, 65],       # parking lot
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

def coordinate_to_pixel(coord, resolution, center):
    """Converts world coordinates to pixel coordinates."""
    # coord: [world_x, world_y]
    # center: world coordinates of the pixel (0,0) (e.g., bottom-left map corner)
    # resolution: map resolution (world units per pixel)
    x_pixel = int((coord[0] - center[0]) / resolution)
    y_pixel = int((coord[1] - center[1]) / resolution)
    return x_pixel, y_pixel

def pixel_to_coordinate(pixel, resolution, center):
    """Converts pixel coordinates to world coordinates (center of pixel)."""
    # pixel: [col, row]
    # center: world coordinates of the pixel (0,0) (e.g., bottom-left map corner)
    # resolution: map resolution (world units per pixel)
    # To get the center of the pixel, we add 0.5
    x = center[0] + (pixel[0] + 0.5) * resolution
    y = center[1] + (pixel[1] + 0.5) * resolution
    return x, y

def load_map_data(filename):
    """
    Load the map data from a .npz file.
    (This function was not directly used in plot_belief_on_city_map,
     but its logic for origin calculation is relevant)
    """
    npz_file = np.load(filename)
    data = npz_file["data"]
    center_coords = npz_file["center"]
    res = float(npz_file["resolution"][0])

    map_width_pixels = data.shape[1]
    map_height_pixels = data.shape[0]
    
    # World coordinates of the bottom-left corner of the map
    origin_x_coord = center_coords[0] - (map_width_pixels / 2.0) * res
    origin_y_coord = center_coords[1] - (map_height_pixels / 2.0) * res
    print(f"Map Origin (world coords of pixel (0,0) bottom-left): ({origin_x_coord}, {origin_y_coord})")
    return data, center_coords, res, np.array([origin_x_coord, origin_y_coord])


def load_roads(filename, visualize=True):
    seg_map = np.load(filename)
    seg_ids = seg_map["data"]
    # Ensure resolution is a scalar float
    resolution_val = float(seg_map["resolution"][0]) if isinstance(seg_map["resolution"], (np.ndarray, list)) else float(seg_map["resolution"])


    # Find which of the global SEGID_COLORS match any color in the ROAD list
    road_color_indices = np.where(
        np.any( # Check if any color in ROAD matches
            np.all(SEGID_COLORS[:, None] == np.array(ROAD), axis=2), # Compare each SEGID_COLOR to all ROAD colors
            axis=1 # For each SEGID_COLOR, is there a match in ROAD?
        )
    )[0]

    print("Road Segment IDs (indices into SEGID_COLORS):", road_color_indices)

    # Create a binary mask where 1 means road, 0 otherwise
    # seg_ids contains the direct indices that should match road_color_indices
    roads_mask = np.isin(seg_ids, road_color_indices).astype(np.uint8)
    
    if visualize:
        plt.imshow(roads_mask, cmap='gray')
        plt.title("Roads in City Map")
        plt.gca().invert_yaxis() # Invert Y to show origin at bottom-left if desired, or use origin='lower' in imshow
        plt.show()

    return roads_mask, resolution_val, seg_map["center"]

# Removed get_belief_map as its logic is integrated into plot_belief_on_city_map for clarity

def plot_belief_on_city_map(segmap_file, mission_description_file, obstacle_map_file, visualize=True):
    """
    Plot the belief map polygons on the city map.
    """
    roads_mask, map_resolution, map_world_center = load_roads(segmap_file, visualize=False)
    
    mission, eoi_list, constraints = load_mission(mission_description_file) # eoi_list is a list of entities

    map_height_pixels, map_width_pixels = roads_mask.shape # roads_mask is (height, width)

    # Calculate world coordinates of the map's origin (bottom-left corner for pixel (0,0))
    # This is the 'center' argument for coordinate_to_pixel
    map_origin_world_x = map_world_center[0] - (map_width_pixels / 2.0) * map_resolution
    map_origin_world_y = map_world_center[1] - (map_height_pixels / 2.0) * map_resolution
    world_coords_of_pixel_0_0 = np.array([map_origin_world_x, map_origin_world_y])

    # Create the partial function for transforming a single world coordinate pair to a pixel coordinate pair
    _world_to_pixel_transformer = partial(coordinate_to_pixel,
                                     resolution=map_resolution,
                                     center=world_coords_of_pixel_0_0)

    all_polygons_in_pixels = []
    # Iterate through entities of interest to extract their belief polygons
    for entity_data in eoi_list:
        entity_priors = entity_data.get("entity_priors", {})
        location_belief_maps = entity_priors.get("location_belief_map", []) # This is a list of belief maps
        
        for belief_map_item in location_belief_maps: # Each item is a dict with "polygon_vertices"
            world_vertices = belief_map_item.get("polygon_vertices") # Nx2 list/array of [x,y] world coords
            
            if world_vertices is not None and len(world_vertices) > 0:
                # Transform all vertices of the current polygon
                pixel_vertices_for_current_poly = [
                    _world_to_pixel_transformer(vertex) for vertex in world_vertices
                ]
                all_polygons_in_pixels.append(np.array(pixel_vertices_for_current_poly))

    # Plotting
    fig, ax = plt.subplots()
    # Using origin='lower' means pixel (0,0) is at the bottom-left of the displayed image.
    # This aligns with how world_coords_of_pixel_0_0 was calculated (as bottom-left).
    # And how coordinate_to_pixel calculates y_pixel (increases upwards from bottom edge).
    # No need for invert_yaxis() if origin='lower' is used and y_pixel is "row from bottom".
    ax.imshow(roads_mask, cmap='gray', origin='lower') 

    # Plot the collected polygons
    # Assuming the first polygon found is "start" and second is "end" for demonstration
    if len(all_polygons_in_pixels) > 0:
        start_poly_data = all_polygons_in_pixels[0]
        if start_poly_data.ndim == 2 and start_poly_data.shape[0] > 0: # Check if it's a valid Nx2 array
            start_patch = patches.Polygon(start_poly_data, closed=True, edgecolor='r', facecolor='r', linewidth=2, alpha=0.5)
            ax.add_patch(start_patch)
        else:
            print("Warning: Start polygon data is not valid for plotting.")

    if len(all_polygons_in_pixels) > 1:
        end_poly_data = all_polygons_in_pixels[1]
        if end_poly_data.ndim == 2 and end_poly_data.shape[0] > 0: # Check if it's a valid Nx2 array
            end_patch = patches.Polygon(end_poly_data, closed=True, edgecolor='g', facecolor='g', linewidth=2, alpha=0.5)
            ax.add_patch(end_patch)
        else:
            print("Warning: End polygon data is not valid for plotting.")
    
    # If you still prefer (0,0) at top-left for the image display and manual y-axis inversion:
    # ax.imshow(roads_mask, cmap='gray') # Default origin='upper'
    # plt.gca().invert_yaxis() # Then y-axis runs from H-1 at top to 0 at bottom

    ax.set_title("Belief Polygons on Roads Map")
    ax.set_xlabel("X Pixel Coordinate (Column)")
    ax.set_ylabel("Y Pixel Coordinate (Row, from bottom with origin='lower')")
    plt.show()


if __name__ == "__main__":
    # Ensure you have these files in the correct path or provide full paths
    segmap_file = "city_1000_1000_seg_segids.npz" 
    mission_description_file = "description.json" # Make sure this file exists and is correctly formatted
    obstacle_map_file = "city_1000_1000.npz" # This file is loaded by load_mission in some examples, ensure it's available if needed by your load_mission

    # Example: Create dummy files if they don't exist for testing basic execution
    # This is just for the script to run; replace with your actual files.
    try:
        np.load(segmap_file)
    except FileNotFoundError:
        print(f"Warning: '{segmap_file}' not found. Creating a dummy file.")
        # Dummy segmap: 100x100, resolution 1.0, center (50,50)
        # Data is 0s, meaning no specific road_ids will be found unless SEGID_COLORS[0] is a road color
        dummy_data = np.zeros((100,100), dtype=np.uint8)
        dummy_center = np.array([50.0, 50.0])
        dummy_resolution = np.array([1.0])
        np.savez_compressed(segmap_file, data=dummy_data, center=dummy_center, resolution=dummy_resolution)
        # Adjust ROAD and SEGID_COLORS if needed for dummy data to show roads, e.g.
        # ROAD = [[0,0,0]] if SEGID_COLORS[0] = [0,0,0] represents a road type for the dummy.

    # Assume load_mission can handle a missing or dummy description.json gracefully or create one.
    # For a robust test, you'd need a description.json that defines eoi with polygon_vertices.
    # Example dummy load_mission if you don't have one:
    # def load_mission(mission_file):
    #     print(f"Attempting to load mission: {mission_file}")
    #     # Dummy EOI data for testing polygon plotting
    #     eoi = [
    #         {
    #             "entity_priors": {
    #                 "location_belief_map": [
    #                     {"polygon_vertices": [[10, 10], [10, 20], [20, 20], [20, 10]]} # Start Polygon in world coords
    #                 ]
    #             }
    #         },
    #         {
    #             "entity_priors": {
    #                 "location_belief_map": [
    #                     {"polygon_vertices": [[70, 70], [70, 85], [85, 85], [85, 70]]} # End Polygon in world coords
    #                 ]
    #             }
    #         }
    #     ]
    #     return {}, eoi, {}
    # from load_mission import load_mission # Ensure this is the actual or dummy version

    plot_belief_on_city_map(segmap_file, mission_description_file, obstacle_map_file, visualize=True)