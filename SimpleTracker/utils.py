import numpy as np
import matplotlib.pyplot as plt
import json
import matplotlib.patches as patches
from functools import partial
from matplotlib import transforms
from matplotlib.path import Path

def discretize_obstacle_map(obstacle_map: np.ndarray, dim: tuple[int, int], obs_thresh: float) -> np.ndarray:
    """
    Downsamples a high-resolution obstacle map to a lower resolution.

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
        np.ndarray: The new, lower-resolution obstacle map.
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
    return np.where(block_means > obs_thresh, 1, 0)


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


def get_evader_path_from_file(file_name,obstacle_map,roads,resolution,center):
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




if __name__ == "__main__":
    segmap_file = "city_1000_1000_seg_segids.npz"
    mission_description_file = "description.json"
    obstacle_map_file = "city_1000_1000.npz"    


    """Plot roads, obstacles, beleif region, 
    """
    # plot mission details on 1000x1000gridworld 
    plot_belief_on_city_map(segmap_file, mission_description_file, obstacle_map_file, visualize=True)


    #load binary numpy array where 1s are roads 
    roads, resolution = load_roads(segmap_file, visualize=False)
    roads = np.rot90(roads.T)


    #load binary numpy array where 1s are obstacles, depth is obstacle height
    obstacle_map, resolution, (origin_y,origin_x) = load_obstacle_map(obstacle_map_file, depth=10)
    obstacle_map = np.rot90(obstacle_map.T)

    dim = (100,100)
    obs_thresh = 0.5

    #down sample binary numpy array to lower resolutoin grid world. dim is gridworld size
    low_res_map = discretize_obstacle_map(obstacle_map,dim,obs_thresh)
    plot_map(low_res_map)




    
 



    






        

    