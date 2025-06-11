import math
import random
import matplotlib.pyplot as plt
import numpy as np
from utils import * # Assuming you have this for loading maps

show_animation = True


class RRT:
    """
    RRT class adapted for grid-based maps.
    """
    class Node:
        def __init__(self, x, y):
            self.x = x
            self.y = y
            self.path_x = []
            self.path_y = []
            self.parent = None

    def __init__(self,
                 start,
                 goal,
                 roads_map,  # ### MODIFICATION ###: Expects a numpy binary grid
                 rand_area,
                 expand_dis=5.0,
                 path_resolution=1.0,
                 goal_sample_rate=10, # Increased goal sample rate
                 max_iter=1000):

        self.start = self.Node(start[0], start[1])
        self.end = self.Node(goal[0], goal[1])
        
        self.min_rand_x = float(rand_area[0])
        self.max_rand_x = float(rand_area[1])
        self.min_rand_y = float(rand_area[2])
        self.max_rand_y = float(rand_area[3])

        self.expand_dis = expand_dis
        self.path_resolution = path_resolution
        self.goal_sample_rate = goal_sample_rate
        self.max_iter = max_iter
        
        # ### MODIFICATION ###: Store the roads_map and pre-calculate valid sample points
        self.roads_map = roads_map
        self.map_height, self.map_width = roads_map.shape
        # argwhere gives (row, col) which corresponds to (y, x)
        self.valid_road_points = np.argwhere(roads_map == 1)
        
        self.node_list = []

    def planning(self, animation=True):
        """
        RRT path planning.
        animation: bool for plotting the RRT tree growth
        """
        self.node_list = [self.start]
        for i in range(self.max_iter):
            rnd_node = self.get_random_node()
            nearest_ind = self.get_nearest_node_index(self.node_list, rnd_node)
            nearest_node = self.node_list[nearest_ind]

            new_node = self.steer(nearest_node, rnd_node, self.expand_dis)

            # ### MODIFICATION ###: Use the new grid-based validity check
            if self.is_path_valid(new_node):
                self.node_list.append(new_node)

            if animation and i % 10 == 0:
                self.draw_graph(rnd_node)

            # Check if goal is reachable
            if self.calc_dist_to_goal(self.node_list[-1].x, self.node_list[-1].y) <= self.expand_dis:
                final_node = self.steer(self.node_list[-1], self.end, self.expand_dis)
                if self.is_path_valid(final_node):
                    return self.generate_final_course(len(self.node_list) - 1)

        return None  # Cannot find path

    def steer(self, from_node, to_node, extend_length=float("inf")):
        new_node = self.Node(from_node.x, from_node.y)
        d, theta = self.calc_distance_and_angle(new_node, to_node)

        new_node.path_x = [new_node.x]
        new_node.path_y = [new_node.y]

        if extend_length > d:
            extend_length = d

        n_expand = math.floor(extend_length / self.path_resolution)

        for _ in range(n_expand):
            new_node.x += self.path_resolution * math.cos(theta)
            new_node.y += self.path_resolution * math.sin(theta)
            new_node.path_x.append(new_node.x)
            new_node.path_y.append(new_node.y)

        d, _ = self.calc_distance_and_angle(new_node, to_node)
        if d <= self.path_resolution:
            new_node.path_x.append(to_node.x)
            new_node.path_y.append(to_node.y)
            new_node.x = to_node.x
            new_node.y = to_node.y
        
        new_node.parent = from_node
        return new_node

    def generate_final_course(self, goal_ind):
        path = [[self.end.x, self.end.y]]
        node = self.node_list[goal_ind]
        while node.parent is not None:
            path.append([node.x, node.y])
            node = node.parent
        path.append([node.x, node.y])
        path.reverse()
        return path

    def calc_dist_to_goal(self, x, y):
        return math.hypot(x - self.end.x, y - self.end.y)

    def get_random_node(self):
        # ### MODIFICATION ###: Efficiently sample from valid road points
        if random.randint(0, 100) > self.goal_sample_rate:
            # Pick a random index from the list of valid points
            random_index = random.randint(0, len(self.valid_road_points) - 1)
            # Get the (y, x) coordinate from our pre-calculated list
            y, x = self.valid_road_points[random_index]
            rnd = self.Node(x, y)
        else:  # Biased sample to goal
            rnd = self.Node(self.end.x, self.end.y)
        return rnd

    def draw_graph(self, rnd=None):
        plt.clf()
        plt.gcf().canvas.mpl_connect(
            'key_release_event',
            lambda event: [exit(0) if event.key == 'escape' else None])
        
        # ### MODIFICATION ###: Draw the roads_map as the background
        plt.imshow(self.roads_map, cmap='gray', origin='lower')
        
        if rnd is not None:
            plt.plot(rnd.x, rnd.y, "^k")

        for node in self.node_list:
            if node.parent:
                plt.plot(node.path_x, node.path_y, "-g", alpha=0.5, linewidth=0.8)

        plt.plot(self.start.x, self.start.y, "og", markersize=10, label="Start")
        plt.plot(self.end.x, self.end.y, "or", markersize=10, label="Goal")
        
        plt.axis([self.min_rand_x, self.max_rand_x, self.min_rand_y, self.max_rand_y])
        plt.grid(False) # Grid is less useful on an image background
        plt.legend(loc='upper right')
        plt.pause(0.01)

    # ### MODIFICATION ###: Replaced check_collision with is_path_valid
    def is_path_valid(self, node):
        """
        Checks if all points along a node's path are on valid road pixels (value of 1).
        """
        if node is None:
            return False

        for i in range(len(node.path_x)):
            x, y = node.path_x[i], node.path_y[i]
            
            # Convert float coordinates to integer grid indices
            ix, iy = int(x), int(y)

            # Check if the point is within map bounds
            if not (0 <= ix < self.map_width and 0 <= iy < self.map_height):
                return False # Path is outside the map
            
            # Check if the pixel corresponds to a road
            if self.roads_map[iy, ix] == 0:
                return False # Path is on an obstacle/invalid area

        return True # Path is valid

    @staticmethod
    def get_nearest_node_index(node_list, rnd_node):
        dlist = [(node.x - rnd_node.x)**2 + (node.y - rnd_node.y)**2
                 for node in node_list]
        return dlist.index(min(dlist))

    @staticmethod
    def calc_distance_and_angle(from_node, to_node):
        dx = to_node.x - from_node.x
        dy = to_node.y - from_node.y
        d = math.hypot(dx, dy)
        theta = math.atan2(dy, dx)
        return d, theta


def main():
    print("start " + __file__)

    # --- Load map data (same as your previous script) ---
    segmap_file = "city_1000_1000_seg_segids.npz"
    roads, resolution = load_roads(segmap_file, visualize=False)
    roads = roads.T
    roads = np.rot90(roads)

    map_height, map_width = roads.shape

    # --- Set Initial parameters ---
    start_pos = [360.0, 470.0]
    end_pos = [470.0, 335.0]

    # The random sampling area should cover the entire map
    rand_area = [0, map_width, 0, map_height]

    # --- Instantiate and run RRT ---
    rrt = RRT(
        start=start_pos,
        goal=end_pos,
        roads_map=roads, # Pass the binary road map
        rand_area=rand_area,
        expand_dis=15.0, # Larger step size can speed up search on large maps
        path_resolution=1.0,
        max_iter=5000 # More iterations may be needed for complex maps
    )
    path = rrt.planning(animation=show_animation)

    if path is None:
        print("Cannot find path")
    else:
        print("found path!!")
        if show_animation:
            rrt.draw_graph()
            plt.plot([x for (x, y) in path], [y for (x, y) in path], '-r', linewidth=2, label="Final Path")
            plt.legend()
            plt.title("RRT Path on Roads Map")
            plt.show()

if __name__ == '__main__':
    main()