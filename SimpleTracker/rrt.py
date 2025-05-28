import math
import random
import matplotlib.pyplot as plt
import numpy as np


""" This code was derived from this repo: https://github.com/AtsushiSakai/PythonRobotics/blob/master/PathPlanning/RRT/rrt.py

CITE: PythonRobotics: a Python code collection of robotics algorithms
"""

show_animation = True

# --- Geometric Helper Functions for Polygon Collision Checking ---
def on_segment(p, q, r):
    """Given three collinear points p, q, r, check if point q lies on segment 'pr'."""
    return (q[0] <= max(p[0], r[0]) and q[0] >= min(p[0], r[0]) and
            q[1] <= max(p[1], r[1]) and q[1] >= min(p[1], r[1]))

def orientation(p, q, r):
    """Find orientation of ordered triplet (p, q, r).
    Returns: 0 (collinear), 1 (clockwise), 2 (counterclockwise)
    """
    val = (q[1] - p[1]) * (r[0] - q[0]) - (q[0] - p[0]) * (r[1] - q[1])
    if val == 0: return 0
    return 1 if val > 0 else 2

def segments_intersect(p1, q1, p2, q2):
    """Check if line segment 'p1q1' and 'p2q2' intersect."""
    o1 = orientation(p1, q1, p2)
    o2 = orientation(p1, q1, q2)
    o3 = orientation(p2, q2, p1)
    o4 = orientation(p2, q2, q1)

    if o1 != o2 and o3 != o4:
        return True
    if o1 == 0 and on_segment(p1, p2, q1): return True
    if o2 == 0 and on_segment(p1, q2, q1): return True
    if o3 == 0 and on_segment(p2, p1, q2): return True
    if o4 == 0 and on_segment(p2, q1, q2): return True
    return False

def is_point_in_polygon(point, polygon_vertices):
    """Checks if a point is inside a polygon using the Ray Casting algorithm."""
    n = len(polygon_vertices)
    if n < 3: return False
    
    point = np.array(point) # Ensure point is a numpy array
    far_x = point[0] + 1e7 # A large number, ensuring the ray is long enough
    extreme_point = np.array([far_x, point[1]])
    intersections = 0
    
    for i in range(n):
        p1 = np.array(polygon_vertices[i])
        p2 = np.array(polygon_vertices[(i + 1) % n])
        if segments_intersect(p1, p2, point, extreme_point):
            if orientation(p1, point, p2) == 0: # Collinear
                return on_segment(p1, point, p2) # On boundary is considered inside for safety
            intersections += 1
    return intersections % 2 == 1

def dist_point_to_segment(p, a, b):
    """Calculates the shortest distance from point p to line segment ab."""
    p = np.array(p); a = np.array(a); b = np.array(b)
    if np.array_equal(a,b): # Segment is a point
        return np.linalg.norm(p - a)

    line_vec = b - a
    point_vec = p - a
    line_len_sq = np.dot(line_vec, line_vec)
    
    t = np.dot(point_vec, line_vec) / line_len_sq
    
    if t < 0.0:
        closest_point = a
    elif t > 1.0:
        closest_point = b
    else:
        closest_point = a + t * line_vec
        
    return np.linalg.norm(p - closest_point)

class RRT:
    class Node:
        def __init__(self, x, y):
            self.x = x
            self.y = y
            self.path_x = []
            self.path_y = []
            self.parent = None

    class AreaBounds:
        def __init__(self, area):
            self.xmin = float(area[0])
            self.xmax = float(area[1])
            self.ymin = float(area[2])
            self.ymax = float(area[3])

    def __init__(self,
                 start,
                 goal,
                 obstacle_list, # MODIFICATION: Now expects list of polygon vertices
                 rand_area,
                 expand_dis=3.0,
                 path_resolution=0.5,
                 goal_sample_rate=5,
                 max_iter=500,
                 play_area=None,
                 robot_radius=0.0):
        self.start = self.Node(start[0], start[1])
        self.end = self.Node(goal[0], goal[1])
        self.min_rand_x = float(rand_area[0]) # MODIFICATION: Assume rand_area can be [min_x, max_x, min_y, max_y]
        self.max_rand_x = float(rand_area[1])
        self.min_rand_y = float(rand_area[2])
        self.max_rand_y = float(rand_area[3])

        if play_area is not None:
            self.play_area = self.AreaBounds(play_area)
        else:
            self.play_area = None
        self.expand_dis = expand_dis
        self.path_resolution = path_resolution
        self.goal_sample_rate = goal_sample_rate
        self.max_iter = max_iter
        self.obstacle_list = obstacle_list # List of polygons (each polygon is a list of [x,y] vertices)
        self.node_list = []
        self.robot_radius = robot_radius

    def planning(self, animation=True):
        self.node_list = [self.start]
        for i in range(self.max_iter):
            rnd_node = self.get_random_node()
            nearest_ind = self.get_nearest_node_index(self.node_list, rnd_node)
            nearest_node = self.node_list[nearest_ind]

            new_node = self.steer(nearest_node, rnd_node, self.expand_dis)

            if self.check_if_outside_play_area(new_node, self.play_area) and \
               self.check_collision(new_node, self.obstacle_list, self.robot_radius):
                self.node_list.append(new_node)

            if animation and i % 5 == 0:
                self.draw_graph(rnd_node)

            if self.calc_dist_to_goal(self.node_list[-1].x, self.node_list[-1].y) <= self.expand_dis:
                final_node = self.steer(self.node_list[-1], self.end, self.expand_dis)
                if self.check_collision(final_node, self.obstacle_list, self.robot_radius):
                    return self.generate_final_course(len(self.node_list) - 1)

            if animation and i % 5 == 0: # Draw more frequently for better viz
                 self.draw_graph(rnd_node)

        return None

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
        if d <= self.path_resolution: # Ensure goal is reached if within one step
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
        path.reverse() # Path should be from start to goal
        return path

    def calc_dist_to_goal(self, x, y):
        return math.hypot(x - self.end.x, y - self.end.y)

    def get_random_node(self):
        if random.randint(0, 100) > self.goal_sample_rate:
            rnd = self.Node(
                random.uniform(self.min_rand_x, self.max_rand_x), # Use x/y specific random areas
                random.uniform(self.min_rand_y, self.max_rand_y))
        else:
            rnd = self.Node(self.end.x, self.end.y)
        return rnd

    def draw_graph(self, rnd=None):
        plt.clf()
        plt.gcf().canvas.mpl_connect(
            'key_release_event',
            lambda event: [exit(0) if event.key == 'escape' else None])
        
        if rnd is not None:
            plt.plot(rnd.x, rnd.y, "^k")
            if self.robot_radius > 0.0: # Visualize robot radius around sample point
                self.plot_robot_clearance(rnd.x, rnd.y, self.robot_radius, '--r')


        for node in self.node_list:
            if node.parent:
                plt.plot(node.path_x, node.path_y, "-g", alpha=0.5)

        # --- MODIFICATION: Draw polygonal obstacles ---
        for polygon_vertices in self.obstacle_list:
            poly = plt.Polygon(polygon_vertices, closed=True, facecolor='gray', edgecolor='black', alpha=0.7)
            plt.gca().add_patch(poly)

        if self.play_area is not None:
            plt.plot([self.play_area.xmin, self.play_area.xmax, self.play_area.xmax, 
                      self.play_area.xmin, self.play_area.xmin],
                     [self.play_area.ymin, self.play_area.ymin, self.play_area.ymax, 
                      self.play_area.ymax, self.play_area.ymin], "-k")

        plt.plot(self.start.x, self.start.y, "og", markersize=10, label="Start") # Use 'og' for green circle
        plt.plot(self.end.x, self.end.y, "or", markersize=10, label="Goal")   # Use 'or' for red circle
        
        plt.axis("equal")
        plt.axis([self.min_rand_x, self.max_rand_x, self.min_rand_y, self.max_rand_y]) # Use x/y specific random areas
        plt.grid(True)
        plt.legend(loc='upper right')
        plt.pause(0.01)

    @staticmethod
    def plot_robot_clearance(x, y, radius, color="-b"): # For visualizing robot radius effect
        deg = list(range(0, 360, 15))
        deg.append(0)
        xl = [x + radius * math.cos(np.deg2rad(d)) for d in deg]
        yl = [y + radius * math.sin(np.deg2rad(d)) for d in deg]
        plt.plot(xl, yl, color, alpha=0.3)


    @staticmethod
    def get_nearest_node_index(node_list, rnd_node):
        dlist = [(node.x - rnd_node.x)**2 + (node.y - rnd_node.y)**2 for node in node_list]
        return dlist.index(min(dlist))

    @staticmethod
    def check_if_outside_play_area(node, play_area):
        if play_area is None: return True
        if node.x < play_area.xmin or node.x > play_area.xmax or \
           node.y < play_area.ymin or node.y > play_area.ymax:
            return False
        return True

    @staticmethod
    def check_collision(node, obstacle_list, robot_radius): # MODIFIED
        if node is None: return False

        for path_idx in range(len(node.path_x)):
            px, py = node.path_x[path_idx], node.path_y[path_idx]
            point_to_check = np.array([px, py])

            for polygon_vertices in obstacle_list:
                # 1. Check if the robot's center is inside the polygon
                if is_point_in_polygon(point_to_check, polygon_vertices):
                    return False  # Collision

                # 2. If robot_radius > 0, check distance from point to each polygon edge
                if robot_radius > 0.0:
                    n_verts = len(polygon_vertices)
                    for i in range(n_verts):
                        v1 = np.array(polygon_vertices[i])
                        v2 = np.array(polygon_vertices[(i + 1) % n_verts])
                        if dist_point_to_segment(point_to_check, v1, v2) <= robot_radius:
                            return False # Collision: robot body hits an edge
        return True

    @staticmethod
    def calc_distance_and_angle(from_node, to_node):
        dx = to_node.x - from_node.x
        dy = to_node.y - from_node.y
        d = math.hypot(dx, dy)
        theta = math.atan2(dy, dx)
        return d, theta

def main(gx=18.0, gy=0.0): # Target the end of the corridor
    print("start " + __file__)

    # --- MODIFICATION: Define polygonal obstacles (your KOZs) ---
    koz1_verts = np.array([(4,2.5),(-15,2.5),(-15,1.5),(5,1.5),(5,10),(4,10)])
    koz2_verts = np.array([(5,-2.5),(-15,-2.5),(-15,-1.5),(4,-1.5),(4,-10),(5,-10),(5,-1.5)])
    koz3_verts = np.array([(8,10),(8,-10),(9,-10),(9,10)]) # The end barrier
    
    # Combine all KOZs into the obstacle_list
    # Each element is a list/array of vertices for one polygon
    obstacleList = [koz1_verts, koz2_verts, koz3_verts]

    # Set Initial parameters
    # Define rand_area as [min_x, max_x, min_y, max_y]
    # Ensure this area covers your start, goal, and obstacles
    rand_area = [-20, 20, -12, 12] 

    rrt = RRT(
        start=[-18, 0], # Start position from your simulate_with_jax.py
        goal=[gx, gy],  # Goal (e.g., end of the corridor before koz3)
        rand_area=rand_area,
        obstacle_list=obstacleList,
        expand_dis=2.0,       # Reduced expand distance for tighter spaces
        path_resolution=0.2,  # Finer path resolution
        max_iter=2000,        # Increased iterations for complex environments
        robot_radius=0.5      # Example robot radius
    )
    path = rrt.planning(animation=show_animation)

    if path is None:
        print("Cannot find path")
    else:
        print("found path!!")
        if show_animation:
            rrt.draw_graph() # Draw the final tree
            plt.plot([x for (x, y) in path], [y for (x, y) in path], '-r', linewidth=2)
            plt.title("RRT Path with Polygonal Obstacles")
            plt.show()

if __name__ == '__main__':
    main()