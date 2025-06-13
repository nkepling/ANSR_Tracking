import numpy as np
try:
    # OMPL is a C++ library, so the Python bindings are in a specific format
    from ompl import base as ob
    from ompl import geometric as og
except ImportError:
    raise ImportError("OMPL is not installed or not found. Please install it.")

class OMPLGridPlanner:
    """
    A wrapper class to handle 2D motion planning on a grid map using OMPL.
    This class sets up the OMPL environment for a grid-based world where
    the robot is treated as a point.
    """
    def __init__(self, grid_map):
        """
        Initializes the planner with a given grid map.

        Args:
            grid_map (np.array): A 2D numpy array representing the world.
                                 0 means free space, non-zero means obstacle.
        """
        self.grid_map = grid_map
        self.map_height, self.map_width = grid_map.shape

        # 1. Define the State Space for our 2D world
        # We are planning for a point robot, so its state is just (x, y).
        self.space = ob.RealVectorStateSpace(2)

        # 2. Set the bounds for the State Space
        # The bounds should match the dimensions of our grid map.
        bounds = ob.RealVectorBounds(2)
        bounds.setLow(0, 0)
        bounds.setHigh(0, self.map_width)
        bounds.setLow(1, 0)
        bounds.setHigh(1, self.map_height)
        self.space.setBounds(bounds)

        # 3. Create a SimpleSetup object
        # This is a convenience class that encapsulates the planning problem setup.
        self.ss = og.SimpleSetup(self.space)

        # 4. Set the State Validity Checker
        # This is the most important part, where we tell OMPL how to
        # check if a state is valid (i.e., not in an obstacle).
        self.ss.setStateValidityChecker(ob.StateValidityCheckerFn(self.is_state_valid))

    def is_state_valid(self, state):
        """
        Checks if a given state (x, y) is valid.
        OMPL calls this function repeatedly to check potential states.
        """
        # OMPL state is a C++ object; we need to get its values.
        # We cast to int to use as grid indices.
        x = int(state[0])
        y = int(state[1])

        # Check if the state is within the map boundaries.
        if not (0 <= x < self.map_width and 0 <= y < self.map_height):
            return False

        # Check if the state is in an obstacle.
        # In our grid_map, non-zero values are obstacles.
        # Note the indexing: grid_map[row, col] -> grid_map[y, x].
        if self.grid_map[y, x] != 0:
            return False

        # If all checks pass, the state is valid.
        return True

    def plan(self, start_point, goal_point, planner_type="RRTstar", timeout=0.1):
        """
        Plans a path from a start to a goal point.

        Args:
            start_point (tuple): The (x, y) starting coordinates.
            goal_point (tuple): The (x, y) goal coordinates.
            planner_type (str): The OMPL planner to use (e.g., "RRT", "RRTstar").
            timeout (float): The maximum time in seconds allowed for planning.

        Returns:
            list: A list of (x, y) tuples representing the path, or None if no
                  path is found.
        """
        # Clear any previous planning data
        self.ss.clear()

        # Create OMPL State objects for start and goal
        start = ob.State(self.space)
        # IMPORTANT: OMPL expects float values, so we cast the inputs
        start[0] = float(start_point[0])
        start[1] = float(start_point[1])

        goal = ob.State(self.space)
        goal[0] = float(goal_point[0])
        goal[1] = float(goal_point[1])

        # Set the start and goal states in the problem definition
        self.ss.setStartAndGoalStates(start, goal)
        
        # Set the desired planner
        if planner_type == "RRT":
            planner = og.RRT(self.ss.getSpaceInformation())
        elif planner_type == "RRTstar":
            planner = og.RRTstar(self.ss.getSpaceInformation())
        elif planner_type == "BITstar":
            planner = og.BITstar(self.ss.getSpaceInformation())
        else:
            raise ValueError(f"Planner type '{planner_type}' not recognized.")
        
        self.ss.setPlanner(planner)

        # Attempt to solve the problem within the given time limit
        solved = self.ss.solve(timeout)

        if solved:
            print(f"Found solution in {self.ss.getLastPlanComputationTime():.4f} seconds!")
            # Simplify the solution to make it smoother
            self.ss.simplifySolution()
            # Get the path as a geometric path
            path = self.ss.getSolutionPath()
            # Convert the OMPL path to a simple list of Python tuples
            return [(s[0], s[1]) for s in path.getStates()]
        else:
            print("No solution found.")
            return None