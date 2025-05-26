import numpy as np
from dataclasses import dataclass
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
import math
import typing

# --- Corrected Evader Class and Kinematic Functions ---

@dataclass
class Evader:
    """Represents the state of the evading agent."""
    x: float
    y: float
    theta: float  # Heading in radians
    v: float      # Speed in units/sec

    @property
    def vec(self) -> np.ndarray:
        """Returns the full state vector [x, y, theta, v]."""
        return np.array([self.x, self.y, self.theta, self.v])

    @property
    def pos(self) -> np.ndarray:
        """Returns the position vector [x, y]."""
        return np.array([self.x, self.y])

def forward(current_state: Evader, delta_t: float) -> Evader:
    """
    FIXED: Move evader forward with correct simple Euler integration.
    This model assumes constant velocity and heading over the small time step delta_t.
    """
    # Calculate the change in position based on current heading and speed
    dx = current_state.v * np.cos(current_state.theta) * delta_t
    dy = current_state.v * np.sin(current_state.theta) * delta_t
    
    # Return a new Evader state object with the updated position
    return Evader(
        x=current_state.x + dx,
        y=current_state.y + dy,
        theta=current_state.theta, # Heading remains constant for a straight-away model
        v=current_state.v          # Speed remains constant for a straight-away model
    )

def get_straight_away_trajectories(initial_evader_state: Evader,
                                   num_trajectories: int,
                                   num_time_steps: int,
                                   delta_t: float,
                                   speed_variation_std: float = 0.1,
                                   heading_variation_std: float = 0.01) -> np.ndarray:
    """
    FIXED: Generates M (num_trajectories) plausible straight-line trajectories.
    Each trajectory is a sequence of T (num_time_steps) positions.
    Returns a 3D numpy array of shape (M, T, 2).
    """
    all_trajectories_list = []

    for _ in range(num_trajectories):
        # Create a perturbed initial state for this specific trajectory
        # This simulates uncertainty in the evader's current speed and heading
        perturbed_speed = max(0, initial_evader_state.v + np.random.normal(0, speed_variation_std))
        perturbed_heading = initial_evader_state.theta + np.random.normal(0, heading_variation_std)
        
        # This is the starting state for this one simulated future
        current_state = Evader(x=initial_evader_state.x,
                               y=initial_evader_state.y,
                               theta=perturbed_heading,
                               v=perturbed_speed)
        
        trajectory_points_list = []
        for _ in range(num_time_steps):
            # Add the current (x, y) position to our list of points
            trajectory_points_list.append(current_state.pos)
            # Propagate the state forward to the next time step
            current_state = forward(current_state, delta_t)
            
        all_trajectories_list.append(np.array(trajectory_points_list))
        
    return np.array(all_trajectories_list)

# --- Visualization and Demonstration (using the function from your previous code) ---

def visualize(ground_truth: np.ndarray, predicted_trajectories: np.ndarray,koz:typing.Union[np.ndarray,None]=None):
    """Simple visualization function to plot the results."""
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.set_aspect('equal', adjustable='box')
    
    # Plot Ground Truth
    ax.plot(ground_truth[:, 0], ground_truth[:, 1], 'r-', label='Ground Truth', linewidth=2.5)
    ax.plot(ground_truth[0, 0], ground_truth[0, 1], 'ro', markersize=8, label='Start')
    
    # Plot Predicted Trajectories
    for i, trajectory in enumerate(predicted_trajectories):
        label = 'Predicted' if i == 0 else None
        ax.plot(trajectory[:, 0], trajectory[:, 1], 'k--', alpha=0.1, label=label)

    if koz is not None:
        poly = Polygon(koz, closed=True, color='orange', alpha=0.4, label='KOZ')
        ax.add_patch(poly)

    ax.set_xlim(-20,20)
    ax.set_ylim(-20,20)

    ax.set_xlabel('X Position')
    ax.set_ylabel('Y Position')
    ax.set_title('Straight-Away Trajectory Predictions')
    ax.legend()
    ax.grid(True)
    plt.show()

if __name__ == '__main__':
    # --- Simulation Parameters ---
    initial_state = Evader(x=0.0, y=0.0, theta=np.deg2rad(0), v=10.0)
    
    M = 20  # Number of predicted trajectories
    T = 10  # Number of time steps in each trajectory
    dt = 0.1 # Time delta
r
    # --- Generate Predictions ---
    predicted_trajectories = get_straight_away_trajectories(
        initial_evader_state=initial_state,
        num_trajectories=M,
        num_time_steps=T,
        delta_t=dt,
        speed_variation_std=2.0,       # Standard deviation for speed
        heading_variation_std=np.deg2rad(5.0) # Standard deviation for heading
    )

    # --- Generate a single Ground Truth trajectory for comparison (no noise) ---
    gt_trajectory = []
    current_gt_state = initial_state
    for _ in range(T):
        gt_trajectory.append(current_gt_state.pos)
        current_gt_state = forward(current_gt_state, dt)
    ground_truth_np = np.array(gt_trajectory)

    # --- Visualize the results ---

    koz_verts = np.array([(1,1),(2,1),(2,-1),(1,-1),(1,1)])
    visualize(ground_truth=ground_truth_np, predicted_trajectories=predicted_trajectories,koz=koz_verts)