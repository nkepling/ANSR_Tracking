import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
import typing

def visualize_plan(pursuer_plan,evader_ground_truth: np.ndarray, predicted_trajectories: np.ndarray,koz:typing.Union[np.ndarray,None]=None):
    """Simple visualization function to plot the results."""
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.set_aspect('equal', adjustable='box')
    
    # Plot Evader Ground Truth
    ax.plot(evader_ground_truth[:, 0], evader_ground_truth[:, 1], 'r--', label='Evader Ground Truth', linewidth=2.5)
    ax.plot(evader_ground_truth[0, 0], evader_ground_truth[0, 1], 'ro', markersize=8, label='Evader Start')
    
    # Plot Evader Predicted Trajectories
    for i, trajectory in enumerate(predicted_trajectories):
        label = 'Predicted' if i == 0 else None
        ax.plot(trajectory[:, 0], trajectory[:, 1], 'k--', alpha=0.2, label=label)

    ax.plot(pursuer_plan[:,0],pursuer_plan[:,1],'bx',label="Pursuer",linewidth=2.5)
    ax.plot(pursuer_plan[0, 0], pursuer_plan[0, 1], 'bo', markersize=8, label='Pursuer Start')

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