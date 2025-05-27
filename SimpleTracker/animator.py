# In file: animator.py
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
# --- MODIFICATION: Import Polygon patch ---
from matplotlib.patches import Polygon

def create_animation(data,filename=None):
    """
    Creates and saves a GIF animation of the pursuit-evasion simulation.
    
    Args:
        data (dict): A dictionary containing the simulation history data.
    """
    # Unpack the data
    pursuer_history = data["pursuer_history"]
    evader_history = data["evader_history"]
    pursuer_plans = data["pursuer_plans"]
    evader_predictions = data["evader_predictions"]
    # --- MODIFICATION: Unpack KOZ data, providing an empty list as a default ---
    keep_out_zones = data.get("keep_out_zones", [])
    
    num_frames = len(pursuer_plans)

    # --- 1. Set up the plot ---
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.set_aspect('equal')
    
    # Determine axis limits dynamically to fit all data
    all_points = np.vstack([pursuer_history, evader_history])
    ax.set_xlim(all_points[:, 0].min() - 5, all_points[:, 0].max() + 5)
    ax.set_ylim(all_points[:, 1].min() - 5, all_points[:, 1].max() + 5)
    ax.grid(True)
    ax.set_title("Pursuit-Evasion Simulation")

    # --- MODIFICATION: Draw the static Keep-Out Zone patches ---
    if keep_out_zones:
        for i, vertices in enumerate(keep_out_zones):
            # The label will only be added for the first KOZ to avoid clutter in the legend
            label = 'Keep-Out Zone' if i == 0 else None
            koz_patch = Polygon(vertices, closed=True, color='red', alpha=0.4, label=label)
            ax.add_patch(koz_patch)
    
    # --- 2. Initialize Plot Elements ---
    # These are the plot objects that will be updated in each frame
    
    # History paths
    pursuer_path_line, = ax.plot([], [], 'b-', lw=2, label='Pursuer Path')
    evader_path_line, = ax.plot([], [], 'g-', lw=2, label='Evader Path')
    
    # Current positions (heads of the paths)
    pursuer_head, = ax.plot([], [], 'bo', markersize=10)
    evader_head, = ax.plot([], [], 'go', markersize=10)
    
    # Pursuer's current plan
    pursuer_plan_line, = ax.plot([], [], 'b--', lw=1, alpha=0.7, label='Pursuer Plan')
    
    # Evader's predicted trajectories
    num_predictions = evader_predictions[0].shape[0]
    evader_pred_lines = [ax.plot([], [], 'r--', lw=1, alpha=0.2)[0] for _ in range(num_predictions)]
    
    # Time step text
    time_text = ax.text(0.02, 0.95, '', transform=ax.transAxes)
    
    ax.legend(loc='upper left')

    # --- 3. Define the Animation Update Function ---
    # This function is called for each frame of the animation
    def update(frame):
        # Update pursuer history
        pursuer_path_line.set_data(pursuer_history[:frame+1, 0], pursuer_history[:frame+1, 1])
        pursuer_head.set_data([pursuer_history[frame, 0]], [pursuer_history[frame, 1]])
        
        # Update evader history
        evader_path_line.set_data(evader_history[:frame+1, 0], evader_history[:frame+1, 1])
        evader_head.set_data([evader_history[frame, 0]], [evader_history[frame, 1]])
        
        # Update current plan and predictions for the current frame
        current_plan = pursuer_plans[frame]
        pursuer_plan_line.set_data(current_plan[:, 0], current_plan[:, 1])
        
        current_predictions = evader_predictions[frame]
        for i, line in enumerate(evader_pred_lines):
            line.set_data(current_predictions[i, :, 0], current_predictions[i, :, 1])
            
        # Update time text
        time_text.set_text(f'Step: {frame+1}/{num_frames}')
        
        # Return all plot elements that have been modified
        # The KOZ patches are not returned because they are static and do not need to be redrawn.
        return (pursuer_path_line, pursuer_head, evader_path_line, evader_head, 
                pursuer_plan_line, time_text) + tuple(evader_pred_lines)


    # --- 4. Create the Animation ---
    anim = FuncAnimation(fig, update, frames=num_frames,
                         interval=50, blit=True)

    # --- 5. Save or Show the Animation ---
    try:
        filename_to_save = filename if filename is not None else 'pursuit_animation.gif'
        print(f"Saving animation to {filename_to_save}... (This may take a moment)")
        anim.save(filename_to_save, writer='pillow', fps=20)
        print("Animation saved successfully.")
    except Exception as e:
        print(f"Error saving animation: {e}")
        print("Displaying animation instead.")
        plt.show()