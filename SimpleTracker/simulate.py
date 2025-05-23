import numpy as np
import track  # Contains the SLSQP solver logic
import vis # Contains the visualization functions
from dummy_pwm import Evader, forward, get_straight_away_trajectories
import animator

def run_simulation():
    """
    Runs a multi-step pursuit-evasion simulation.
    """
    # --- 1. Simulation Setup ---
    T = 10    # Planning horizon (how far ahead the pursuer plans)
    dt = 0.1    # Time step duration
    M = 10      # Number of evader trajectories to predict
    sim_steps = 200 # Total number of steps in the simulation

    TURN_STEP = 5

    # --- 2. Initialize Agent States ---
    # These variables will be updated in each step of the loop
    pursuer_current_pos = np.array([-5.0, 0.0])
    evader_current_state = Evader(x=0.0, y=0.0, theta=np.deg2rad(0), v=1.0)

    # --- 3. Log History for Final Visualization ---

    pursuer_history = [pursuer_current_pos]
    evader_history = [evader_current_state.pos]
    pursuer_plans_history = []
    evader_predictions_history = []


    print("Starting simulation...")
    # --- 4. Main Simulation Loop ---
    for i in range(sim_steps):
        print(f"Running step {i+1}/{sim_steps}...")

        # --- a. PREDICT ---
        # Generate M possible future trajectories for the evader from its CURRENT state
        predicted_evader_trajectories = get_straight_away_trajectories(
            initial_evader_state=evader_current_state,
            num_trajectories=M,
            num_time_steps=T,
            delta_t=dt,
            speed_variation_std=1.0,
            heading_variation_std=np.deg2rad(15.0)
        )

        # --- b. PLAN ---
        # Solve for the optimal pursuer plan starting from its CURRENT position
        # A large box is used as the "Safe Walk Zone" for this example
        swx_verts = np.array([[-30, -30], [30, -30], [30, 30], [-30, 30], [-30, -30]])
        pursuer_plan = track.solve(
            evader_trajectories=predicted_evader_trajectories,
            T=T,
            dt=dt,
            start_pos=pursuer_current_pos,
            max_velo=1.0
        )

        if pursuer_plan is None:
            print("Solver failed to find a plan. Ending simulation.")
            break
            
        # Optional: Visualize the plan at certain steps
        # if i % 1 == 0:
        #     gt_trajectory = []
        #     current_gt_state = evader_current_state
        #     for _ in range(T):
        #         gt_trajectory.append(current_gt_state.pos)
        #         current_gt_state = forward(current_gt_state, dt)
        #     ground_truth_np = np.array(gt_trajectory)
        #     vis.visualize_plan(pursuer_plan,ground_truth_np,predicted_evader_trajectories)

        pursuer_plans_history.append(pursuer_plan)
        evader_predictions_history.append(predicted_evader_trajectories)

        if i % TURN_STEP == 0:
            print(f"--- EVADER MAKING A 90-DEGREE TURN AT STEP {i} ---")
            # Create a new state object with the updated heading
            evader_current_state = Evader(
                x=evader_current_state.x,
                y=evader_current_state.y,
                theta=evader_current_state.theta + np.deg2rad(3), # Add 90 degrees
                v=evader_current_state.v
            )
        

        # --- d. ACT & UPDATE STATE ---
        pursuer_current_pos = pursuer_plan[1, :]
        evader_current_state = forward(evader_current_state, dt)

        # Append the new ground-truth positions to the history
        pursuer_history.append(pursuer_current_pos)
        evader_history.append(evader_current_state.pos)

    print("Simulation finished.")
    # --- 5. Final Visualization ---
    # Convert lists to numpy arrays for easier plotting
        # --- 5. Return all collected data ---
    return {
        "pursuer_history": np.array(pursuer_history),
        "evader_history": np.array(evader_history),
        "pursuer_plans": pursuer_plans_history,
        "evader_predictions": evader_predictions_history
    }

    
    


if __name__ == "__main__":

       # Run the simulation to get the data
    simulation_data = run_simulation()

    # Pass the data to the animator
    # This will create and save the animation as a GIF
    animator.create_animation(simulation_data,"cricle.gif")