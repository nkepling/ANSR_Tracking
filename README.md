# Simple Tracker

This is a simple receeding horizon controller for tracking a **non-evasive** target. We assume that when the evader in view we are recieving rollout trajectories from the PWM. 

## psd

## Problem Formulation

- We want to minimize the distance between the expected pursuer trajectory and pursuer trajectories over a look ahead horizon of ￼ time steps.
- When the target is in the field of view we will use PWM provided trajectories to do planning with receding horizon control with out of the box solvers. 
- We will formulate the problem as a non-linear program. 
- We assume discrete time trajectories but continuous state space. 
- We can (probably?) solve this online with an IPOPT solver or something similar. 

## Objective Function

$$
J(x) = \sum_{j=1}^M p_j(\sum_{k=1}^{T}w^k||x_k-y_k^{(j)}||^2) + \lambda_{penalty} \sum_{k=1}^{T}(\max0,d_{min}^2- ||x_k - \bar y_k ||^2)
$$


We will minimize expected squared distance between pursuer and evader over a horizon of $T$ time steps. I've added a penatly term to discourage getting too close the the evader but it is not a hard constraint.


- $p_j$ Prob of jth evader trajectory. We can let $p_j=1$  if we assume all paths are equally likely. 
- $w_k$ This weighs the trajaectory at different time steps. We can either prioritize near or shortterm interception depending on how reliable we believe rollouts are.
- $x_k$ : Puruser position at timestep $k$
- $y_k$ : Evader position at timestep $k$
- $M$ : Total pursuer trajectories
- $T$ : Look ahead horizon where each timespte is of size $\Delta t$
- $d$ : Min allowable distance between evader and pursuer. 

## Constraints

For each timestep $k \in [1,T]$ we impose the following constaints on the pursuer;

- Fixed starting location (linear): $x_1$ is the current pursuer location
$$
x_1 = x_{current}
$$
- Boundary constraints (linear) : $x_k$ has to stay within stay within zone and out of KOZ
- Motion constraints (quadratic) : Distance covered in one time step is bounded by max velocity, i.e $||x_{k+1} - x_k||^2 \le (v_{max}\Delta t)^2$ for $k = 1 ,..., N-1$.


Of course. Here is a file structure for your project, formatted as a Markdown code block. You can copy and paste this directly into your `README.md` file.

This structure includes brief descriptions of each file's role based on the code you've provided.


## File Structure


```
SimpleTracker/
│
├── README.md        Project explanation, setup, and usage instructions
│
├── simulate.py    # Main script to run the pursuit-evasion simulation and save the results
├── animator.py    # Creates a GIF animation from the simulation history data
│
├── track.py     # Trajectory optimization solver using SLSQP from SciPy
│
├── dummy_pwm.py    # Contains PWM code: Evader kinematic model and functions to generate predicted trajectories
```

