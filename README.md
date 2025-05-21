# Simple Tracker

This is a simple receeding horizon controller for tracking a **non-evasive** target. We assume that when the evader in view we are recieving rollout trajectories from the PWM. 

## Problem Formulation

- We want to minimize the distance between the expected pursuer trajectory and pursuer trajectories over a look ahead horizon of ￼ time steps.
- When the target is in the field of view we will use PWM provided trajectories to do planning with receding horizon control with out of the box solvers. 
- We will formulate the problem as a non-linear program. 
- We assume discrete time trajectories but continuous state space. 
- We can (probably?) solve this online with an IPOPT solver or something similar. 

## Objective Function

$$
J(x) = \sum_{j=1}^M p_j(\sum_{k=1}^{T}w_k||x_k-y_k^{(j)}||^2)
$$


We will minimize expected squared distance between pursuer and evader over a horizon of $T$ time steps.


- $p_j$ Prob of jth evader trajectory. We can let $p_j=1$  if we assume all paths are equally likely. 
- $w_k$ This weighs the trajaectory at different time steps. We can either prioritize near or shortterm interception depending on how reliable we believe rollouts are.
- $x_k$ : Puruser position at timestep $k$
- $y_k$ : Evader position at timestep $k$
- $M$ : Total pursuer trajectories
- $T$ : Look ahead horizon where each timespte is of size $\Delta t$
  
## Constraints

For each timestep $k \in [1,T]$ we impose the following constaints on the pursuer;

- Fixed starting location (linear): $x_1$ is the current pursuer location
- Boundary constraints (linear) : $x_k$ has to stay within stay within zone and out of KOZ
- Motion constraints (quadratic) : Distance covered in one time step is bounded by max velocity, i.e $||x_{k+1} - x_k||^2 \le (v_{max}\Delta t)^2$ for $k = 1 ,..., N-1$.
