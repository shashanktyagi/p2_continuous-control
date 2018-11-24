[//]: # (Image References)

[image1]: https://user-images.githubusercontent.com/10624937/43851024-320ba930-9aff-11e8-8493-ee547c6af349.gif "Trained Agent"
[image2]: https://github.com/shashanktyagi/p2_continuous-control/blob/master/training_scores.png

# Project 1: Navigation

### Introduction

For this project, we worked with the [Reacher](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Learning-Environment-Examples.md#reacher) environment.

![Trained Agent][image1]

In this environment, a double-jointed arm can move to target locations. A reward of +0.1 is provided for each step that the agent's hand is in the goal location. Thus, the goal of your agent is to maintain its position at the target location for as many time steps as possible.

The observation space consists of 33 variables corresponding to position, rotation, velocity, and angular velocities of the arm. Each action is a vector with four numbers, corresponding to torque applicable to two joints. Every entry in the action vector should be a number between -1 and 1.

**Environment solved criterion:** We solve the single agent version of the environment. The task is episodic, and in order to solve the environment, the agent must get an average score of +30 over 100 consecutive episodes.


### 2. Learning Algorithm
We train the network using DDPG algorithm. For the Actor, we use a three layer MLP with 128 and 128 neurons respectively in hidden layers. The state vector is the 33 dimensional vector described in section 1. The output vector is of size 4.
We train the network using Adam optimize with an actor learning rate of 0.0002, critic learning rate of 0.0002 and batch size of 128. We use a discount factor of 0.99.

### 3. Results
The figure below shows average rewards per episode as the agent is being trained. The training is terminated when the average reward per episode reaches 30. We were able to solve the environement in 125 episodes.

![Rewards per episode][image2]

### 4. Future Work
  
### References
DDPG paper: https://arxiv.org/abs/1509.02971
