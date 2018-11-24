[//]: # (Image References)

[image1]: https://user-images.githubusercontent.com/10624937/42135619-d90f2f28-7d12-11e8-8823-82b970a54d7e.gif "Trained Agent"
[image2]: https://github.com/shashanktyagi/p1_navigation/blob/master/training_scores.png

# Project 1: Navigation

### 1. Introduction

For this project, we train an agent to navigate (and collect bananas!) in a large, square world.  

![Trained Agent][image1]

A reward of +1 is provided for collecting a yellow banana, and a reward of -1 is provided for collecting a blue banana.  Thus, the goal of your agent is to collect as many yellow bananas as possible while avoiding blue bananas.  

The state space has 37 dimensions and contains the agent's velocity, along with ray-based perception of objects around agent's forward direction.  Given this information, the agent has to learn how to best select actions.  Four discrete actions are available, corresponding to:
- **`0`** - move forward.
- **`1`** - move backward.
- **`2`** - turn left.
- **`3`** - turn right.

**Environment solved criterion:** The task is episodic, and in order to solve the environment, the agent must get an average score of +13 over 100 consecutive episodes.

### 2. Learning Algorithm
We train the network using deep Q-Learning algorithm. For the Q-Network, we use a four layer MLP with 64, 128, 128 and 128 neurons respectively in hidden layers. The state vector is the 37 dimensional vector described in section 1.
We train the network using Adam optimize with a learning rate of 0.0005 and batch size of 64. We use a discount factor of 0.99.

### 3. Results
The figure below shows average rewards per episode as the agent is being trained. The training is terminated when the average reward per episode reaches 13. We were able to solve the environement in 401 episodes.

![Rewards per episode][image2]

### 4. Future Work
  Since the discovery of Q-Learing algorithm, there have been a lot of improvement in this field viz. double DQN, Duelling DQN and prioritized experience replay. The future work involves the implementation of these algorithms and compare the results. The algorithm have shown better results on numerous problems. Thus, we expect them to train the agent in less number of episodes.
  We would also like to train the network based on the raw image as input state vector. This would require a Convolutional Neural Network and the Q-Network.
### References
DQN paper: https://storage.googleapis.com/deepmind-media/dqn/DQNNaturePaper.pdf
