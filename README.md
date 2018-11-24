[//]: # (Image References)

[image1]: https://user-images.githubusercontent.com/10624937/43851024-320ba930-9aff-11e8-8493-ee547c6af349.gif "Trained Agent"
[image2]: https://user-images.githubusercontent.com/10624937/43851646-d899bf20-9b00-11e8-858c-29b5c2c94ccc.png "Crawler"


# Project 2: Continuous Control

### Introduction

For this project, you will work with the [Reacher](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Learning-Environment-Examples.md#reacher) environment.

![Trained Agent][image1]

In this environment, a double-jointed arm can move to target locations. A reward of +0.1 is provided for each step that the agent's hand is in the goal location. Thus, the goal of your agent is to maintain its position at the target location for as many time steps as possible.

The observation space consists of 33 variables corresponding to position, rotation, velocity, and angular velocities of the arm. Each action is a vector with four numbers, corresponding to torque applicable to two joints. Every entry in the action vector should be a number between -1 and 1.

**Environment solved criterion:** We solve the single agent version of the environment. The task is episodic, and in order to solve the environment, the agent must get an average score of +30 over 100 consecutive episodes.

### Getting Started
The project has been tested on Mac OSX only.

#### Download the environment
1. Download the environment from the link below.
    - Mac OSX: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/one_agent/Reacher.app.zip)

2. Place the file in the `p2_continuous-control/` source folder, and unzip (or decompress) the file.

#### Install all the dependencies
1. Install virtualenv
```
sudo apt install virtualenv
```
2. Create a virtualenv for python3
```
virtualenv -p python3 drlnd
```
3. Activate the environment
```
source drlnd/bin/activate
```
4. Install all the dependencies
```
cd p2_continuous-control/
pip3 install python/
```
Note that the setup assumes that Tkinter is installed for python3. If not, install using the following
```
sudo apt install python3-tk
```

### Training an agent
To train an agent run the following command
```
python DDPG.py --num_episodes 500
```
