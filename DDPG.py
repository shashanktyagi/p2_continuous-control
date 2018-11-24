import os
import argparse
import random
import torch
import numpy as np
import matplotlib.pyplot as plt
from unityagents import UnityEnvironment
from collections import deque
from ddpg_agent import Agent

parser = argparse.ArgumentParser('Deep Deterministic Policy Gradient algorithm')
parser.add_argument('--train', required=False, default=False, action='store_true',
                    help='if true the agent is trained otherwise checkpoint is used')
parser.add_argument('--checkpoint', required=False, type=str, default='checkpoint.pth',
                    help='path to the checkpoint')
parser.add_argument('--num_episodes', required=False, type=int, default=1000,
                    help='num of episodes to train for')
args = parser.parse_args()

SHOW_RANDOM_AGENT = False

env = UnityEnvironment(file_name='./Reacher.app')
# get default brain
brain_name = env.brain_names[0]
brain = env.brains[brain_name]
# reset the environment
env_info = env.reset(train_mode=True)[brain_name]

# number of agents
num_agents = len(env_info.agents)
print('Number of agents:', num_agents)

# number of actions
action_size = brain.vector_action_space_size
print('Size of each action: ', action_size)

# examine the state space
states = env_info.vector_observations
state_size = states.shape[1]
print('There are {} agents. Each observes a state with length: {}'.format(states.shape[0],
                                                                          state_size))
print('The state for the first agent looks like:', states[0])

if SHOW_RANDOM_AGENT:
    env_info = env.reset(train_mode=False)[brain_name]     # reset the environment
    states = env_info.vector_observations                  # get the current state (for each agent)
    scores = np.zeros(num_agents)                          # initialize the score (for each agent)
    while True:
        actions = np.random.randn(num_agents, action_size) # select an action (for each agent)
        actions = np.clip(actions, -1, 1)                  # all actions between -1 and 1
        env_info = env.step(actions)[brain_name]           # send all actions to tne environment
        next_states = env_info.vector_observations         # get next state (for each agent)
        dones = env_info.local_done                        # see if episode finished
        scores += env_info.rewards                         # update the score (for each agent)
        states = next_states                               # roll over states to next time step
        if np.any(dones):                                  # exit loop if episode finished
            break
    print('Total score (averaged over agents) this episode: {}'.format(np.mean(scores)))

agent = Agent(state_size=state_size, action_size=action_size, random_seed=10)

def ddpg(n_episodes=1000, max_t=300, print_every=100):
    scores_deque = deque(maxlen=print_every)
    scores = []
    mean_scores = []                 # list containing running mean scores from 100 episodes
    for i_episode in range(1, n_episodes+1):
        env_info = env.reset(train_mode=True)[brain_name]
        state = env_info.vector_observations[0]
        agent.reset()
        score = 0
        for t in range(max_t):
            action = agent.act(state)
            env_info = env.step(action)[brain_name]
            next_state = env_info.vector_observations[0]
            reward = env_info.rewards[0]
            done = env_info.local_done[0]
            agent.step(state, action, reward, next_state, done)

            state = next_state
            score += reward
            if done:
                break
        scores_deque.append(score)
        scores.append(score)
        mean_scores.append(np.mean(scores_deque))  # save running mean
        print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_deque)), end="")
        torch.save(agent.actor_local.state_dict(), 'checkpoint_actor.pth')
        torch.save(agent.critic_local.state_dict(), 'checkpoint_critic.pth')
        if i_episode % print_every == 0:
            print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_deque)))

        if np.mean(scores_deque) > 30.0:
            print('\nEnvironment solved in {:d} episodes!\tAverage Score: {:.2f}'.format(i_episode-100,
                                                                                         np.mean(scores_deque)))
            break

    return scores, mean_scores


if args.train:
    print('Training the agent for {} episodes...'.format(args.num_episodes))
    scores, mean_scores = ddpg(n_episodes=args.num_episodes, max_t=1000)

    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.plot(np.arange(1, len(scores)+1), scores, label='scores')
    plt.plot(np.arange(1, len(mean_scores)+1), mean_scores, label='running mean over 100 episodes')
    plt.ylabel('Score')
    plt.xlabel('Episode #')
    plt.legend(loc='lower right')
    plt.grid()
    plt.savefig('training_scores.png', format='png', dpi=1000)
    plt.show()

print('\nvisualizing trained agent')

try:
    # load the weights from file
    agent.actor_local.load_state_dict(torch.load('checkpoint_actor.pth'))
    agent.critic_local.load_state_dict(torch.load('checkpoint_critic.pth'))
except:
    raise Exception('Could not load checkpoint!')

env_info = env.reset(train_mode=False)[brain_name] # reset the environment
state = env_info.vector_observations[0]            # get the current state
scores = np.zeros(num_agents)                          # initialize the score (for each agent)
while True:
    action = agent.act(state)
    action = np.clip(action, -1, 1)                  # all actions between -1 and 1
    env_info = env.step(action)[brain_name]           # send all actions to tne environment
    next_states = env_info.vector_observations         # get next state (for each agent)
    dones = env_info.local_done[0]                        # see if episode finished
    scores += env_info.rewards                         # update the score (for each agent)
    states = next_states                               # roll over states to next time step
    if np.any(dones):                                  # exit loop if episode finished
        break
print('Total score (averaged over agents) this episode: {}'.format(np.mean(scores)))
env.close()
