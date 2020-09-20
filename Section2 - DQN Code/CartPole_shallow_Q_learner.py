"""
A deep q learner that implements the CartPole environment
using a shallow neural network with no experience replay

4 inputs
40 hidden perceptrons
2 outputs

Author: Craig Sim
Related Coursework: "Exploring Reinforcement Learning using Q Learning and Deep Q Network techniques"
Related Coursework Section: "Single DQN without Memory Replay"
Course: INM707 Deep Learning 3: Optimization (PRD2 A 2019/20)
Institution: City, University of London

IDE: Spyder
Python: 3.6

References:
    This code has been written by Craig Sim with the following references:
        - The solution was loosely inspired by the tutorials from the book
        "Hands-on Intelligent Agents with Open AI Gym" by Praveen Palanisamy
        - The code to create action, model loss, bellman residual and state
        chart visualisations where sourced from 
        https://github.com/criteo-research/paiss_deeprl/blob/master/utils.py
"""
import gym
import random
import torch
import numpy as np
from utils.perceptron import SLP
from utils.utils import RLDebugger 
import matplotlib.pyplot as plt
from  matplotlib.ticker import MaxNLocator
import seaborn as sns
import pandas as pd

env = gym.make("CartPole-v1")
MAX_NUM_EPISODES = 300
MAX_STEPS_PER_EPISODE = 300

class Shallow_Q_Learner(RLDebugger):
    def __init__(self, state_shape, action_shape, learning_rate=0.005, gamma=0.8):
        RLDebugger.__init__(self)
        self.state_shape = state_shape
        self.action_shape = action_shape
        self.gamma = gamma #Agents Discount Factor
        self.learning_rate = learning_rate  # Agents Q learning rate

        #self.Q is the Action Value function 
        # Q-values
        self.Q = SLP(state_shape, action_shape)
        self.Q_optimizer = torch.optim.Adam(self.Q.parameters(), lr=1e-3)
        
        #self.policy is the Policy followed by the agent
        #we're using an epsilon greedy policy
        self.policy = self.epsilon_greedy_Q
        self.epsilon = 1.0
        self.epsilon_max = 1.0
        self.epsilon_min = 0.05
        self.epsilon_decay = 0.99999
        
        self.step_num = 0

    def get_action(self, observation):
        return self.policy(observation)

    def learn(self, obs, action, reward, next_obs):
        td_target = reward + self.gamma * torch.max(self.Q(next_obs))
        td_error = torch.nn.functional.mse_loss(self.Q(obs)[action], td_target)
        
        self.Q_optimizer.zero_grad()
        td_error.backward()
        self.Q_optimizer.step()
        self.record(action, obs, self.Q(obs), self.Q(next_obs), td_error, reward)
    
    def epsilon_greedy_Q(self, observation):
        # Decay Epsilon/exploration as per schedule
        self.step_num +=1
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
            
        if random.random() < self.epsilon: #self.epsilon_decay(self.step_num): # and not self.params["test"]:
            action = random.choice([i for i in range(self.action_shape)])
        else:
            action = np.argmax(self.Q(observation).data.numpy())
        return action

if __name__ == "__main__":
    observation_shape = env.observation_space.shape
    action_shape = env.action_space.n
     
    agent = Shallow_Q_Learner(observation_shape, action_shape)
    first_episode = True
    episode_rewards = list()
    results_list = []
    for episode in range(MAX_NUM_EPISODES):
        obs = env.reset()
        cum_reward = 0.0 #Cumulative Reward
        for step in range(MAX_STEPS_PER_EPISODE):
            env.render()
            action = agent.get_action(obs)
            next_obs, reward, done, info = env.step(action)
            agent.learn(obs, action, reward, next_obs)
            
            obs= next_obs
            cum_reward += reward
            
            if done is True:
                if first_episode: #Initialize reward at the end of the first episode
                    max_reward = cum_reward
                    first_episode = False
                episode_rewards.append(cum_reward)
                if cum_reward > max_reward:
                    max_reward = cum_reward
                this_step = step+1
                print("\nEpisode#{} ended in  {} steps. reward ={} mean_reward={} best_reward={}".
                      format(episode, this_step, cum_reward, np.mean(episode_rewards), max_reward))
                new_result = []
                new_result.append(episode)     
                new_result.append(this_step)
                new_result.append(cum_reward)
                new_result.append(np.mean(episode_rewards))     
                new_result.append(max_reward)
                results_list.append(new_result)
                break
    agent.plot_diagnostics()
    env.close()
    
    results = pd.DataFrame(results_list, columns = ['Episode', 
                                     'Steps', 'Cumulative Reward',
                                     'Mean Reward','Best Reward'])
    
    sns.set_context("notebook", font_scale=0.9, rc={"lines.linewidth": 2.5})
    plt.figure(figsize=(16, 6))
    plt.xticks(rotation=45, horizontalalignment='right')
    plt.gca().yaxis.set_major_locator(MaxNLocator(integer=True))
    title = "CartPole Single DQN without Memory Replay"
    plt.title(title, fontsize= 15)
    ax = sns.barplot(x='Episode', y='Cumulative Reward', data=results)
    new_ticks = [i.get_text() for i in ax.get_xticklabels()]
    plt.xticks(range(0, len(new_ticks), 50), new_ticks[::50])
    #Add a cumulative lineplot with separate y axis, on the right hand side of the plot
    ax2 = ax.twinx()
    ax2.yaxis.set_major_locator(MaxNLocator(integer=True))
    sns.lineplot(x='Episode', y='Mean Reward', data=results, ax=ax2)
    
    plt.show(sns)