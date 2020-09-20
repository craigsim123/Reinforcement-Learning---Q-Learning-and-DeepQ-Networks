"""
A deep q learner that implements the CartPole environment
using Double shallow neural networks with memory replay

4 inputs
40 hidden perceptrons
2 outputs

Author: Craig Sim
Related Coursework: "Exploring Reinforcement Learning using Q Learning and Deep Q Network techniques"
Related Coursework Section: "Single DQN with Memory Replay"
Course: INM707 Deep Learning 3: Optimization (PRD2 A 2019/20)
Institution: City, University of London

IDE: Spyder
Python: 3.6

References:
    This code has been written by Craig Sim with the following references:
        - The solution was loosely inspired by the tutorials from the book
        "Hands-on Intelligent Agents with Open AI Gym" by Praveen Palanisamy
        - The code to create action, model loss, bellman residual and state
cc
"""
import gym
import random
import torch
import numpy as np
from utils.perceptron import SLP
from utils.utils import RLDebugger
from utils.experience_memory import ExperienceMemory
from utils.experience_memory import Experience
import matplotlib.pyplot as plt
from  matplotlib.ticker import MaxNLocator
import seaborn as sns
import pandas as pd

env = gym.make("CartPole-v1")
MAX_NUM_EPISODES = 300
MAX_STEPS_PER_EPISODE = 300
REPLAY_BATCH_SIZE = 64
REPLAY_START_SIZE = 1000
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Shallow_Q_Learner(RLDebugger):
    def __init__(self, state_shape, action_shape, learning_rate=0.005, gamma=0.8):
        RLDebugger.__init__(self)
        self.state_shape = state_shape
        self.action_shape = action_shape
        self.gamma = gamma #Agents Discount Factor
        self.learning_rate = learning_rate  # Agents Q learning rate
        self.best_mean_reward = - float("inf") # Agent's personal best mean episode reward
        self.best_reward = - float("inf")
        self.training_steps_completed = 0  # Number of training batch steps completed so far

        #self.Q is the Action Value function 
        # Q-values
        self.Q = SLP(state_shape, action_shape)
        self.Q_optimizer = torch.optim.Adam(self.Q.parameters(), lr=learning_rate)
        
        #self.policy is the Policy followed by the agent
        #we're using an epsilon greedy policy
        self.policy = self.epsilon_greedy_Q
        self.epsilon = 1.0
        self.epsilon_max = 1.0
        self.epsilon_min = 0.05
        self.epsilon_decay = 0.99999
        
        self.step_num = 0
        self.memory = ExperienceMemory()  # Initialize an Experience memory with 1000 capacity


    def replay_experience(self, batch_size=REPLAY_BATCH_SIZE):
        """
        replays a mini batch of experience sampled from the ExperienceMemory
        """
        experience_batch = self.memory.sample(batch_size)
        self.learn_from_batch_experience(experience_batch)
        

    def learn_from_batch_experience(self, experiences):
        """
        Update the DQN based on the learning from a mini batch of experience
        """
        batch_xp = Experience(*zip(*experiences))
        obs_batch = np.array(batch_xp.obs)
        action_batch = np.array(batch_xp.action)
        reward_batch = np.array(batch_xp.reward)
        next_obs_batch = np.array(batch_xp.next_obs)
        done_batch = np.array(batch_xp.done)
        
        td_target = reward_batch + ~done_batch * \
                np.tile(self.gamma, len(next_obs_batch)) * \
                self.Q(next_obs_batch).detach().max(1)[0].data.cpu().numpy()
        
        td_target = torch.from_numpy(td_target).to(device)
        action_idx = torch.from_numpy(action_batch).to(device)
        td_error = torch.nn.functional.mse_loss( self.Q(obs_batch).gather(1, action_idx.view(-1, 1)),
                                                       td_target.float().unsqueeze(1))
        print("\nReplay Experience!!!!")
        self.Q_optimizer.zero_grad()
        td_error.mean().backward()
        self.Q_optimizer.step()
        
    def get_action(self, observation):
        return self.policy(observation)
        
    def learn(self, obs, action, reward, next_obs, done):
        # TD(0) Q-learning
        if done:  # End of episode
            td_target = self.Q(next_obs) * 0.0 + reward
        else:
            td_target = reward + self.gamma * torch.max(self.Q(next_obs))

        td_error = torch.nn.functional.mse_loss(self.Q(obs)[action], td_target)
        
        # Update Q estimate
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
    #agent_params = params_manager.get_agent_params()
    #agent_params["test"] = args.test
    print(env.observation_space, env.action_space)
    agent = Shallow_Q_Learner(observation_shape, action_shape)
    first_episode = True

    episode_rewards = list()

    episode = 0
    global_step_num = 0
    results_list = []    
    for episode in range(MAX_NUM_EPISODES):
    #while global_step_num <= MAX_NUM_EPISODES:
        obs = env.reset()
        cum_reward = 0.0  # Cumulative reward
        done = False
        step = 0
        #for step in range(MAX_STEPS_PER_EPISODE):
        while not done:
            env.render()
            action = agent.get_action(obs)
            next_obs, reward, done, info = env.step(action)
            agent.learn(obs, action, reward, next_obs, done)
            agent.memory.store(Experience(obs, action, reward, next_obs, done))

            obs = next_obs
            cum_reward += reward
            step += 1
            global_step_num +=1

            if done is True:
                episode += 1
                episode_rewards.append(cum_reward)
                if cum_reward > agent.best_reward:
                    agent.best_reward = cum_reward

                this_step = step+1
                print("\nEpisode#{} ended in  {} steps. reward ={} mean_reward={} best_reward={} mem_size{}".
                      format(episode, this_step, cum_reward, np.mean(episode_rewards), agent.best_reward, agent.memory.get_size()))
                # Learn from batches of experience once a certain amount of xp is available unless in test only mode
                
                new_result = []
                new_result.append(episode)     
                new_result.append(this_step)
                new_result.append(cum_reward)
                new_result.append(np.mean(episode_rewards))     
                new_result.append(agent.best_reward)
                results_list.append(new_result)
                
                if agent.memory.get_size() >= 2 * REPLAY_START_SIZE:
                    agent.replay_experience()
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
    title = "CartPole Single DQN with Memory Replay"
    plt.title(title, fontsize= 15)
    ax = sns.barplot(x='Episode', y='Cumulative Reward', data=results)
    new_ticks = [i.get_text() for i in ax.get_xticklabels()]
    plt.xticks(range(0, len(new_ticks), 50), new_ticks[::50])
    #Add a cumulative lineplot with separate y axis, on the right hand side of the plot
    ax2 = ax.twinx()
    ax2.yaxis.set_major_locator(MaxNLocator(integer=True))
    sns.lineplot(x='Episode', y='Mean Reward', data=results, ax=ax2)
    
    plt.show(sns)
