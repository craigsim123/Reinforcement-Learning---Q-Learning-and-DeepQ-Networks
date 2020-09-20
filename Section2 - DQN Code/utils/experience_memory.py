"""
A Class to Hold Memories for Mini Batch Learning 

Author: Craig Sim
Related Coursework: "Exploring Reinforcement Learning using Q Learning and Deep Q Network techniques"
Course: INM707 Deep Learning 3: Optimization (PRD2 A 2019/20)
Institution: City, University of London

References:
    This code has been written by Craig Sim with the following references:
        - The solution was inspired by the tutorials from the book
        "Hands-on Intelligent Agents with Open AI Gym" by Praveen Palanisamy
"""
from collections import namedtuple
import random

Experience = namedtuple("Experience", ['obs', 'action', 'reward', 'next_obs',
                                       'done'])


class ExperienceMemory(object):
    def __init__(self, capacity=int(1e6)):
        self.capacity = capacity
        self.mem_idx = 0  # Index of the current experience
        self.memory = []

    def store(self, experience):
        if self.mem_idx < self.capacity:
            # Extend the memory and create space
            self.memory.append(None)
        self.memory[self.mem_idx % self.capacity] = experience
        self.mem_idx += 1

    def sample(self, batch_size):
        assert batch_size <= len(self.memory), "Sample batch_size is more than available in memory"
        return random.sample(self.memory, batch_size)

    def get_size(self):
        return len(self.memory)
