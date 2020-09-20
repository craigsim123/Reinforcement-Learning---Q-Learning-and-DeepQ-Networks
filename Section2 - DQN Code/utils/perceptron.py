"""
Shallow Feed Forward Neural Network written in Pytorch 

Author: Craig Sim
Related Coursework: "Exploring Reinforcement Learning using Q Learning and Deep Q Network techniques"
Course: INM707 Deep Learning 3: Optimization (PRD2 A 2019/20)
Institution: City, University of London

References:
    This code has been written by Craig Sim with the following references:
        - The solution was inspired by the tutorials from the book
        "Hands-on Intelligent Agents with Open AI Gym" by Praveen Palanisamy
"""

import torch


class SLP(torch.nn.Module):
    """
    A Single Layer Perceptron (SLP) class to approximate functions
    """
    def __init__(self, input_shape, output_shape, device=torch.device("cpu")):
        super(SLP, self).__init__()
        self.device = device
        self.input_shape = input_shape[0]
        self.hidden_shape = 40
        self.linear1 = torch.nn.Linear(self.input_shape, self.hidden_shape)
        self.out = torch.nn.Linear(self.hidden_shape, output_shape)

    def forward(self, x):
        x = torch.from_numpy(x).float().to(self.device)
        x = torch.nn.functional.relu(self.linear1(x))
        x = self.out(x)
        return x
