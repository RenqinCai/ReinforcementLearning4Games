from __future__ import print_function
import torch, os, gym, time, glob, argparse, sys
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.multiprocessing as mp
from collections import namedtuple
import random

Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))

class ReplayMemory(object):
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, *args):
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position+1)%self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


class NNPolicy(nn.Module):
    def __init__(self, num_actions, state_dim):
        super(NNPolicy, self).__init__()
        self.fc1 = nn.Linear(state_dim, 200)
        self.fc2 = nn.Linear(200, num_actions)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        action_value = self.fc2(x)

        return action_value




    
