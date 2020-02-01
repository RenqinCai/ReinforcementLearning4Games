from __future__ import print_function
import torch, os, gym, time, glob, argparse, sys
import numpy as np

from scipy.signal import lfilter
from scipy.misc import imresize

import torch.nn as nn
import torch.nn.functional as F
import torch.multiprocessing as mp

os.environ['OMP_NUM_THREADS'] = '1'

def get_args():
    parser = argparse.ArgumentParser(description=None)
    parser.add_argument('--env', default='Breakout-v4', type=str, help="gym environment")
    parser.add_argument('--processes', default=20, type=int, help="number of processes to train with")
    parser.add_argument('--render', default=False, type=bool, help="renders the atari environment")
    parser.add_argument('--test', default=False, type=bool, help='sets lr=0, chooses most likely actions')
    parser.add_argument('--rnn_steps', default=20, type=int, help='steps to train LSTM over')
    parser.add_argument('--lr', default=1e-4, type=float, help='learning rate')
    parser.add_argument('--seed', default=1, type=int, help='seed random # generators (for reproducibility)')
    parser.add_argument('--gamma', default=0.99, type=float, help='rewards discount factor')
    parser.add_argument('--tau', default=1.0, type=float, help='generalized adv estimation discount')
    parser.add_argument('--horizon', default=0.99, type=float, help='horizon for running averages')
    parser.add_argument('--hidden', default=256, type=int, help='hidden size of GRU')

    return parser.parse_args()

discount = lambda x, gamma: lfilter([1], [1, -gamma], x[::-1])[::-1]
prepro = lambda img: imresize(img[35:195].mean(2), (80, 80)).astype(np.float32).reshape(1, 80, 80)/255.

def printlog(args, s, end='\n', mode='a'):
    print(s, end=end); 
    f = open(args.save_dir+'log.txt', mode)
    f.write(s+"\n")
    f.close()

class NNPolicy(nn.Module):
    def __init__(self, channels, memsize, num_actions):
        super(NNPolicy, self).__init__()
        self.conv1 = nn.Conv2d(channels, 32, 3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(32, 32, 3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(32, 32, 3, stride=2, padding=1)
        self.conv4 = nn.Conv2d(32, 32, 3, stride=2, padding=1)

        self.gru = nn.GRUCell(32*5*5, memsize)
        self.critic_linear, self.actor_linear = nn.Linear(memsize, 1), nn.Linear(memsize, num_actions)

    def forward(self, inputs, train=True, hard=False):
        inputs, hx = inputs
        x = 

    def try_load(self. save_dir):
        paths = glob.glob(save_dir+"*.tar")
        step = 0
        if len(paths) > 0:
            ckpts = [int(s.split(".")[-1]) for s in paths]
            ix = np.argmax(ckpts)
            step = ckpts[ix]
            self.load_state_dict(torch.load(paths[ix]))
        if step is 0:
            print("\t no saved models")
        else:
            print("\t loaded model: {}".format(paths[ix]))

        return step

class SharedAdam(torch.optim.Adam):
    def __init__(self, params, lr=1e-3, betas=(0.0, 0.999), eps=1e-8, weight_decay=0):
        super(SharedAdam, self).__init__(params, lr, betas, eps, weight_decay)

        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]
                state['shared_steps'] =
            