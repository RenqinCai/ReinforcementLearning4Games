from __future__ import print_function
from collections import namedtuple
from itertools import count
from PIL import Image
import random
import torch, os, gym, time, glob, argparse, sys
from gym import wrappers
from Atari_DQN import NNPolicy, ReplayMemory
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T
import math
import numpy as np

os.environ['OMP_NUM_THREADS'] = '1'
Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))

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

def printlog(args, s, end='\n', mode='a'):
    print(s, end=end); 
    f = open(args.save_dir+'log.txt', mode)
    f.write(s+"\n")
    f.close()

discount = lambda x, gamma: lfilter([1], [1, -gamma], x[::-1])[::-1]
prepro = lambda img: imresize(img[35:195].mean(2), (80, 80)).astype(np.float32).reshape(1, 80, 80)/255.

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

BATCH_SIZE = 128
GAMMA = 0.999
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 200
TARGET_UPDATE = 10

env = gym.make('PongNoFrameskip-v4')
env = wrappers.Monitor(env, 'tmp/pong-base', force=True) # record the game as as an mp4 file
n_actions = env.action_space.n
print("n_actions", n_actions)


def prepro(I):
    # print("0 I", I.shape)
    I = I[35:185]
    # print("1 I", I.shape)
    I = I[::2, ::2, 0]
    # print("2 I", I.shape)
    I[I == 144] = 0
    # print("3 I", I.shape)
    I[I == 109] = 0
    # print("4 I", I.shape)
    I[I != 0] = 1
    I = np.array(I)
    # print("I", I.dtype)
    # print("I size", I.shape)
    return I.flatten()    

observation = env.reset()
print("observation", observation.shape)
state = prepro(observation)
state_size = state.shape[0]
print("state_size", state_size)

# observation = env.reset()
# print("observation", observation.shape)
# state = prepro(observation)
# print("state size", state.shape)
# exit()

policy_net = NNPolicy(n_actions, state_size).to(device)
target_net = NNPolicy(n_actions, state_size).to(device)
target_net.load_state_dict(policy_net.state_dict())
target_net.eval()

optimizer = optim.RMSprop(policy_net.parameters())
memory = ReplayMemory(1000)

steps_done = 0
D = state_size

def select_action(state):
    global steps_done
    sample = random.random()
    eps_threshold = EPS_END+(EPS_START-EPS_END)*math.exp(-1.0*steps_done/EPS_DECAY)

    steps_done += 1
    if sample > eps_threshold:
        with torch.no_grad():
            # print("state", state)
            state = torch.FloatTensor(state).unsqueeze(0).to(device)
            value = policy_net(state)
            action_max_value, index = torch.max(value, 1)
            action = index.item()
            return action
    else:
        action = np.random.choice(range(n_actions), 1).item()
        return action

episode_durations = []

def optimize_model():
    if len(memory) < BATCH_SIZE:
        return 

    transitions = memory.sample(BATCH_SIZE)

    batch = Transition(*zip(*transitions))
    next_state_batch = batch.next_state

    # print("next_state_batch", next_state_batch)
    # exit()

    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, next_state_batch)), device=device, dtype=torch.bool)

    s_list = []
    for s in next_state_batch:
        if s is not None:
            # print("s", s)
            s_tensor = torch.FloatTensor(s).unsqueeze(0)
            # print("s_tensor", s_tensor)
            s_list.append(s_tensor)

    non_final_next_states = torch.cat([torch.FloatTensor(s).unsqueeze(0) for s in next_state_batch if s is not None]).to(device)
    # exit()

    state_batch = torch.cat([torch.FloatTensor(i).unsqueeze(0) for i in batch.state]).to(device)
    
    action_batch = torch.cat([torch.FloatTensor([i]).unsqueeze(0) for i in batch.action]).long().to(device)

    reward_batch = torch.cat([torch.FloatTensor([i]) for i in batch.reward]).to(device)

    state_action_values = policy_net(state_batch).gather(1, action_batch)
    next_state_values = torch.zeros(BATCH_SIZE, device=device)

    non_final_next_states = target_net(non_final_next_states)
    # print("non_final_next_states", non_final_next_states.size())

    next_state_values[non_final_mask] = non_final_next_states.max(1)[0].detach()
    # print("next_state_values", next_state_values.size())

    expected_state_action_values = (next_state_values*GAMMA) + reward_batch

    # print("state_action_values", state_action_values.size())
    # print("expected_state_action_values", expected_state_action_values.size())

    loss = F.smooth_l1_loss(state_action_values, expected_state_action_values.unsqueeze(1))

    optimizer.zero_grad()
    loss.backward()

    for param in policy_net.parameters():
        param.grad.data.clamp_(-1, 1)
    
    optimizer.step()

def observation_2_state(observation, prev_x):

    cur_x = prepro(observation)

    state = cur_x - prev_x if prev_x is not None else torch.zeros(D)

    return state, cur_x

num_episodes = 50

### D: input dim

# observation = env.reset()
# prev_x = None

for i_episode in range(num_episodes):
    # if i_episode %5 == 0:
    print("episode ", i_episode)
    # env.reset()
    # print("state", state)
    observation = env.reset()
    prev_x = None
    state = None

    for t in count():
        
        state, prev_x = observation_2_state(observation, prev_x)

        action = select_action(state)

        next_observation, reward, done, info = env.step(action)
        # reward = [reward]
        # reward = torch.tensor([reward], device=device)

        next_state, prev_x = observation_2_state(next_observation, prev_x)

        if done:
            next_state = None

        memory.push(state, action, next_state, reward)

        observation = next_observation

        optimize_model()  

        if done:
            episode_durations.append(t+1)
            break
    # exit()
    if i_episode % TARGET_UPDATE == 0:
        target_net.load_state_dict(policy_net.state_dict())

print("complete")          
