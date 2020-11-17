'''
DQN Soccer Robot Simulation with Unity Environment.

Author : Shinhyeok Hwang
Course : CoE202
Algorithm : DQN(Deep Q-Network Learning)
https://arxiv.org/pdf/1312.5602.pdf
'''

import math
import random
import numpy as np
from collections import deque

import torch
import torch.optim as optim
import torch.nn.functional as F

from model import DQN


#Hyperparameters for Learning
BATCH_SIZE = 128
GAMMA = 0.999
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 200
TARGET_UPDATE = 10

BUFFER_SIZE = 10000

class ReplayMemory(object):
    def __init__(self, buffer_size, name_buffer=''):
        self.buffer_size=buffer_size  #choose buffer size
        self.num_exp = 0
        self.buffer=deque()

    def add(self, state, action, reward, next_state):
        experience=(state, action, reward, next_state)
        if self.num_exp < self.buffer_size:
            self.buffer.append(experience)
            self.num_exp +=1
        else:
            self.buffer.popleft()
            self.buffer.append(experience)

    def size(self):
        return self.buffer_size

    def count(self):
        return self.num_exp

    def sample(self, batch_size):
        if self.num_exp < batch_size:
            batch=random.sample(self.buffer, self.num_exp)
        else:
            batch=random.sample(self.buffer, batch_size)

        state, action, reward, terminal, next_state = map(np.stack, zip(*batch))

        return state, action, reward, terminal, next_state

    def clear(self):
        self.buffer = deque()
        self.num_exp=0



class Agent():
    def __init__(self, state_dim, action_dim, device = 'cpu'):

        #Initialize Replay Memory
        self.memory = ReplayMemory(BUFFER_SIZE)


        # Initialize target network
        # There will be 9*9 kinds of outputs: [foward, side, turn] 3 * 3 * 3 = 9
        self.policy_net = DQN(state_dim, action_dim, 81).to(device)
        self.target_net = DQN(state_dim, action_dim, 81).to(device)

        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.RMSprop(self.policy_net.parameters())
        self.device = device

    def select_action(self, state, steps_done):
        sample = random.random()
        eps_threshold = EPS_END + (EPS_START - EPS_END) * \
            math.exp(-1. * steps_done / EPS_DECAY)
        steps_done += 1
        if sample > eps_threshold:
            with torch.no_grad():
                # t.max (1)은 각 행의 가장 큰 열 값을 반환합니다.
                # 최대 결과의 두번째 열은 최대 요소의 주소값이므로,
                # 기대 보상이 더 큰 행동을 선택할 수 있습니다.
                return self.policy_net(state).max(1)[1].view(1, 1)
        else:	#n_actions = 9, reasoned above
            return torch.tensor([[random.randrange(81)]], device=self.device, dtype=torch.long)

    def reset(self):
        return

    def store_transtion(self, state, action ,next_state, reward):
        self.memory.push(state, action, next_state, reward)

    def train(self):
        #keep adding experiences to the memory until there are at least minibatch size samples
        if len(self.memory) < BATCH_SIZE:
            return

        #Sample a Random-minibatch of N transitions from R
        s_batch, a_batch, r_batch, s2_batch =  self.memory.sample(BATCH_SIZE)

        s_batch = torch.FloatTensor(s_batch).to(self.device) #(batch_dim, state_dim)
        a_batch = torch.FloatTensor(a_batch).to(self.device) #(batch_dim, action_dim)
        r_batch = torch.FloatTensor(r_batch).unsqueeze(1).to(self.device)
        s2_batch = torch.FloatTensor(s2_batch).to(self.device)

        # Q(s_t, a) 계산 - 모델이 Q(s_t)를 계산하고, 취한 행동의 열을 선택합니다.
        # 이들은 policy_net에 따라 각 배치 상태에 대해 선택된 행동입니다.
        state_action_values = self.policy_net(s_batch).gather(1, a_batch)

        # 모든 다음 상태를 위한 V(s_{t+1}) 계산
        # non_final_next_states의 행동들에 대한 기대값은 "이전" target_net을 기반으로 계산됩니다.
        # max(1)[0]으로 최고의 보상을 선택하십시오.
        # 이것은 마스크를 기반으로 병합되어 기대 상태 값을 갖거나 상태가 최종인 경우 0을 갖습니다.
        next_state_values = torch.zeros(BATCH_SIZE, device=self.device)
        next_state_values[non_final_mask] = self.target_net(non_final_next_states).max(1)[0].detach()
        # 기대 Q 값 계산
        expected_state_action_values = (next_state_values * GAMMA) + r_batch

        # Huber 손실 계산
        loss = F.smooth_l1_loss(state_action_values, expected_state_action_values.unsqueeze(1))

        # 모델 최적화
        optimizer.zero_grad()
        loss.backward()
        for param in policy_net.parameters():
            param.grad.data.clamp_(-1, 1)
            optimizer.step()
            return
