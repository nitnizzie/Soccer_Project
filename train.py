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
from copy import copy, deepcopy

import torch

from utils import step
from dqn_agent import Agent
from mlagents_envs.environment import UnityEnvironment


#Hyperparameters for tuning
num_episodes = 100
num_steps = 500

#set GPU for faster training
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#Set Unity Environment
env = UnityEnvironment(file_name = 'CoE202')
env.reset()


#p : Purple Team, b: Blue Team
behavior_name_1 = list(env.behavior_specs)[0]
behavior_name_2 = list(env.behavior_specs)[1]

decision_steps_p, terminal_steps_p = env.get_steps(behavior_name_1)
decision_steps_b, terminal_steps_b = env.get_steps(behavior_name_2)

front_b1, back_b1, front_b2, back_b2, reward_b = step(decision_steps_b)
front_p1, back_p1, front_p2, back_p2, reward_p = step(decision_steps_p)


'''
# front[2], back[2] : current time values
print(front_b1[0])
print(front_b1[1])
print(front_b1[2])
'''

agent_b = Agent(state_dim=5, action_dim=1, device=device)
agent_p = Agent(state_dim=5, action_dim=1, device=device)

for episode in range(num_episodes):

    agent_b.reset()
    agent_p.reset()
    
    #Receive Initial Observation state.
    decision_steps_p, terminal_steps_p = env.get_steps(behavior_name_1)
    decision_steps_b, terminal_steps_b = env.get_steps(behavior_name_2)
    
    front_b1, back_b1, front_b2, back_b2, reward_b = step(decision_steps_b)
    front_p1, back_p1, front_p2, back_p2, reward_p = step(decision_steps_p)

    for step in range(num_steps):

        #select action
        action_b = agent_b.select_action(state, step)
        action_p = agent_b.select_action(state, step)

        #Execute action a_t
        env.set_actions(behavior_name_1, np.array([(1,0,0),(2,0,0)])) #p
        env.set_actions(behavior_name_2, np.array([(0,0,1),(0,1,0)])) #b
        env.step()

        #Observe reward r_t and next state s_(t+1)
        decision_steps_p, terminal_steps_p = env.get_steps(behavior_name_1)
        decision_steps_b, terminal_steps_b = env.get_steps(behavior_name_2)
        
        front_b1, back_b1, front_b2, back_b2, reward_b = step(decision_steps_b)
        front_p1, back_p1, front_p2, back_p2, reward_p = step(decision_steps_p)

        #Store Transition to Memory
        agent_b.store_transtion(state, action, next_state, reward)
        agent_p.store_transtion(state, action, next_state, reward)

        # 
        agent_b.train()
        agent_p.train()


env.close()
