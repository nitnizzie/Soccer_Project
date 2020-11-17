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


#Set basic actions
Temp = []
for i in range(3):
    for j in range(3):
         for k in range(3):
                Temp.append((i, j, k))
Actions = []
for a in Temp:
    for b in Temp:
         Actions.append((a, b))

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
agent_p = Agent(state_dim=28, action_dim=1, device=device)

for episode in range(num_episodes):


	# why not env.reset()?
    agent_b.reset()
    agent_p.reset()
    
    #Receive Initial Observation state.
    decision_steps_p, terminal_steps_p = env.get_steps(behavior_name_1)
    decision_steps_b, terminal_steps_b = env.get_steps(behavior_name_2)
    
    front_b1, back_b1, front_b2, back_b2, reward_b = step(decision_steps_b)
    front_p1, back_p1, front_p2, back_p2, reward_p = step(decision_steps_p)

	# get sonsor values, total 28: 14 for each player
    # Only get the most recent state (ignore front_b1[1] and front_b1[2])
    # This is because this model is DQN. This model considers the sequence by default.
    sensor_b = []
    sensor_p = []
    for i in range (11):
         sensor_b.append(front_b1[0][i])
         sensor_p.append(front_p1[0][i])
         sensor_b.append(front_b2[0][i])
         sensor_p.append(front_p2[0][i])
    for i in range (3):
         sensor_b.append(front_b1[0][i])
         sensor_p.append(front_p1[0][i])
         sensor_b.append(front_b2[0][i])
         sensor_p.append(front_p2[0][i])

    state_b = torch.tensor(sensor_b, device = device, dtype = torch.bool)
    state_p = torch.tensor(sensor_p, device = device, dtype = torch.bool)

    for step in range(num_steps):

        # select action
		# I think just putting step as parameter is not meaningful
		# The step value will not be changed unless it is global or you save it otherwise
        action_b = agent_b.select_action(state_b, step)
        action_p = agent_b.select_action(state_p, step)

        #Execute action a_t
        env.set_actions(behavior_name_1, np.array([Actions[action_p]])) #p
        env.set_actions(behavior_name_2, np.array([Actions[action_b]])) #b
        env.step()

        #Observe reward r_t and next state s_(t+1)
        decision_steps_p, terminal_steps_p = env.get_steps(behavior_name_1)
        decision_steps_b, terminal_steps_b = env.get_steps(behavior_name_2)
        
        front_b1, back_b1, front_b2, back_b2, reward_b = step(decision_steps_b)
        front_p1, back_p1, front_p2, back_p2, reward_p = step(decision_steps_p)
		
        sensor_b = []
        sensor_p = []
        for i in range (11):
             sensor_b.append(front_b1[0][i])
             sensor_p.append(front_p1[0][i])
             sensor_b.append(front_b2[0][i])
             sensor_p.append(front_p2[0][i])
        for i in range (3):
             sensor_b.append(front_b1[0][i])
             sensor_p.append(front_p1[0][i])
             sensor_b.append(front_b2[0][i])
             sensor_p.append(front_p2[0][i])

        next_state_b = torch.tensor(sensor_b, device = device, dtype = torch.bool)
        next_state_p = torch.tensor(sensor_b, device = device, dtype = torch.bool)

        #Store Transition to Memory
        agent_b.store_transtion(state_b, Actions[action_b], next_state_b, reward_b)
        agent_p.store_transtion(state_p, Actions[action_p], next_state_p, reward_p)

        state_b = next_state_b
        state_p = next_state_p

        # 
        agent_b.train()
        agent_p.train()


env.close()
