import numpy as np
from mlagents_envs.base_env import DecisionSteps

#preprocess sensor data
def sensor_front_sig(data):
    player=[]
    sensor_data=[]
    for sensor in range(33):
        player.append(data[8*sensor:(8*sensor)+8])
    
    for stack in range(3):
        sensor_data.append(player[11*stack:(11*stack)+11])

    return sensor_data

def sensor_back_sig(data):
    player=[]
    sensor_data=[]
    for sensor in range(9):
        player.append(data[8*sensor:(8*sensor)+8])
    
    for stack in range(3):
        sensor_data.append(player[3*stack:(3*stack)+3])

    return sensor_data

def step(decision_steps, max_timesteps=250):

    #Get Signal From Agent 1
    signal_front_1 = np.array(sensor_front_sig(decision_steps.obs[0][0,:]))   #(3, 11, 8)
    signal_back_1 = np.array(sensor_back_sig(decision_steps.obs[1][0,:]))     #(3, 3, 8)

    #Get Signal From Agent 2
    signal_front_2 = np.array(sensor_front_sig(decision_steps.obs[0][1,:]))    #(3, 11, 8)
    signal_back_2 = np.array(sensor_back_sig(decision_steps.obs[1][1,:]))      #(3, 3, 8)

    #preprocess state
    state_1 = np.concatenate((signal_front_1, signal_back_1), axis=1)         #(3, 14, 8)
    state_2 = np.concatenate((signal_front_2, signal_back_2), axis=1)         #(3, 14, 8)

    reward = decision_steps.reward[0]
    reward_total = get_reward(state_1, state_2, reward, max_timesteps)

    done = (reward != 0.0)
    #if done: print("goal! reward =", reward)

    return state_1, state_2, reward_total, done

def change_action_shape(action_1, action_2):

    a1 = 0; b1 = 0; c1 = 0; a2 = 0; b2 = 0; c2 = 0;
    if action_1 < 3:
        a1 = action_1
    elif action_1 < 6:
        b1 = action_1 - 3
    else:
        c1 = action_1 -6
    if action_2 < 3:
        a2 = action_2
    elif action_2 < 6:
        b2 = action_2 -3
    else:
        c2 = action_2 - 6

    action = np.array([(a1, b1, c1), (a2, b2, c2)])
    return action

def get_reward(state_1, state_2, reward, max_timesteps):

    dist1_before, dist1_after = find_min_ball_distance(state_1)
    dist2_before, dist2_after = find_min_ball_distance(state_2)

    reward_ball = 0

    # #if agent 1 is close to ball
    # if dist1_after < 0.3:
    #     reward_ball += 1
    # else if 

    #if agent 1 get closer to the ball
    if dist1_before > dist1_after:
        reward_ball += 1
    else:
        reward_ball -= 1
    
    #if agent 2 get closer to the ball 
    if dist2_before > dist2_after:
        reward_ball += 1
    else :
        reward_ball -= 1

    #final output
    reward_total = reward * max_timesteps + reward_ball
    # print("reward_ball = ", reward_ball)
    return reward_total

def find_min_ball_distance(state):
    min_before = 1.0
    min_after = 1.0

    #find minimum distance from past state
    for sensor in state[1]: # sensor: (14 * 8)
        if (sensor[0].item() == 1) and sensor[-1].item() < min_before:
            min_before = sensor[-1]
        
    #find minimum distance from current state
    for sensor in state[2]: # sensor: (14 * 8)
        if (sensor[0].item() == 1) and sensor[-1].item() < min_after:
            min_after = sensor[-1]

    return min_before, min_after
