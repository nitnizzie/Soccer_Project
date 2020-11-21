import numpy as np


from mlagents_envs.environment import UnityEnvironment
env = UnityEnvironment(file_name = 'CoE202')

env.reset()
behavior_name_1 = list(env.behavior_specs)[0]
behavior_name_2 = list(env.behavior_specs)[1]

#print('b_n',env.brains)
decision_steps_p, _ = env.get_steps(behavior_name_1)
decision_steps_b, _ = env.get_steps(behavior_name_2)
cur_obs_b = decision_steps_b.obs[0][0,:]
empty_list=[]

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

for i in range(100):
    decision_steps_p, terminal_steps_p = env.get_steps(behavior_name_1)
    decision_steps_b, terminal_steps_b = env.get_steps(behavior_name_2)
    
    cur_obs_b = decision_steps_b.obs[0][0,:]
    cur_obs_p = decision_steps_p.obs[0][0,:]

    signal = sensor_front_sig(decision_steps_b.obs[0][0,:])
    signal_back = sensor_back_sig(decision_steps_b.obs[1][0,:])

    print("cur observations : ")
    print(signal[0][0][7])
    print(signal_back[0])
    
    # Set the actions
    env.set_actions(behavior_name_1, np.array([(0,0,0),(0,0,0)]))
    env.set_actions(behavior_name_2, np.array([(0,0,0),(0,0,0)]))
    
    # Move the simulation forward
    env.step()

env.close()
