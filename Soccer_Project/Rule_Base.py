import numpy as np
from Action import attack, defend
from mlagents_envs.environment import UnityEnvironment

env = UnityEnvironment(file_name = 'CoE202')

env.reset()
purple_team = list(env.behavior_specs)[0]
blue_team = list(env.behavior_specs)[1]

empty_list=[]

# preprocess sensor data
def sensor_front_sig(data):
    player = []
    sensor_data = []
    for sensor in range(33):
        player.append(data[8 * sensor:(8 * sensor) + 8])

    for stack in range(3):
        sensor_data.append(player[11 * stack:(11 * stack) + 11])

    return sensor_data


def sensor_back_sig(data):
    player = []
    sensor_data = []
    for sensor in range(9):
        player.append(data[8 * sensor:(8 * sensor) + 8])

    for stack in range(3):
        sensor_data.append(player[3 * stack:(3 * stack) + 3])

    return sensor_data


def step(decision_steps):
    # Get Signal From Agent 1
    signal_front_1 = np.array(sensor_front_sig(decision_steps.obs[0][0, :]))  # (3, 11, 8)
    signal_back_1 = np.array(sensor_back_sig(decision_steps.obs[1][0, :]))  # (3, 3, 8)

    # Get Signal From Agent 2
    signal_front_2 = np.array(sensor_front_sig(decision_steps.obs[0][1, :]))  # (3, 11, 8)
    signal_back_2 = np.array(sensor_back_sig(decision_steps.obs[1][1, :]))  # (3, 3, 8)

    # preprocess state
    state_1 = np.concatenate((signal_front_1, signal_back_1), axis=1)  # (3, 14, 8)
    state_2 = np.concatenate((signal_front_2, signal_back_2), axis=1)  # (3, 14, 8)

    reward = decision_steps.reward[0]
    done = (reward != 0.0)
    # print(reward, done)
    if done: print("goal! reward =", reward)

    return state_1, state_2, reward, done

def change_action_shape(action_1, action_2):
    action = np.array([action_1, action_2])
    return action

max_episodes = 100          # max episodes
max_timesteps = 1000        # max timesteps in one episode

for i_episodes in range(max_episodes):
    # set delay
    delay_p1 = 0
    delay_p2 = 0
    delay_b1 = 0
    delay_b2 = 0

    # mark agent
    purple = 0
    blue = 1

    for t in range(max_timesteps):
        # Get Signal
        decision_steps_p, terminal_steps_p = env.get_steps(purple_team)
        decision_steps_b, terminal_steps_b = env.get_steps(blue_team)
        state_p1, state_p2, reward_p, done = step(decision_steps_p)
        state_b1, state_b2, reward_b, _ = step(decision_steps_b)

        # running action
        act_p1, delay_p1 = attack(purple, delay_p1, t, state_p1)
        act_p2, delay_p2 = defend(purple, delay_p2, t, state_p2)
        act_b1, delay_b1 = attack(blue, delay_b1, t, state_b1)
        act_b2, delay_b2 = defend(blue, delay_b2, t, state_b2)

        # Set the actions
        action_p = change_action_shape(act_p1, act_p2)
        action_b = change_action_shape(act_b1, act_b2)

        # Set the actions
        env.set_actions(purple_team, np.array(action_p))
        env.set_actions(blue_team, np.array(action_b))

        # Move the simulation forward
        env.step()

        # print parameter
        print("timesteps : ", t)
        print("action_p_atk : ", act_p1)
        print("action_p_def : ", act_p2)
        print("action_b_atk : ", act_b1)
        print("action_b_def : ", act_b2)
        print("\n")

    env.close()