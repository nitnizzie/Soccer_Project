import numpy as np
from mlagents_envs.environment import UnityEnvironment
from utils import sensor_front_sig, sensor_back_sig
from abc import *

def get_score_from_decision_step(decision_steps):
    score = decision_steps.reward[0]
    return score

class Agent(metaClass=ABCMeta):
    @abstractmethod
    def decide_movement(self):
        pass

class Agent_dis(Agent):
    def __init__(self,index):
        self.find_enemy = False
        self.distance_enemy = 0
        self.direction_enemy = 0
        #Only turns on when collison to wall occured.
        self.must_find_enemy = False
        self.index = index

    def update_enemy(self,dec_step_obs):
        self.must_find_enemy = not (self.find_enemy)
        if self.must_find_enemy:
            pass
        else:
            pass
        return

    def decide_movement(self,dec_step_obs):
        self.update_enemy(dec_step_obs)
        return (,,)

class Agent_ball(Agent):
    def __init__(self,index):
        self.find_ball = False
        self.distance_ball = 0
        self.direction_ball = 0
        self.not_detected_for = 0
        self.threshold = 10
        self.index = index

    def update_ball(self,dec_step_obs):
        return

    def decide_movement(self,dec_step_obs):
        self.update_ball(dec_step_obs)
        return (,,)

input_n = int(input('Please write how many times do you want to run the code.'))
env = UnityEnvironment(file_name = 'CoE202')

env.reset()
purple_team = list(env.behavior_specs)[0]
blue_team = list(env.behavior_specs)[1]

max_timestep = 10000
__ = 0

while __ < input_n:
    #input_n번의 경기만큼 실행
    env.reset()
    score_p = 0
    score_b = 0
    Agent1_p = Agent_dis(0)
    Agent2_p = Agent_ball(1)
    for i in range(max_timestep):
        decision_steps_p, terminal_steps_p = env.get_steps(purple_team)
        decision_steps_b, terminal_steps_b = env.get_steps(blue_team)
        done = False
        score_p = get_score_from_decision_step(decision_steps_p)
        score_b = get_score_from_decision_step(decision_steps_b)
        if  score_p != 0 or score_b != 0:
            done = True

        action_p = np.array([Agent1_p.decide_movement(),Agent2_p.decide_movement()])
        env.set_actions(purple_team , action_p)

        if done:
            break
        env.step()
    #Logging
    print("Loop. {} ended with score_p : {} / score_b : {}".format(__,score_p,score_b))
    __ += 1