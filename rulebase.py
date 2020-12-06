import numpy as np
from mlagents_envs.environment import UnityEnvironment
import random
from utils import sensor_front_sig, sensor_back_sig

def get_score_from_decision_step(decision_steps):
    score = decision_steps.reward[0]
    return score

def decide_direction(i):
    if i == 0:
        #front
        return 0
    if i%2 == 0:
        #left
        return -1
    if i%2 == 1:
        #right
        return 1

class Agent_dis():
    def __init__(self,index):
        self.find_enemy = False
        self.distance_enemy = 0
        self.direction_enemy = 0
        #Only turns on when collison to wall occured.
        self.must_find_enemy = True
        self.index = index
        self.backward_steps = 0
        self.collison_threshold = 2
        self.temp=0
        self.stepper = 0

    def reset(self):
        self.find_enemy = False
        self.distance_enemy = 0
        self.direction_enemy = 0
        # Only turns on when collison to wall occured.
        self.must_find_enemy = True
        self.backward_steps = 0
        self.collison_threshold = 2
        self.stepper= 0

    def accel_true_by_velocity(self,velocity_tuple):
        if len(velocity_tuple) != 2:
            raise TypeError('Velocity Tuple must have 2 items. numerator, denominator')
        v_dec_temp_var = self.stepper%velocity_tuple[1]
        if v_dec_temp_var < velocity_tuple[0]:
            return 1
        return 0

    def update_enemy(self,dec_step_obs):
        self.must_find_enemy = not (self.find_enemy)
        sensor_front = sensor_front_sig(dec_step_obs[0][self.index,:])[0] #(11, 8))
        sensor_back = sensor_back_sig(dec_step_obs[1][self.index,:])[0] #(3, 8))
        if self.must_find_enemy:
            for i in range(11):
                print(sensor_front[i][5])
                if sensor_front[i][5] == 1:
                    #found enemy
                    self.find_enemy = True
                    self.distance_enemy = sensor_front[i][7]
                    self.direction_enemy = decide_direction(i)
                    #print("FOUND / ",i)
                    self.temp=0
                    break
            #if not found
            return True
        else:
            for i in range(11):
                if sensor_front[i][5] == 1:
                    #found enemy
                    if self.distance_enemy < sensor_front[i][7]:
                        self.find_enemy = True
                        self.distance_enemy = sensor_front[i][7]
                        self.direction_enemy = decide_direction(i)
                        #print("FOUND / ",i)
                        self.temp=0
                        break
                #if sensor_front [i][3] == 1 and sensor_front[i][7]>self.collison_threshold:
                    #pass
            if self.temp >= 30:
                #COLLISON
                self.find_enemy = False
                self.backward_steps = 10
                self.temp=0
            self.temp += 1
        return False

    def decide_movement(self,dec_step_obs):
        #print("AGENT_DIS/dir_enemy",self.direction_enemy)
        #print(self.find_enemy)
        up_bool = self.update_enemy(dec_step_obs)
        v = self.accel_true_by_velocity((1,4))
        if not self.find_enemy:
            #print("NOT FOUND")
            candidates = [0, 1, 2]
            return (v, random.choice(candidates), random.choice(candidates))
        else:
            if self.backward_steps>0:
                self.backward_steps-=1
                return(2*v,0,0)
            if up_bool:
                return (0,0,1)
            if self.direction_enemy == 0:
                return (1*v,0,0)
            if self.direction_enemy == -1:
                return (1*v,0,1)
            if self.direction_enemy == 1:
                return (1*v,0,2)
        return (1*v,0,0)

class Agent_ball():
    def __init__(self,index):
        self.find_ball = False
        self.distance_ball = 0
        self.direction_ball = 0
        self.not_detected_for = 0
        self.distance_threshold = 0.5
        self.threshold = 20
        self.index = index
        self.stepper = 0
        self.shoot_dir_boolean = True
        self.protocol_running = False
        self.protocol_start_stepper = 0
        self.protocol_len = 10

    def accel_true_by_velocity(self, velocity_tuple):
        if len(velocity_tuple) != 2:
            raise TypeError('Velocity Tuple must have 2 items. numerator, denominator')
        v_dec_temp_var = self.stepper % velocity_tuple[1]
        if v_dec_temp_var <= velocity_tuple[0]:
            return 1
        return 0
    def search(self,sensor_front, sensor_back):
        for i in range(11):
            if sensor_front[i][0] == 1:
                distance = sensor_front[i][7]
                is_detected = True
                if i == 0:
                    #Front
                    direction = 0
                elif i%2 == 0:
                    #Left
                    direction = -1
                else:
                    #Right
                    direction = 1
                return is_detected, direction, distance

        for j in range(3):
            if sensor_back[j][0] == 1:
                distance = sensor_back[j][7]
                is_detected = True
                if i== 0:
                    #Front
                    direction = 0
                elif i== 1:
                    #Left
                    direction = -1
                else:
                    #Right
                    direction = 1
                return is_detected, direction, distance
        return False,0,0
    def search_op_goal(self,sensor_front,sensor_back):
        for i in range(11):
            if sensor_front[i][2] == 1:
                is_detected = True
                #FRONT DIRECTION
                direction = 1
                return is_detected, direction

        for j in range(3):
            if sensor_back[j][2] == 1:
                is_detected = True
                direction = 1
                return is_detected, direction
        return False, 0
    def search_our_goal(self,sensor_front,sensor_back):
        for i in range(11):
            if sensor_front[i][1] == 1:
                is_detected = True
                # FRONT DIRECTION
                direction = 1
                return is_detected, direction

        for j in range(3):
            if sensor_back[j][1] == 1:
                is_detected = True
                direction = 1
                return is_detected, direction
        return False, 0

    def update_ball(self,sensor_front,sensor_back):
        is_detected, direction, distance = self.search(sensor_front, sensor_back)
        is_op_goal_det , dir_op_goal = self.search_op_goal(sensor_front, sensor_back)
        is_our_goal_det, dir_our_goal =self.search_our_goal(sensor_front, sensor_back)
        if is_detected:
            self.find_ball = True
            self.not_detected_for=0
            #Next is record the direction and distance to class.
            self.distance_ball = distance
            self.direction_ball = direction
        else:
            self.not_detected_for+=1
        if self.not_detected_for > self.threshold:
            self.find_ball = False
        if (is_op_goal_det and dir_op_goal == 1) or (is_our_goal_det and dir_our_goal == -1):
            #To Forward! Shoot at present direction
            self.shoot_dir_boolean = True
        if (is_op_goal_det and dir_op_goal == -1) or (is_our_goal_det and dir_our_goal == 1):
            #To Backward!
            self.shoot_dir_boolean = False
        return

    def decide_movement(self,dec_step_obs):
        sensor_front = sensor_front_sig(dec_step_obs[0][self.index, :])[0]  # (11, 8))
        sensor_back = sensor_back_sig(dec_step_obs[1][self.index, :])[0]  # (3, 8))
        v = self.accel_true_by_velocity((1, 6))
        self.update_ball(sensor_front, sensor_back)
        if self.protocol_running:
            delta_step = self.stepper - self.protocol_start_stepper
            #Specify protocol

            ################
            if delta_step > self.protocol_len:
                self.protocol_running = False
                self.shoot_dir_boolean = True
        rot_const = 0
        if self.find_ball:
            if self.distance_ball < self.distance_threshold:
                if not self.shoot_dir_boolean:
                    self.protocol_running = True
                    return (0,0,0)
                else:
                    #Shoot!
                    v = 1
            if self.direction_ball == -1:
                rot_const = 1
            elif self.direction_ball == 1:
                rot_const = 2
            else:
                rot_const = 0
            return (v,0,rot_const)
        candidates = [0,1,2]
        return (v,random.choice(candidates),random.choice(candidates))

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
        a1_dec = Agent1_p.decide_movement(decision_steps_p.obs)
        a2_dec = Agent2_p.decide_movement(decision_steps_p.obs)
        #print(a1_dec,a2_dec)
        action_p = np.array([a1_dec,a2_dec])
        #action_p = np.array([a1_dec, (0,0,0)])
        env.set_actions(purple_team , action_p)

        if done:
            break
        env.step()
        Agent1_p.stepper+=1
        Agent2_p.stepper+=1
    #Logging
    print("Loop. {} ended with score_p : {} / score_b : {}".format(__,score_p,score_b))
    __ += 1