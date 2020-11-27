import numpy as np
import random
import torch
import torch.nn as nn
from torch.distributions import Categorical
import gym

from mlagents_envs.environment import UnityEnvironment


# 연산 base
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# followed by https://github.com/nikhilbarhate99/PPO-PyTorch
# helped by
# https://towardsdatascience.com/proximal-policy-optimization-tutorial-part-1-actor-critic-method-d53f9afffbf6

# 액션, 상태, logprobs(?), reward, is_terminals(?) 저장 및 삭제
class Memory:
    def __init__(self):
        self.actions = []
        self.states = []
        self.logprobs = []
        self.rewards = []
        self.is_terminals = []

    def clear_memory(self):
        del self.actions[:]
        del self.states[:]
        del self.logprobs[:]
        del self.rewards[:]
        del self.is_terminals[:]

class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim, n_latent_var):
        super(ActorCritic, self).__init__()
        # input_dimension : 1 * 3 * 28 * 8
        # n_latent_var : 연산에 대한 매개변수?, Softmax로 가장 큰 값을 가지는 것을 action
        # actor
        self.action_layer = nn.Sequential(
            nn.Conv2d(in_channels = 3, out_channels = 6, kernel_size = (5,2) , stride = 1, padding = 0, padding_mode = 'zeros'),
            nn.Tanh(), nn.ReLU(),
            nn.Conv2d(in_channels = 6, out_channels=4, kernel_size=(6, 2), stride=1, padding=0, padding_mode='zeros'),
            nn.Tanh(), nn.ReLU(),
            nn.Conv2d(in_channels= 4, out_channels=3, kernel_size=(6, 2), stride=1, padding=0, padding_mode='zeros'),
            nn.Tanh(), nn.ReLU(),
            nn.Conv2d(in_channels=3, out_channels=2, kernel_size=(5, 2), stride=1, padding=0, padding_mode='zeros'),
            nn.Tanh(), nn.ReLU(),
            nn.Conv2d(in_channels=2, out_channels=1, kernel_size=(5, 2), stride=1, padding=0, padding_mode='zeros'),
            nn.Tanh(), nn.ReLU()
        )

        # critic, 수정 필요
        self.value_layer = nn.Sequential(
            nn.Linear(state_dim, n_latent_var),
            nn.Tanh(),
            nn.Linear(n_latent_var, n_latent_var),
            nn.Tanh(),
            nn.Linear(n_latent_var, 1)
        )

    def forward(self):
        raise NotImplementedError

    def act(self, state, memory):
        state = torch.tensor(state).float().to(device)
        action_probs = self.action_layer(state)
        action_probs = action_probs.tolist()
        action_probs = action_probs[0]
        action_probs = action_probs[0]
        for i in range(6):
            for j in range(3):
                action_probs[i][j] = action_probs[i][j] + 0.1
        # Normalize
        action_list = []
        for i in range(6):
            acti = []
            summ = 0
            for j in range(3):
                summ = summ + action_probs[i][j]
            for j in range(3):
                action_probs[i][j] = action_probs[i][j] / summ
            acti = action_probs[i]
            acti = torch.tensor(acti)
            dist = Categorical(acti)
            actio = dist.sample()
            actio = actio.tolist()
            action_list.append(actio)

        action = []
        l = 0
        for i in range(2):
            act_set = []
            for j in range(3):
                act_set.append(action_list[j+l])
            l = 3
            action.append(act_set)
        action = torch.tensor(action)



        memory.states.append(state)
        memory.actions.append(action)
        memory.logprobs.append(dist.log_prob(action))

        return action

    def evaluate(self, state, action):
        state = state.clone().float().to(device)#torch.tensor(state).float().to(device)
        action_probs = self.action_layer(state)
        action_probs = action_probs.tolist()
        action_probs = action_probs[0]
        action_probs = action_probs[0]
        for i in range(6):
            for j in range(3):
                action_probs[i][j] = action_probs[i][j] + 0.1
        # Normalize
        action_list = []
        act_log = []
        state_val = []
        ent = []

        for i in range(6):
            acti = []
            summ = 0
            for j in range(3):
                summ = summ + action_probs[i][j]
            for j in range(3):
                action_probs[i][j] = action_probs[i][j] / summ
            acti = action_probs[i]
            acti = torch.tensor(acti)
            dist = Categorical(acti)
            act_log.append(dist.log_prob(action))
            dist_entropy = dist.entropy()
            ent.append(dist_entropy)
            state_value = self.value_layer(state)
            state_val.append(torch.squeeze(state_value))


        return act_log, state_val, ent


class PPO:
    def __init__(self, state_dim, action_dim, n_latent_var, lr, betas, gamma, K_epochs, eps_clip):
        self.lr = lr
        self.betas = betas
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.K_epochs = K_epochs

        self.policy = ActorCritic(state_dim, action_dim, n_latent_var).to(device)
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=lr, betas=betas)
        self.policy_old = ActorCritic(state_dim, action_dim, n_latent_var).to(device)
        self.policy_old.load_state_dict(self.policy.state_dict())

        self.MseLoss = nn.MSELoss()

    def update(self, memory):
        # Monte Carlo estimate of state rewards:
        rewards = []
        discounted_reward = 0
        for reward, is_terminal in zip(reversed(memory.rewards), reversed(memory.is_terminals)):
            if is_terminal:
                discounted_reward = 0
            discounted_reward = reward + (self.gamma * discounted_reward)
            rewards.insert(0, discounted_reward)

        # Normalizing the rewards:
        rewards = torch.tensor(rewards, dtype=torch.float32).to(device)
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-5)

        # convert list to tensor
        old_states = torch.stack(memory.states).to(device).detach()
        old_actions = torch.stack(memory.actions).to(device).detach()
        old_logprobs = torch.stack(memory.logprobs).to(device).detach()

        # Optimize policy for K epochs:
        for _ in range(self.K_epochs):
            # Evaluating old actions and values :
            logprobs, state_values, dist_entropy = self.policy.evaluate(old_states, old_actions)

            # Finding the ratio (pi_theta / pi_theta__old):
            ratios = torch.exp(logprobs - old_logprobs.detach())

            # Finding Surrogate Loss:
            advantages = rewards - state_values.detach()
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages
            loss = -torch.min(surr1, surr2) + 0.5 * self.MseLoss(state_values, rewards) - 0.01 * dist_entropy

            # take gradient step
            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()

        # Copy new weights into old policy:
        self.policy_old.load_state_dict(self.policy.state_dict())

def add_sensor_sig(a, b, c, d):
    Final_signal = []
    F = []
    for num in range(3):
        aaa = a[num][0:][0:]
        bbb = b[num][0:][0:]
        ccc = c[num][0:][0:]
        ddd = d[num][0:][0:]

        active = []

        for k in range(len(aaa)):
            active.append(aaa[k][0:])
        for k in range(len(bbb)):
            active.append(bbb[k][0:])
        for k in range(len(ccc)):
            active.append(ccc[k][0:])
        for k in range(len(ddd)):
            active.append(ddd[k][0:])
        Final_signal.append(active)
    F.append(Final_signal)
    return F

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
def main():
    ###### Check about all parameters ######
    ###### Hypebeast ######
    env = UnityEnvironment(file_name='CoE202')
    env.reset()
    purple_team = list(env.behavior_specs)[0]
    blue_team = list(env.behavior_specs)[1]
    # print('b_n',env.brains)
    decision_steps_p, _ = env.get_steps(purple_team)  # purple_team
    decision_steps_b, _ = env.get_steps(blue_team)  # blue_team
    cur_obs_b = decision_steps_b.obs[0][0, :]

    state_dim = 1
    action_dim = 1
    solved_reward = 10
    log_interval = 20
    max_episodes = 50000
    max_timesteps = 500
    n_latent_var = 1
    update_timestep = 2000
    lr = 0.002
    betas = (0.9, 0.999)
    gamma = 0.99
    K_epochs = 4
    eps_clip = 0.2

    memory_b = Memory()
    memory_p = Memory()

    ppo_1 = PPO(state_dim, action_dim, n_latent_var, lr, betas, gamma, K_epochs, eps_clip)
    ppo_2 = PPO(state_dim, action_dim, n_latent_var, lr, betas, gamma, K_epochs, eps_clip)

    running_reward_p = 0
    avg_length = 0
    timestep = 0

    #######################
    for i_episode in range(max_episodes):
        env.reset()
        decision_steps_p, terminal_steps_p = env.get_steps(purple_team)
        decision_steps_b, terminal_steps_b = env.get_steps(blue_team)

        cur_obs_b_1 = decision_steps_b.obs[0][0, :]
        cur_obs_b_2 = decision_steps_b.obs[0][1, :]
        cur_obs_p_1 = decision_steps_p.obs[0][0, :]
        cur_obs_p_2 = decision_steps_p.obs[0][1, :]

        # ball, our goal post, opposite goal post, wall, our team, enemy, null(?), distance
        # order by left to right : 10, 8, 6, 4, 2, 0, 1, 3, 5, 7, 9
        # order by left to right back : 1, 0, 2

        signal_b_1 = sensor_front_sig(decision_steps_b.obs[0][0, :])  # front_signal before 3 steps
        signal_b_2 = sensor_front_sig(decision_steps_b.obs[0][1, :])
        signal_back_b_1 = sensor_back_sig(decision_steps_b.obs[1][0, :])
        signal_back_b_2 = sensor_back_sig(decision_steps_b.obs[1][1, :])  # back_signal before 3 steps
        state_b = add_sensor_sig(signal_b_1, signal_back_b_1, signal_b_2, signal_back_b_2)
        state_b = np.array(state_b)

        signal_p_1 = sensor_front_sig(decision_steps_p.obs[0][0, :])  # front_signal before 3 steps
        signal_p_2 = sensor_front_sig(decision_steps_p.obs[0][1, :])
        signal_back_p_1 = sensor_back_sig(decision_steps_p.obs[1][0, :])
        signal_back_p_2 = sensor_back_sig(decision_steps_p.obs[1][1, :])  # back_signal before 3 steps
        state_p = add_sensor_sig(signal_p_1, signal_back_p_1, signal_p_2, signal_back_p_2)
        state_p = np.array(state_p)

        for t in range(max_timesteps):
            timestep += 1
            # ppo_1 : purple, ppo_2 = blue
            signal_b_1 = sensor_front_sig(decision_steps_b.obs[0][0, :])  # front_signal before 3 steps
            signal_b_2 = sensor_front_sig(decision_steps_b.obs[0][1, :])
            signal_back_b_1 = sensor_back_sig(decision_steps_b.obs[1][0, :])
            signal_back_b_2 = sensor_back_sig(decision_steps_b.obs[1][1, :])  # back_signal before 3 steps
            state_b = add_sensor_sig(signal_b_1, signal_back_b_1, signal_b_2, signal_back_b_2)
            state_b = np.array(state_b)

            signal_p_1 = sensor_front_sig(decision_steps_p.obs[0][0, :])  # front_signal before 3 steps
            signal_p_2 = sensor_front_sig(decision_steps_p.obs[0][1, :])
            signal_back_p_1 = sensor_back_sig(decision_steps_p.obs[1][0, :])
            signal_back_p_2 = sensor_back_sig(decision_steps_p.obs[1][1, :])  # back_signal before 3 steps
            state_p = add_sensor_sig(signal_p_1, signal_back_p_1, signal_p_2, signal_back_p_2)
            state_p = np.array(state_p)

            act_p = ppo_1.policy_old.act(state_p, memory_p)
            act_p_sp = act_p.reshape(6)

            # state_p, reward_p, done_b, _ = env.step(act_p_sp)
            act_b = ppo_2.policy_old.act(state_b, memory_b)
            act_b_sp = act_b.reshape(6)
            print('act_p_sp : ', act_p_sp)

            # Set the actions
            env.set_actions(purple_team, np.array(act_p))
            env.set_actions(blue_team, np.array(act_b))
            # Move the simulation forward
            env.step()

            decision_steps_p, terminal_steps_p = env.get_steps(purple_team)
            decision_steps_b, terminal_steps_b = env.get_steps(blue_team)
          #  print(decision_steps_b.reward)
          #  print(terminal_steps_p.reward)
          #  print(terminal_steps_b.reward)
          #  print(decision_steps_b.obs)
            reward_b = 0
            reward_p = 0
            done = False

            if not decision_steps_b:
                done = True
                reward_b = terminal_steps_b.reward[0]
                reward_p = terminal_steps_p.reward[0]
            # state_p, state_b

            memory_b.rewards.append(reward_b)
            memory_p.rewards.append(reward_p)
            memory_b.is_terminals.append(done)
            memory_p.is_terminals.append(done)
            #print(memory_b.rewards)
            #print(memory_b.is_terminals)
            #print(memory_b.rewards)
            if timestep % update_timestep == 0:		#memory 오타 수정
                ppo_1.update(memory_b)
                ppo_2.update(memory_p)
                memory_p.clear_memory()
                memory_b.clear_memory()
                timestep = 0

            running_reward_p += reward_p

            if done:
                break
        avg_length += t

        if running_reward_p > (log_interval * solved_reward):
            print("########## PPO_1 has scored over solved_reward! ##########")
            torch.save(ppo_1.policy.state_dict(), './PPO_1_{}.pth'.format(env_name))
            break

            # logging
        if i_episode % log_interval == 0:
            avg_length = int(avg_length / log_interval)
            running_reward_p = int((running_reward_p / log_interval))

            print('Episode {} \t avg length: {} \t reward: {}'.format(i_episode, avg_length, running_reward_p))
            running_reward_p = 0
            avg_length = 0

    env.close()

if __name__ == '__main__':
    main()
