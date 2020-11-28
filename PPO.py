import torch
import torch.nn as nn
from torch.distributions import Categorical

import random
import numpy as np

from utils import step, change_action_shape
from mlagents_envs.environment import UnityEnvironment


# followed by https://github.com/nikhilbarhate99/PPO-PyTorch
# helped by
# https://towardsdatascience.com/proximal-policy-optimization-tutorial-part-1-actor-critic-method-d53f9afffbf6


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
    def __init__(self, state_dim, action_dim, n_latent_var, device="cpu"):
        '''
        input shape : (N, 3*14*3)
        output shape : (N, 3*3*3)
        '''
        super(ActorCritic, self).__init__()

        # actor
        self.action_layer = nn.Sequential(
                nn.Linear(state_dim, n_latent_var),
                nn.Tanh(),
                nn.Linear(n_latent_var, n_latent_var),
                nn.Tanh(),
                nn.Linear(n_latent_var, action_dim),
                nn.Softmax(dim=-1)
                ) 

        # critic
        self.value_layer = nn.Sequential(
            nn.Linear(state_dim, n_latent_var),
            nn.Tanh(),
            nn.Linear(n_latent_var, n_latent_var),
            nn.Tanh(),
            nn.Linear(n_latent_var, 1)
        )

        self.device = device

    def forward(self):
        raise NotImplementedError

    def act(self, state, memory):
        state = torch.from_numpy(state).float().to(self.device)

        state = torch.reshape(state, (1, -1))
        #print(state.shape)   #(1, 336)

        action_probs = self.action_layer(state)
        dist = Categorical(action_probs)
        action = dist.sample()
        
        memory.states.append(state)
        memory.actions.append(action)
        memory.logprobs.append(dist.log_prob(action))

        return action.item()

    def evaluate(self, state, action):
        action_probs = self.action_layer(state)
        dist = Categorical(action_probs)
        
        action_logprobs = dist.log_prob(action)
        dist_entropy = dist.entropy()
        
        state_value = self.value_layer(state)
        
        return action_logprobs, torch.squeeze(state_value), dist_entropy


class PPO:
    def __init__(self, state_dim, action_dim, n_latent_var, lr, betas, gamma, K_epochs, eps_clip, device="cpu"):
        self.lr = lr
        self.betas = betas
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.K_epochs = K_epochs

        self.policy = ActorCritic(state_dim, action_dim, n_latent_var, device).to(device)
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=lr, betas=betas)
        self.policy_old = ActorCritic(state_dim, action_dim, n_latent_var, device).to(device)
        self.policy_old.load_state_dict(self.policy.state_dict())

        self.MseLoss = nn.MSELoss()

        self.device = device

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
        rewards = torch.tensor(rewards, dtype=torch.float32).to(self.device)
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-5)

        # convert list to tensor
        old_states = torch.stack(memory.states).to(self.device).detach()
        old_actions = torch.stack(memory.actions).to(self.device).detach()
        old_logprobs = torch.stack(memory.logprobs).to(self.device).detach()

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


def main():
    
    #set GPU for faster training
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("device ", device, " is available")

    #set Unity Env
    env = UnityEnvironment(file_name='CoE202')
    env.reset()


    #set Hyperparameters
    state_dim = 3 * 14 * 8
    action_dim = 3 * 3 * 3
    solved_reward = 10 #230     # stop training if avg_reward > solved_reward
    log_interval = 20           # print avg reward in the interval
    max_episodes = 50000        # max training episodes
    max_timesteps = 300         # max timesteps in one episode
    n_latent_var = 64           # number of variables in hidden layer
    update_timestep = 500       # update policy every n timesteps
    lr = 0.002
    betas = (0.9, 0.999)
    gamma = 0.99                # discount factor
    K_epochs = 4                # update policy for K epochs
    eps_clip = 0.2              # clip parameter for PPO


    purple_team = list(env.behavior_specs)[0]
    blue_team = list(env.behavior_specs)[1]

    #Initialize memory
    memory_p = Memory()
    memory_b = Memory()

    ppo_p = PPO(state_dim, action_dim, n_latent_var, lr, betas, gamma, K_epochs, eps_clip, device)
    ppo_b = PPO(state_dim, action_dim, n_latent_var, lr, betas, gamma, K_epochs, eps_clip, device)

    #logging variables
    running_reward_p = 0
    running_reward_b = 0
    avg_length = 0
    timestep = 0


    #training loop
    for i_episode in range(max_episodes):
        env.reset()

        decision_steps_p, terminal_steps_p = env.get_steps(purple_team)
        decision_steps_b, terminal_steps_b = env.get_steps(blue_team)

        state_p1, state_p2, reward_p, done = step(decision_steps_p, terminal_steps_p)
        state_b1, state_b2, reward_b, _ = step(decision_steps_b, terminal_steps_b)

        #print(state_b1.shape, reward_b)
        
        for t in range(max_timesteps):
            timestep += 1

            #running policy_old:
            act_p1 = ppo_p.policy_old.act(state_p1, memory_p)
            act_p2 = ppo_p.policy_old.act(state_p2, memory_p)
            
            act_b1 = ppo_p.policy_old.act(state_b1, memory_b)
            act_b2 = ppo_p.policy_old.act(state_b2, memory_b)

            # Set the actions
            action_p = change_action_shape(act_p1, act_p2)
            action_b = change_action_shape(act_b1, act_b2)
            #print(act_b1, act_b2)
            #print(action_b)

            env.set_actions(purple_team, np.array(action_p))
            env.set_actions(blue_team, np.array(action_b))

            # Move the simulation forward
            env.step()

            #Observe state, reward
            decision_steps_p, terminal_steps_p = env.get_steps(purple_team)
            decision_steps_b, terminal_steps_b = env.get_steps(blue_team)

            state_p1, state_p2, reward_p, done = step(decision_steps_p, terminal_steps_p)
            state_b1, state_b2, reward_b, _ = step(decision_steps_b, terminal_steps_b)

            # Saving reward and is_terminal:
            memory_p.rewards.append(reward_p)
            memory_p.rewards.append(reward_p)
            memory_p.is_terminals.append(done)
            memory_p.is_terminals.append(done)

            memory_b.rewards.append(reward_b)
            memory_b.rewards.append(reward_b)
            memory_b.is_terminals.append(done)
            memory_b.is_terminals.append(done)

            # update if its time
            if timestep % update_timestep == 0:
                ppo_p.update(memory_p)
                ppo_b.update(memory_b)
                memory_p.clear_memory()
                memory_b.clear_memory()
                timestep = 0

            running_reward_p += reward_p
            running_reward_b += reward_b

            if done:
                break

        avg_length += t

        # stop training if avg_reward > solved_reward
        if running_reward_p > (log_interval * solved_reward):
            print("########## PPO_p has scored over solved_reward! ##########")
            torch.save(ppo_p.policy.state_dict(), './PPO_p_{}.pth'.format("CoE202"))
            break

        if running_reward_b > (log_interval * solved_reward):
            print("########## PPO_b has scored over solved_reward! ##########")
            torch.save(ppo_b.policy.state_dict(), './PPO_b_{}.pth'.format("CoE202"))
            break

        # logging
        if i_episode % log_interval == 0:
            avg_length = int(avg_length / log_interval)
            running_reward_p = int((running_reward_p / log_interval))
            running_reward_b = int((running_reward_p / log_interval))

            print('Episode {} \t avg length: {} \t reward_p: {} \t reward_b: {}'.format(i_episode, avg_length, running_reward_p, running_reward_b))
            running_reward_p = 0
            running_reward_b = 0
            avg_length = 0

    env.close()

if __name__ == '__main__':
    main()
