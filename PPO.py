import torch
import torch.nn as nn
from torch.distributions import Categorical
from mlagents_envs.environment import UnityEnvironment
import numpy as np

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

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
        
    def forward(self):
        raise NotImplementedError
        
    def act(self, state, memory):
        state = torch.from_numpy(state).float().to(device) 
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
            surr2 = torch.clamp(ratios, 1-self.eps_clip, 1+self.eps_clip) * advantages
            loss = -torch.min(surr1, surr2) + 0.5*self.MseLoss(state_values, rewards) - 0.01*dist_entropy
            
            # take gradient step
            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()
        
        # Copy new weights into old policy:
        self.policy_old.load_state_dict(self.policy.state_dict())

#TRAIN CODE -> SHOULD MODIFY THIS CODE FOR DEVELOPING PPO ALGORITHM FOR SOCCER PROJECT.
def action_int_to_action(n):
    temp = n
    tl_1= []; tl_2 = []
    for i in range(3):
        val = temp%3
        tl_1.append(val)
        temp = int(temp//3)
    for i in range(3):
        val = temp % 3
        tl_2.append(val)
        temp = int(temp // 3)
    return np.array([tuple(tl_1), tuple(tl_2)])


def main():
    ############## Hyperparameters ##############
    env_name = "SoccerProject"
    # creating environment
    env = UnityEnvironment(file_name='CoE202')
    env.reset()
    behavior_name_1 = list(env.behavior_specs)[0]
    behavior_name_2 = list(env.behavior_specs)[1]

    # State Dimension, The first Linear layer will be made to be fit to this dimension.
    # ---Below code should be changed---#
    state_dim =264
    action_dim = 27*27
    # ---------------------------------#

    #Not used for soccer_PPO
    render = False


    solved_reward = 5         # stop training if avg_reward > solved_reward
    log_interval = 20           # print avg reward in the interval

    max_episodes = 50000        # max training episodes
    max_timesteps = 3000         # max timesteps in one episode
    n_latent_var = 64*8           # number of variables in hidden layer
    update_timestep = 2000      # update policy every n timesteps
    lr = 0.002
    betas = (0.9, 0.999)
    gamma = 0.99                # discount factor
    K_epochs = 4                # update policy for K epochs
    eps_clip = 0.2              # clip parameter for PPO
    #random_seed = None
    #############################################
    
    '''if random_seed:
        torch.manual_seed(random_seed)
        #Should check manual and modify.
        # env.seed(random_seed)'''

    #only ppo_1 trained. ppo_2 is just opponent.
    memory_1 = Memory()
    memory_2 = Memory()
    ppo_1 = PPO(state_dim, action_dim, n_latent_var, lr, betas, gamma, K_epochs, eps_clip)
    ppo_2 = PPO(state_dim, action_dim, n_latent_var, lr, betas, gamma, K_epochs, eps_clip)
    print(lr,betas)
    
    # logging variables
    running_reward_1 = 0
    avg_length_1 = 0
    running_reward_2 = 0
    avg_length_2 = 0

    timestep = 0
    
    # training loop
    for i_episode in range(1, max_episodes+1):
        env.reset()
        decision_steps_p, _ = env.get_steps(behavior_name_1)
        decision_steps_b, _ = env.get_steps(behavior_name_2)
        cur_obs_b = decision_steps_b.obs[0][0, :]
        cur_obs_p = decision_steps_p.obs[0][0, :]
        #print(cur_obs_p)
        #print(len(cur_obs_p))
        state_1 =cur_obs_b
        state_2 =cur_obs_p
        for t in range(max_timesteps):

            timestep += 1
            
            # Running policy_old:
            action_1 = ppo_1.policy_old.act(state_1, memory_1)
            action_2 = ppo_2.policy_old.act(state_2, memory_2)

            # set actions
            env.set_actions(behavior_name_1, np.array(action_int_to_action(action_1)))
            env.set_actions(behavior_name_2, np.array(action_int_to_action(action_1)))
            #env.set_actions(behavior_name_1, np.array([(0, 0, 0), (0, 0, 0)]))
            #env.set_actions(behavior_name_2, np.array([(0, 0, 0), (0, 0, 0)]))

            # step(Need to change how to get reward and state, done variable.)
            ##########step########################
            env.step()
            decision_steps_p, terminal_steps_p = env.get_steps(behavior_name_1)
            decision_steps_b, terminal_steps_b = env.get_steps(behavior_name_2)

            cur_obs_b = decision_steps_b.obs[0][0, :]
            cur_obs_p = decision_steps_p.obs[0][0, :]
            state_1 = cur_obs_b
            state_2 = cur_obs_p
            reward = 0
            done = False
            #state, reward, done, _ = env.step(action)
            ##########################################

            # Saving reward and is_terminal:
            memory_1.rewards.append(reward)
            memory_1.is_terminals.append(done)
            
            # update if its time
            if timestep % update_timestep == 0:
                ppo_1.update(memory_1)
                memory_1.clear_memory()
                timestep = 0
            
            running_reward_1 += reward

            if done:
                break
                
        avg_length_1 += t
        
        # stop training if avg_reward > solved_reward
        if running_reward_1 > (log_interval*solved_reward):
            print("########## PPO_1 has scored over solved_reward! ##########")
            torch.save(ppo_1.policy.state_dict(), './PPO_1_{}.pth'.format(env_name))
            break
            
        # logging
        if i_episode % log_interval == 0:
            avg_length = int(avg_length/log_interval)
            running_reward = int((running_reward/log_interval))
            
            print('Episode {} \t avg length: {} \t reward: {}'.format(i_episode, avg_length, running_reward))
            running_reward = 0
            avg_length = 0
            
if __name__ == '__main__':
    main()
    
