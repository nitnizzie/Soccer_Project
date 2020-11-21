from mlagents_envs.environment import UnityEnvironment
from PPO import PPO, Memory
from PIL import Image

import torch

def test():
    ############## Hyperparameters ##############
    env_name = "SoccerProject"
    # creating environment
    env = UnityEnvironment(file_name = 'CoE202')

    #State Dimension, The first Linear layer will be made to be fit to this dimension.
    #---Below code should be changed---#
    state_dim = env.observation_space.shape[0]
    action_dim = 27
    # ---------------------------------#

    n_latent_var = 64*8           # number of variables in hidden layer
    lr = 0.0007
    betas = (0.9, 0.999)
    gamma = 0.99                # discount factor
    K_epochs = 4                # update policy for K epochs
    eps_clip = 0.2              # clip parameter for PPO
    #############################################

    #Episode & Timesteps
    n_episodes = 3
    max_timesteps = 300

    #Associated with Model Save&Load
    filename = "PPO_{}.pth".format(env_name)
    directory = "./preTrained/"
    
    memory = Memory()
    ppo = PPO(state_dim, action_dim, n_latent_var, lr, betas, gamma, K_epochs, eps_clip)
    
    ppo.policy_old.load_state_dict(torch.load(directory+filename))
    
    for ep in range(1, n_episodes+1):
        ep_reward = 0
        state = env.reset()
        for t in range(max_timesteps):
            action = ppo.policy_old.act(state, memory)

            #set actions
            env.set_actions(behavior_name_1, np.array([(0, 0, 0), (0, 0, 0)]))
            env.set_actions(behavior_name_2, np.array([(0, 0, 0), (0, 0, 0)]))

            #step(Need to change how to get reward and state, done variable.)
            state, reward, done, _ = env.step(action)

            ep_reward += reward
            if done:
                break

        #print reward for that episode.
        print('Episode: {}\tReward: {}'.format(ep, int(ep_reward)))
        #Reset episode reward.
        ep_reward = 0

        #Close Environment
        env.close()
    
if __name__ == '__main__':
    test()
    
    
