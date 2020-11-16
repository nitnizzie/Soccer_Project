
"""

"""

"""

for episode in range(num_episodes):
   
    agent_b.reset()
    agent_p.reset()
    
    #Receive Initial Observation state.
    decision_steps_p, terminal_steps_p = env.get_steps(behavior_name_1)
    decision_steps_b, terminal_steps_b = env.get_steps(behavior_name_2)
 
    front_b1, back_b1, front_b2, back_b2, reward_b = step(decision_steps_b)
    front_p1, back_p1, front_p2, back_p2, reward_p = step(decision_steps_p)
	
    for step in range(num_steps):

        #select action
        action_b = agent_b.select_action(state, step)
        action_p = agent_b.select_action(state, step)

        #Execute action a_t
        env.set_actions(behavior_name_1, np.array([(1,0,0),(2,0,0)])) #p
        env.set_actions(behavior_name_2, np.array([(0,0,1),(0,1,0)])) #b
        env.step()

        #Observe reward r_t and next state s_(t+1)
        decision_steps_p, terminal_steps_p = env.get_steps(behavior_name_1)
        decision_steps_b, terminal_steps_b = env.get_steps(behavior_name_2)
        
        front_b1, back_b1, front_b2, back_b2, reward_b = step(decision_steps_b)
        front_p1, back_p1, front_p2, back_p2, reward_p = step(decision_steps_p)

        #Store Transition to Memory
        agent_b.store_transtion(state, action, next_state, reward)
        agent_p.store_transtion(state, action, next_state, reward)

        # 
        agent_b.train()
        agent_p.train()


env.close()
"""