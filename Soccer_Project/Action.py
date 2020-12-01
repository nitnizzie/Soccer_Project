import random

def lrway(sensor):
    if sensor % 2 == 0:
        return 1
    if sensor % 2 == 1:
        return 2
    if sensor == 0:
        return 0

max_timesteps = 1000
ball_atk = [[0 for _ in range(max_timesteps)] for _ in range(2)]
ball_def = [[0 for _ in range(max_timesteps)] for _ in range(2)]
ourgp_def = [[0 for _ in range(max_timesteps)] for _ in range(2)]
limit = [0, 0]

def attack(agent, delay, t, state):
    ball_atk[agent].insert(t, 2)

    # max time until find ball
    time_limit = 50

    # Set initial act
    act1 = 0
    act2 = 0
    act3 = 0

    # Consider delay
    if delay != 0:
        act1 = 0
        act2 = 0
        act3 = 0
        delay -= 1
        action = [act1, act2, act3]
        return action, delay

    # follow the ball : attacker
    for i in range(11):
        if state[0][i][0] == 1:
            limit[agent] = 0    # reset limit if agent find ball
            act3 = lrway(i)
            if i < 3:
                ball_atk[agent].insert(t, state[0][i][7])
                act1 = 1
                delay += 1
            break
        if act3 != 2:
            act3 = 1

    # set max time until find ball : attacker
    if ball_atk[agent][t] == 2:
        limit[agent] += 1
    if limit[agent] > time_limit:
        act1 = random.randint(1, 3) % 2
        act2 = random.randint(1, 3) % 2
        act3 = random.randint(1, 3) % 2
        print("Can't find ball\n")

    # Give action
    action = [act1, act2, act3]
    return action, delay

def_time = [4, 4]

def defend(agent, delay, t, state):
    ball_def[agent].insert(t, 2)

    # set wall distance
    wall_dist = 0.2

    # Set initial act
    act1 = 0
    act2 = 0
    act3 = 0

    # Consider delay
    if delay != 0:
        act1 = 0
        act2 = 0
        act3 = 0
        delay -= 1
        action = [act1, act2, act3]
        return action, delay

    # block the ball : defender
    for i in range(11):
        if state[0][i][0] == 1:
            delay += 1
            ball_def[agent].insert(t, state[0][i][7])
            act2 = 3 - lrway(i)
            if i < 3:
                act2 = 0
            if i < 3:
                act2 = 0
            break

    # find goalpost : defender
    if state[0][9][3] == 1 and state[0][9][7] < wall_dist:
        act2 = 2
    if state[0][10][3] == 1 and state[0][10][7] < wall_dist:
        act2 = 1

    # initial setting : defender
    if def_time[agent] != 0:
        act1 = 2
        act2 = 0
        def_time[agent] -= 1

    # Give action
    action = [act1, act2, act3]
    return action, delay