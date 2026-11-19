import numpy as np
import random

# States
states = [0, 1, 2]
actions = [0, 1]     # 0 = Move Forward, 1 = Stay

gamma = 0.9
theta = 0.001

# Transition Model
P = {
    (0,0):(1,-1),
    (0,1):(0,-2),
    (1,0):(2,10),
    (1,1):(1,-2)
}

# ---------------- Value Iteration ----------------

V = np.zeros(len(states))

while True:

    delta = 0

    for s in [0,1]:

        old = V[s]
        values = []

        for a in actions:

            next_state, reward = P[(s,a)]
            value = reward + gamma * V[next_state]
            values.append(value)

        V[s] = max(values)
        delta = max(delta, abs(old - V[s]))

    if delta < theta:
        break

print("\nOptimal State Values")
print(np.round(V,2))


# ---------------- Optimal Policy ----------------

policy = []

for s in [0,1]:

    best_action = None
    best_value = -999

    for a in actions:

        next_state, reward = P[(s,a)]
        value = reward + gamma * V[next_state]

        if value > best_value:
            best_value = value
            best_action = a

    policy.append(best_action)

print("\nOptimal Policy")
for i,a in enumerate(policy):

    if a==0:
        print("State",i,"-> Move Forward")

    else:
        print("State",i,"-> Stay")


# ---------------- Simple Monte Carlo ----------------

returns = []

for episode in range(10):

    state = 0
    total_reward = 0

    while state != 2:

        action = random.choice(actions)

        next_state, reward = P[(state,action)]

        total_reward += reward
        state = next_state

    returns.append(total_reward)

print("\nMonte Carlo Episode Rewards")
print(returns)

print("\nAverage Reward =", round(np.mean(returns),2))
