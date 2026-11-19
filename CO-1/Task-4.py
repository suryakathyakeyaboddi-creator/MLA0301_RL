import numpy as np
import random

states = [0, 1, 2]
actions = [0, 1]

alpha = 0.2
gamma = 0.9
epsilon = 0.1

P = {
    (0,0):(1,-1),
    (0,1):(0,-2),
    (1,0):(2,10),
    (1,1):(1,-2)
}

# ---------------- SARSA ----------------

Q = np.zeros((3,2))

for episode in range(100):

    state = 0

    if random.random() < epsilon:
        action = random.choice(actions)
    else:
        action = np.argmax(Q[state])

    while state != 2:

        next_state, reward = P[(state,action)]

        if next_state == 2:
            next_action = 0
        else:
            if random.random() < epsilon:
                next_action = random.choice(actions)
            else:
                next_action = np.argmax(Q[next_state])

        Q[state][action] += alpha * (
            reward + gamma * Q[next_state][next_action] - Q[state][action]
        )

        state = next_state
        action = next_action

print("\nSARSA Q Table")
print(np.round(Q,2))


# ---------------- Q Learning ----------------

Q = np.zeros((3,2))

for episode in range(100):

    state = 0

    while state != 2:

        if random.random() < epsilon:
            action = random.choice(actions)
        else:
            action = np.argmax(Q[state])

        next_state, reward = P[(state,action)]

        Q[state][action] += alpha * (
            reward + gamma*np.max(Q[next_state]) - Q[state][action]
        )

        state = next_state

print("\nQ Learning Table")
print(np.round(Q,2))


# ---------------- Best Policy ----------------

print("\nOptimal Policy")

for s in [0,1]:

    if np.argmax(Q[s]) == 0:
        print("State",s,"-> Move Forward")
    else:
        print("State",s,"-> Stay")


# ---------------- Numerical Example ----------------

current = 5
reward = 4
next_sarsa = 6
next_q = 8

sarsa = current + alpha*(reward + gamma*next_sarsa-current)
qlearning = current + alpha*(reward + gamma*next_q-current)

print("\nSARSA Updated Q =",round(sarsa,2))
print("Q Learning Updated Q =",round(qlearning,2))


print("\nDQN")
print("Uses Neural Network instead of Q Table")
print("Experience Replay")
print("Target Network")
print("DDQN")
print("Dueling DQN")
print("Prioritized Experience Replay")
