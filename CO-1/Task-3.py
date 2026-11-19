import numpy as np
import random

# States
states = [0, 1, 2]
actions = [0, 1]      # 0 = Move Forward, 1 = Stay

# Parameters
gamma = 0.9
alpha = 0.2
epsilon = 0.1

# Transition Model
P = {
    (0,0):(1,-1),
    (0,1):(0,-2),
    (1,0):(2,10),
    (1,1):(1,-2)
}

# ---------------- First-Visit Monte Carlo ----------------

returns = {0:[], 1:[]}
V = np.zeros(len(states))

for episode in range(10):

    state = 0
    episode_states = []
    rewards = []

    while state != 2:

        action = random.choice(actions)
        next_state, reward = P[(state, action)]

        episode_states.append(state)
        rewards.append(reward)

        state = next_state

    G = 0
    visited = set()

    for i in range(len(episode_states)-1, -1, -1):

        G = gamma * G + rewards[i]

        if episode_states[i] not in visited:
            returns[episode_states[i]].append(G)
            V[episode_states[i]] = np.mean(returns[episode_states[i]])
            visited.add(episode_states[i])

print("\nFirst Visit Monte Carlo Values")
print(np.round(V,2))


# ---------------- Every-Visit Monte Carlo ----------------

returns = {0:[],1:[]}
V = np.zeros(len(states))

for episode in range(10):

    state = 0
    episode_states = []
    rewards = []

    while state != 2:

        action = random.choice(actions)
        next_state, reward = P[(state, action)]

        episode_states.append(state)
        rewards.append(reward)

        state = next_state

    G = 0

    for i in range(len(episode_states)-1,-1,-1):

        G = gamma*G + rewards[i]
        returns[episode_states[i]].append(G)
        V[episode_states[i]] = np.mean(returns[episode_states[i]])

print("\nEvery Visit Monte Carlo Values")
print(np.round(V,2))


# ---------------- Monte Carlo Control ----------------

Q = np.zeros((len(states), len(actions)))

for episode in range(20):

    state = 0

    while state != 2:

        if random.random() < epsilon:
            action = random.choice(actions)
        else:
            action = np.argmax(Q[state])

        next_state, reward = P[(state, action)]

        Q[state][action] += alpha * (reward - Q[state][action])

        state = next_state

print("\nQ Table")
print(np.round(Q,2))


# ---------------- TD(0) ----------------

V = np.zeros(len(states))

for episode in range(20):

    state = 0

    while state != 2:

        action = random.choice(actions)

        next_state, reward = P[(state, action)]

        V[state] = V[state] + alpha * (
            reward + gamma * V[next_state] - V[state]
        )

        state = next_state

print("\nTD(0) State Values")
print(np.round(V,2))


# ---------------- Numerical Example ----------------

current = 5
reward = 3
next_value = 8

updated = current + alpha * (reward + gamma * next_value - current)

print("\nUpdated State Value =", round(updated,2))
