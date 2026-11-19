import numpy as np
import random

ROWS = 5
COLS = 5

start = (0,0)
goal = (4,4)

obstacles = [(1,2),(2,2),(3,1)]

actions = [(-1,0),(1,0),(0,-1),(0,1)]

alpha = 0.2
gamma = 0.9
epsilon = 0.1

Q = {}

for i in range(ROWS):
    for j in range(COLS):
        Q[(i,j)] = np.zeros(4)

# ---------------- Q Learning ----------------

for episode in range(500):

    state = start

    while state != goal:

        if random.random() < epsilon:
            action = random.randint(0,3)
        else:
            action = np.argmax(Q[state])

        dx,dy = actions[action]

        nr = state[0]+dx
        nc = state[1]+dy

        if nr<0 or nr>=ROWS or nc<0 or nc>=COLS:
            next_state = state
            reward = -5

        elif (nr,nc) in obstacles:
            next_state = state
            reward = -10

        elif (nr,nc)==goal:
            next_state=(nr,nc)
            reward=100

        else:
            next_state=(nr,nc)
            reward=-1

        Q[state][action]+=alpha*(
            reward+gamma*np.max(Q[next_state])-Q[state][action]
        )

        state=next_state


# ---------------- Best Path ----------------

state=start

path=[state]

while state!=goal:

    action=np.argmax(Q[state])

    dx,dy=actions[action]

    nr=state[0]+dx
    nc=state[1]+dy

    if (nr,nc)==state:
        break

    state=(nr,nc)

    if state in path:
        break

    path.append(state)

print("\nOptimal Path")

for p in path:
    print(p)

print("\nWarehouse Grid\n")

for i in range(ROWS):

    for j in range(COLS):

        if (i,j)==start:
            print(" S ",end="")

        elif (i,j)==goal:
            print(" G ",end="")

        elif (i,j) in obstacles:
            print(" X ",end="")

        elif (i,j) in path:
            print(" * ",end="")

        else:
            print(" . ",end="")

    print()


print("\nPerformance")

print("Episodes =",500)
print("Learning Rate =",alpha)
print("Discount Factor =",gamma)
print("Exploration =",epsilon)

print("\nAlgorithm Comparison")

print("Dynamic Programming : Model Based")
print("SARSA : On Policy")
print("Q Learning : Off Policy")
print("DQN : Large State Spaces")
