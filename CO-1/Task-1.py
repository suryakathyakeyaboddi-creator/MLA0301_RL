import numpy as np

# Warehouse Grid
ROWS, COLS = 5, 5
START = (0, 0)
GOAL = (4, 4)
OBSTACLES = [(1, 2), (2, 2), (3, 1)]

# Parameters
gamma = 0.9
theta = 0.001

# Rewards
GOAL_REWARD = 100
STEP_REWARD = -1
OBSTACLE_REWARD = -100

actions = [(-1,0),(1,0),(0,-1),(0,1)]   # Up, Down, Left, Right

V = np.zeros((ROWS, COLS))

while True:
    delta = 0

    for i in range(ROWS):
        for j in range(COLS):

            if (i, j) == GOAL or (i, j) in OBSTACLES:
                continue

            values = []

            for dx, dy in actions:

                ni = i + dx
                nj = j + dy

                if 0 <= ni < ROWS and 0 <= nj < COLS:

                    if (ni, nj) in OBSTACLES:
                        reward = OBSTACLE_REWARD
                        value = reward + gamma * V[i][j]

                    elif (ni, nj) == GOAL:
                        value = GOAL_REWARD

                    else:
                        reward = STEP_REWARD
                        value = reward + gamma * V[ni][nj]

                    values.append(value)

            best = max(values)
            delta = max(delta, abs(best - V[i][j]))
            V[i][j] = best

    if delta < theta:
        break

print("\nOptimal State Values\n")
print(np.round(V,2))

print("\nWarehouse Layout\n")

for i in range(ROWS):
    for j in range(COLS):

        if (i,j)==START:
            print(" S ",end="")

        elif (i,j)==GOAL:
            print(" G ",end="")

        elif (i,j) in OBSTACLES:
            print(" X ",end="")

        else:
            print(" . ",end="")
    print()
