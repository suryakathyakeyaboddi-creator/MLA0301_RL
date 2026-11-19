"""Task 1: Warehouse MDP formulation, labelled grid, transitions and value iteration."""
import numpy as np

ROWS = COLS = 5
START, GOAL = (0, 0), (4, 4)
OBSTACLES = {(1, 1), (1, 3), (2, 3), (3, 1)}
ACTIONS = {"Up": (-1, 0), "Down": (1, 0), "Left": (0, -1), "Right": (0, 1)}
GAMMA, THETA = 0.90, 1e-5

def valid(pos):
    r, c = pos
    return 0 <= r < ROWS and 0 <= c < COLS and pos not in OBSTACLES

def step(state, action):
    if state == GOAL:
        return state, 0.0
    dr, dc = ACTIONS[action]
    nxt = (state[0] + dr, state[1] + dc)
    if not valid(nxt):
        return state, -5.0  # collision / boundary penalty
    return nxt, (20.0 if nxt == GOAL else -1.0)

def show_grid():
    print("Warehouse grid: S=start, G=goal, #=obstacle")
    for r in range(ROWS):
        print(" ".join("S" if (r,c)==START else "G" if (r,c)==GOAL else "#" if (r,c) in OBSTACLES else "." for c in range(COLS)))

def value_iteration():
    states=[(r,c) for r in range(ROWS) for c in range(COLS) if (r,c) not in OBSTACLES]
    V={s:0.0 for s in states}
    while True:
        delta=0.0
        for s in states:
            if s == GOAL: continue
            old=V[s]
            V[s]=max(reward + GAMMA*V[nxt] for a in ACTIONS for nxt,reward in [step(s,a)])
            delta=max(delta,abs(old-V[s]))
        if delta < THETA: break
    policy={}
    for s in states:
        if s == GOAL: policy[s]="GOAL"; continue
        policy[s]=max(ACTIONS, key=lambda a: step(s,a)[1] + GAMMA*V[step(s,a)[0]])
    return V,policy

if __name__ == '__main__':
    show_grid(); values,policy=value_iteration()
    print("\nOptimal state values:")
    for r in range(ROWS): print(" ".join("#####" if (r,c) in OBSTACLES else f"{values[(r,c)]:5.1f}" for c in range(COLS)))
    print("\nGreedy policy (arrows):")
    arrows={"Up":"↑","Down":"↓","Left":"←","Right":"→","GOAL":"G"}
    for r in range(ROWS): print(" ".join(" # " if (r,c) in OBSTACLES else f" {arrows[policy[(r,c)]]} " for c in range(COLS)))
    print("\nMDP: S=valid cells; A={Up,Down,Left,Right}; P is deterministic for this baseline; R=+20 goal, -1 step, -5 invalid move; gamma=0.9.")
