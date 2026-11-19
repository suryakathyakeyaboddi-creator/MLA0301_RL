"""Task 4: SARSA and Q-learning on a warehouse grid; DQN concepts in comments/output."""
import numpy as np

ROWS=5; COLS=5; START=(0,0); GOAL=(4,4); BLOCKED={(1,1),(1,3),(2,3),(3,1)}
MOVES=[(-1,0),(1,0),(0,-1),(0,1)]; N_ACTIONS=4

def step(s,a):
    r,c=divmod(s,COLS); dr,dc=MOVES[a]; nr,nc=r+dr,c+dc
    if not(0<=nr<ROWS and 0<=nc<COLS) or (nr,nc) in BLOCKED: return s,-5,False
    ns=nr*COLS+nc; done=(nr,nc)==GOAL
    return ns,(20 if done else -1),done

def train(method='sarsa',episodes=3500,alpha=.15,gamma=.95,epsilon=.2,seed=10):
    rng=np.random.default_rng(seed); Q=np.zeros((ROWS*COLS,N_ACTIONS)); rewards=[]; successes=0
    for ep in range(episodes):
        s=0; total=0; a=int(rng.integers(N_ACTIONS)) if rng.random()<epsilon else int(np.argmax(Q[s]))
        for _ in range(100):
            if rng.random()<epsilon: a=int(rng.integers(N_ACTIONS))
            ns,r,done=step(s,a); total+=r
            if method=='sarsa':
                na=int(rng.integers(N_ACTIONS)) if rng.random()<epsilon else int(np.argmax(Q[ns]))
                target=r+(0 if done else gamma*Q[ns,na])
            else:
                target=r+(0 if done else gamma*np.max(Q[ns]))
                na=None
            Q[s,a]+=alpha*(target-Q[s,a]); s=ns
            if done: successes+=1; break
            a=na if method=='sarsa' else (int(rng.integers(N_ACTIONS)) if rng.random()<epsilon else int(np.argmax(Q[s])))
        rewards.append(total)
    return Q,np.array(rewards),successes/episodes

def greedy_path(Q,max_steps=50):
    s=0; path=[s]; seen={s}
    for _ in range(max_steps):
        a=int(np.argmax(Q[s])); ns,_,done=step(s,a)
        if ns==s or ns in seen: break
        path.append(ns); seen.add(ns); s=ns
        if done: break
    return [(x//COLS,x%COLS) for x in path]

if __name__=='__main__':
    for method in ('sarsa','qlearning'):
        Q,R,sr=train(method); print(f"{method.upper()}: final-100 avg reward={R[-100:].mean():.2f}, training success rate={sr:.1%}")
        print("Greedy path:",greedy_path(Q))
    # Required numerical updates
    q=5; reward=4; alpha=.2; gamma=.9; next_sarsa=6; next_max=8
    print("SARSA Qnew:",q+alpha*(reward+gamma*next_sarsa-q))
    print("Q-learning Qnew:",q+alpha*(reward+gamma*next_max-q))
    print("DQN: neural network approximates Q(s,a); replay buffer breaks correlation; target network stabilizes targets.")
    print("DDQN separates action selection and evaluation; Dueling DQN estimates V(s) and A(s,a); PER samples high-TD-error transitions more often.")
