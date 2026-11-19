"""Task 3: First/Every-Visit MC, epsilon-soft MC control, and TD(0)."""
import numpy as np

def episode(policy=None,rng=None,epsilon=0.2):
    # Tiny corridor: 0 -> 1 -> 2 (dispatch); action 0=forward, 1=stay.
    s=0; traj=[]
    while s!=2 and len(traj)<12:
        if policy is None or rng.random()<epsilon: a=int(rng.integers(2))
        else: a=int(policy[s])
        ns,reward=(s+1,3 if s==1 else -1) if a==0 else (s,-2)
        traj.append((s,a,reward)); s=ns
    return traj

def mc_prediction(episodes=3000,seed=3):
    rng=np.random.default_rng(seed); first={0:[],1:[]}; every={0:[],1:[]}
    for _ in range(episodes):
        ep=episode(rng=rng); G=0; returns=[]
        for s,a,r in reversed(ep): G=r+0.9*G; returns.insert(0,(s,G))
        seen=set()
        for s,G in returns:
            every[s].append(G)
            if s not in seen: first[s].append(G); seen.add(s)
    avg=lambda d:{s:round(float(np.mean(v)),3) if v else 0 for s,v in d.items()}
    return avg(first),avg(every)

def mc_control(episodes=5000,epsilon=0.15,seed=4):
    rng=np.random.default_rng(seed); Q=np.zeros((2,2)); counts=[[[] for _ in range(2)] for _ in range(2)]
    for _ in range(episodes):
        policy={s:int(np.argmax(Q[s])) for s in range(2)}; ep=episode(policy,rng,epsilon)
        G=0; seen=set()
        for s,a,r in reversed(ep):
            G=r+0.9*G
            if (s,a) not in seen: counts[s][a].append(G); Q[s,a]=np.mean(counts[s][a]); seen.add((s,a))
    return Q

def td0(episodes=3000,alpha=0.2,gamma=0.9,seed=5):
    rng=np.random.default_rng(seed); V=np.zeros(3)
    for _ in range(episodes):
        s=0
        while s!=2:
            a=0 if rng.random()<0.8 else 1
            ns,r=(s+1,3 if s==1 else -1) if a==0 else (s,-2)
            target=r+(0 if ns==2 else gamma*V[ns]); V[s]+=alpha*(target-V[s]); s=ns
    return V

if __name__=='__main__':
    first,every=mc_prediction(); print("First-Visit MC values:",first); print("Every-Visit MC values:",every)
    Q=mc_control(); print("Monte Carlo Control Q-values:\n",np.round(Q,3)); print("Greedy actions (0=Forward, 1=Stay):",np.argmax(Q,axis=1))
    V=td0(); print("TD(0) learned state values:",np.round(V,3))
    # Required numerical exercise: V=5, R=3, Vnext=8, alpha=.2, gamma=.9
    old=5; reward=3; next_value=8; alpha=.2; gamma=.9
    target=reward+gamma*next_value; updated=old+alpha*(target-old)
    print(f"TD(0) numerical: target={reward}+{gamma}*{next_value}={target:.2f}; Vnew={old}+{alpha}*({target:.2f}-{old})={updated:.2f}")
