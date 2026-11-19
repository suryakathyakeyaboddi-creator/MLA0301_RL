"""Task 2: Policy Iteration, Value Iteration, and Monte Carlo prediction comparison."""
import numpy as np
GAMMA, THETA = 0.9, 1e-3
STATES=[0,1,2,3,4]  # Start, aisle, shelf, near packing, packing terminal
ACTIONS=["Forward","Stay"]
P={(0,"Forward"):(1,-1),(0,"Stay"):(0,-2),
   (1,"Forward"):(2,-1),(1,"Stay"):(1,-2),
   (2,"Forward"):(3,-1),(2,"Stay"):(2,-2),
   (3,"Forward"):(4,10),(3,"Stay"):(3,-2)}
TERMINAL=4

def q_value(V,s,a):
    ns,reward=P[(s,a)]
    return reward + (0 if ns==TERMINAL else GAMMA*V[ns])

def policy_evaluation(policy):
    V=np.zeros(len(STATES))
    while True:
        delta=0
        for s in STATES[:-1]:
            old=V[s]; V[s]=q_value(V,s,policy[s]); delta=max(delta,abs(old-V[s]))
        if delta<THETA: return V

def policy_iteration():
    policy={s:"Stay" for s in STATES[:-1]}
    while True:
        V=policy_evaluation(policy); stable=True
        for s in STATES[:-1]:
            best=max(ACTIONS,key=lambda a:q_value(V,s,a))
            if best!=policy[s]: policy[s]=best; stable=False
        if stable: return V,policy

def value_iteration():
    V=np.zeros(len(STATES))
    while True:
        delta=0
        for s in STATES[:-1]:
            old=V[s]; V[s]=max(q_value(V,s,a) for a in ACTIONS); delta=max(delta,abs(old-V[s]))
        if delta<THETA: break
    policy={s:max(ACTIONS,key=lambda a:q_value(V,s,a)) for s in STATES[:-1]}
    return V,policy

def monte_carlo_prediction(episodes=5000,seed=7):
    rng=np.random.default_rng(seed); returns={s:[] for s in STATES[:-1]}
    # A simple exploratory episodic behaviour policy; all episodes reach terminal.
    for _ in range(episodes):
        s=0; trajectory=[]
        while s!=TERMINAL and len(trajectory)<20:
            a="Forward" if rng.random()<0.8 else "Stay"
            ns,reward=P[(s,a)]; trajectory.append((s,reward)); s=ns
        G=0
        for st,reward in reversed(trajectory):
            G=reward+GAMMA*G; returns[st].append(G)
    return {s:float(np.mean(v)) if v else 0.0 for s,v in returns.items()}

if __name__=='__main__':
    pv,pp=policy_iteration(); vv,vp=value_iteration(); mc=monte_carlo_prediction()
    print("Policy Iteration values:",np.round(pv,3)); print("Policy Iteration policy:",pp)
    print("Value Iteration values:",np.round(vv,3)); print("Value Iteration policy:",vp)
    print("Monte Carlo sample estimates (exploratory behaviour):",{k:round(v,3) for k,v in mc.items()})
    # Assignment's three-state TODO example, completed explicitly.
    P3={(0,0):(1,-1),(0,1):(0,-2),(1,0):(2,10),(1,1):(1,-2)}
    V=np.zeros(3); gamma=0.9; theta=0.001
    while True:
        delta=0
        for s in [0,1]:
            old=V[s]; action_values=[]
            for a in [0,1]:
                ns,reward=P3[(s,a)]; action_values.append(reward+gamma*V[ns])
            V[s]=max(action_values); delta=max(delta,abs(old-V[s]))
        if delta<theta: break
    print("Completed assignment TODO example optimal values:",np.round(V,3))
    print("Optimal actions (0=Forward, 1=Stay):",{s:int(np.argmax([P3[(s,a)][1]+gamma*V[P3[(s,a)][0]] for a in [0,1]])) for s in [0,1]})
