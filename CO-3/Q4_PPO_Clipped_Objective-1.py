# Q4 – PPO Probability Ratio and Clipped Objective

old_probability = 0.4
new_probability = 0.5
advantage = 2.0
epsilon = 0.2

ratio = new_probability / old_probability
objective1 = ratio * advantage
clipped_ratio = min(max(ratio, 1 - epsilon), 1 + epsilon)
objective2 = clipped_ratio * advantage
ppo_objective = min(objective1, objective2)

print("Probability Ratio:", ratio)
print("Clipped Ratio:", clipped_ratio)
print("PPO Objective:", ppo_objective)
