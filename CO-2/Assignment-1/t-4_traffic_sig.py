
# ============================================================
# TASK 4 - EPSILON-GREEDY MULTI-ARMED BANDIT
# ============================================================

import numpy as np

np.random.seed(7)

# Illustrative true average rewards of four arms
true_rewards = [1.0, 1.5, 2.0, 1.2]

n_arms = len(true_rewards)

epsilon = 0.1

steps = 1000

# Number of times each arm is selected
counts = np.zeros(n_arms)

# Estimated average reward of each arm
estimates = np.zeros(n_arms)

total_reward = 0.0

for step in range(steps):

    # TODO 1: Explore or exploit

    if np.random.rand() < epsilon:

        # Explore: choose a random arm
        action = np.random.randint(n_arms)

    else:

        # Exploit: choose the arm with highest estimate
        action = int(np.argmax(estimates))

    # Simulated reward with Gaussian noise
    reward = np.random.normal(
        true_rewards[action],
        1.0
    )

    # TODO 2: Update the selected arm's count
    counts[action] += 1

    # TODO 3: Update estimated average reward incrementally
    estimates[action] += (
        reward - estimates[action]
    ) / counts[action]

    # TODO 4: Add reward to total reward
    total_reward += reward


# Display results
print("\n========== TASK 4 RESULTS ==========")

print("Arm selections:", counts.astype(int))

print(
    "Estimated rewards:",
    np.round(estimates, 3)
)

print(
    "Average reward:",
    round(total_reward / steps, 3)
)
