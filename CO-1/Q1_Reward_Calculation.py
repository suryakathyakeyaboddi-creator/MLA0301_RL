# Q1 – Reward Calculation

def calculate_reward(action, next_position, treasure, enemy, exit_point):
    reward = -1
    if next_position == treasure:
        reward += 10
    if enemy is not None and next_position == enemy:
        reward -= 20
    if next_position == exit_point:
        reward += 50
    return reward

reward = calculate_reward("RIGHT", (2, 3), (2, 3), None, (4, 4))
print("Game Reward:", reward)
