# Q3 – Actor-Critic TD Target, TD Error and Advantage

gamma = 0.9
reward = 5
value_current = 8
value_next = 10

td_target = reward + gamma * value_next
td_error = td_target - value_current
advantage = td_error

print("TD Target:", td_target)
print("TD Error:", td_error)
print("Advantage:", advantage)
