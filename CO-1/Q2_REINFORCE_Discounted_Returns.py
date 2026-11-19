# Q2 – REINFORCE Discounted Returns

gamma = 0.9

def calculate_returns(rewards, gamma):
    returns = []
    G = 0
    for reward in reversed(rewards):
        G = reward + gamma * G
        returns.insert(0, G)
    return returns

rewards = [1, 1, 1, 10]
returns = calculate_returns(rewards, gamma)
print("Discounted Returns:", returns)
