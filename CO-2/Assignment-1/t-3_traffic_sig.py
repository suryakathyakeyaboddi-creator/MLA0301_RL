
"""
MLA03 - REINFORCEMENT LEARNING
ASSIGNMENT 1 - TASKS 1, 2 AND 3

Project: Smart Traffic Signal Control using Q-Learning

TASK 1:
- RL framework
- Traffic environment
- State and action representation
- Reward function
- Q-learning
- Fixed-time comparison

TASK 2:
- MDP formulation
- State, action, transition and reward example
- Discount factor
- Expected reward calculation

TASK 3:
- Deterministic and stochastic policies
- State-value and action-value functions
- Bellman expectation equation
- Bellman optimality equation
- One-step Bellman calculation
- Policy evaluation complexity

Install:
    pip install numpy matplotlib

Run:
    python traffic_signal_rl.py
"""

import os
import csv
import random
import numpy as np
import matplotlib.pyplot as plt


# ============================================================
# 1. CONFIGURATION
# ============================================================

SEED = 42

EPISODES = 350
TEST_EPISODES = 50

SIMULATION_TIME = 600
GREEN_STEP = 5

MIN_GREEN = 10
MAX_GREEN = 30

YELLOW_TIME = 3
ALL_RED_TIME = 1

LEARNING_RATE = 0.1
DISCOUNT_FACTOR = 0.95

EPSILON = 1.0
EPSILON_MIN = 0.05
EPSILON_DECAY = 0.99

OUTPUT_DIR = "traffic_rl_outputs"

random.seed(SEED)
np.random.seed(SEED)

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================
# 2. TRAFFIC ENVIRONMENT
# ============================================================

class TrafficEnvironment:

    def __init__(self):

        self.approaches = [
            "North",
            "South",
            "East",
            "West"
        ]

        # Vehicle arrival rates per second
        self.arrival_rates = [
            0.32,
            0.30,
            0.38,
            0.34
        ]

        # Vehicle service rates per second
        self.service_rates = [
            0.45,
            0.45,
            0.45,
            0.45
        ]

        self.reset()

    # --------------------------------------------------------
    # Reset environment
    # --------------------------------------------------------

    def reset(self):

        self.queues = np.zeros(4, dtype=int)

        # 0 = North-South green
        # 1 = East-West green
        self.phase = 0

        self.elapsed_green = 0
        self.time = 0

        self.total_arrivals = 0
        self.total_served = 0

        self.total_queue_time = 0

        return self.get_state()

    # --------------------------------------------------------
    # Queue classification
    # --------------------------------------------------------

    def queue_category(self, queue):

        if queue <= 4:
            return 0

        elif queue <= 9:
            return 1

        return 2

    # --------------------------------------------------------
    # State representation
    # --------------------------------------------------------

    def get_state(self):

        queue_states = [
            self.queue_category(q)
            for q in self.queues
        ]

        if self.elapsed_green <= 5:
            time_state = 0

        elif self.elapsed_green <= 10:
            time_state = 1

        elif self.elapsed_green <= 15:
            time_state = 2

        elif self.elapsed_green <= 20:
            time_state = 3

        else:
            time_state = 4

        # 3^4 queue combinations * 2 phases * 5 time bins
        state = 0

        for value in queue_states:
            state = state * 3 + value

        state = state * 2 + self.phase
        state = state * 5 + time_state

        return state

    # --------------------------------------------------------
    # Valid actions
    # --------------------------------------------------------

    def valid_actions(self):

        # At maximum green, switching is compulsory
        if self.elapsed_green >= MAX_GREEN:
            return [3]

        # Before minimum green, phase changes are prohibited
        if self.elapsed_green < MIN_GREEN:
            return [0, 1, 2] if self.phase == 0 else [0, 1, 2]

        return [0, 1, 2, 3]

    # --------------------------------------------------------
    # Execute an action
    #
    # 0 = Select North-South green
    # 1 = Select East-West green
    # 2 = Extend current green
    # 3 = Switch to opposite phase
    # --------------------------------------------------------

    def execute_action(self, action):

        old_phase = self.phase

        if action == 0:
            target_phase = 0

        elif action == 1:
            target_phase = 1

        elif action == 2:
            target_phase = self.phase

        else:
            target_phase = 1 - self.phase

        # Enforce minimum green time
        if (
            target_phase != self.phase
            and self.elapsed_green < MIN_GREEN
        ):
            target_phase = self.phase

        # Enforce maximum green time
        if self.elapsed_green >= MAX_GREEN:
            target_phase = 1 - self.phase

        # Change phase with clearance intervals
        if target_phase != old_phase:

            for _ in range(YELLOW_TIME):

                if self.time >= SIMULATION_TIME:
                    break

                self.advance_second(None)

            for _ in range(ALL_RED_TIME):

                if self.time >= SIMULATION_TIME:
                    break

                self.advance_second(None)

            self.phase = target_phase
            self.elapsed_green = 0

        # Apply green phase
        duration = min(
            GREEN_STEP,
            SIMULATION_TIME - self.time
        )

        served_this_step = 0
        queue_time_this_step = 0
        queue_sum = 0

        for _ in range(duration):

            if self.time >= SIMULATION_TIME:
                break

            served, queue_time, current_queue = (
                self.advance_second(self.phase)
            )

            served_this_step += served
            queue_time_this_step += queue_time
            queue_sum += current_queue

        self.elapsed_green += duration

        if duration > 0:

            average_queue = queue_sum / duration

            average_queue_time = (
                queue_time_this_step / duration
            )

        else:

            average_queue = 0
            average_queue_time = 0

        # Reward function
        #
        # R = -0.5Q - 0.1W + 2V
        #
        # Q = average queue length during the step
        # W = average queue occupancy per second
        # V = vehicles served during the step

        reward = (
            -0.5 * average_queue
            -0.1 * average_queue_time
            +2.0 * served_this_step
        )

        next_state = self.get_state()

        done = self.time >= SIMULATION_TIME

        return next_state, reward, done, served_this_step

    # --------------------------------------------------------
    # Simulate one second of traffic
    # --------------------------------------------------------

    def advance_second(self, active_phase):

        arrivals = [
            np.random.poisson(rate)
            for rate in self.arrival_rates
        ]

        self.queues += arrivals

        self.total_arrivals += sum(arrivals)

        # Queue occupancy before service
        queue_time = int(np.sum(self.queues))

        self.total_queue_time += queue_time

        served = 0

        if active_phase == 0:

            # North-South green
            active_approaches = [0, 1]

        elif active_phase == 1:

            # East-West green
            active_approaches = [2, 3]

        else:

            # Clearance period
            active_approaches = []

        for index in active_approaches:

            possible_departures = np.random.poisson(
                self.service_rates[index]
            )

            actual_departures = min(
                self.queues[index],
                possible_departures
            )

            self.queues[index] -= actual_departures

            served += actual_departures

        self.total_served += served

        self.time += 1

        return served, queue_time, int(np.sum(self.queues))


# ============================================================
# 3. Q-LEARNING AGENT
# ============================================================

class QLearningAgent:

    def __init__(self):

        # 3^4 * 2 * 5 = 810 states
        self.state_size = 810

        # Four available actions
        self.action_size = 4

        self.q_table = np.zeros(
            (self.state_size, self.action_size)
        )

        self.epsilon = EPSILON

    # --------------------------------------------------------
    # Epsilon-greedy policy
    # --------------------------------------------------------

    def select_action(self, state, valid_actions):

        if random.random() < self.epsilon:

            return random.choice(valid_actions)

        q_values = self.q_table[
            state,
            valid_actions
        ]

        maximum = np.max(q_values)

        best_actions = [
            action
            for action in valid_actions
            if self.q_table[state, action] == maximum
        ]

        return random.choice(best_actions)

    # --------------------------------------------------------
    # Q-learning update
    # --------------------------------------------------------

    def learn(
        self,
        state,
        action,
        reward,
        next_state,
        next_valid_actions,
        done
    ):

        current_q = self.q_table[state, action]

        if done or not next_valid_actions:

            target = reward

        else:

            maximum_next_q = np.max(
                self.q_table[
                    next_state,
                    next_valid_actions
                ]
            )

            target = (
                reward
                + DISCOUNT_FACTOR * maximum_next_q
            )

        self.q_table[state, action] = (
            current_q
            + LEARNING_RATE * (target - current_q)
        )

    # --------------------------------------------------------
    # Exploration reduction
    # --------------------------------------------------------

    def reduce_epsilon(self):

        self.epsilon = max(
            EPSILON_MIN,
            self.epsilon * EPSILON_DECAY
        )


# ============================================================
# 4. FIXED-TIME CONTROLLER
# ============================================================

class FixedTimeController:

    def select_action(self, environment):

        if environment.elapsed_green >= 20:
            return 3

        return 2


# ============================================================
# 5. EPISODE EXECUTION
# ============================================================

def run_episode(
    environment,
    agent=None,
    fixed_controller=None,
    training=False
):

    environment.reset()

    total_reward = 0

    while environment.time < SIMULATION_TIME:

        state = environment.get_state()

        valid_actions = environment.valid_actions()

        if agent is not None:

            action = agent.select_action(
                state,
                valid_actions
            )

        else:

            action = fixed_controller.select_action(
                environment
            )

            if action not in valid_actions:
                action = valid_actions[0]

        next_state, reward, done, served = (
            environment.execute_action(action)
        )

        total_reward += reward

        if training and agent is not None:

            next_valid_actions = (
                environment.valid_actions()
                if not done else []
            )

            agent.learn(
                state,
                action,
                reward,
                next_state,
                next_valid_actions,
                done
            )

        if done:
            break

    average_wait_proxy = (
        environment.total_queue_time
        / max(environment.total_arrivals, 1)
    )

    average_queue = (
        environment.total_queue_time
        / max(environment.time, 1)
    )

    return {
        "reward": total_reward,
        "average_wait": average_wait_proxy,
        "average_queue": average_queue,
        "arrivals": environment.total_arrivals,
        "served": environment.total_served
    }


# ============================================================
# 6. TRAINING
# ============================================================

def train_agent():

    environment = TrafficEnvironment()

    agent = QLearningAgent()

    rewards = []

    print("\n========== TRAINING STARTED ==========")

    for episode in range(EPISODES):

        result = run_episode(
            environment,
            agent=agent,
            training=True
        )

        rewards.append(result["reward"])

        agent.reduce_epsilon()

        if (episode + 1) % 25 == 0:

            recent_average = np.mean(
                rewards[-25:]
            )

            print(
                f"Episode {episode + 1}/{EPISODES} | "
                f"Average Reward: {recent_average:.2f} | "
                f"Epsilon: {agent.epsilon:.3f}"
            )

    print("\n========== TRAINING COMPLETED ==========")

    np.save(
        os.path.join(OUTPUT_DIR, "q_table.npy"),
        agent.q_table
    )

    return agent, rewards


# ============================================================
# 7. CONTROLLER EVALUATION
# ============================================================

def evaluate_controller(
    agent=None,
    controller_name="Controller"
):

    environment = TrafficEnvironment()

    fixed_controller = FixedTimeController()

    results = []

    original_epsilon = None

    if agent is not None:

        original_epsilon = agent.epsilon
        agent.epsilon = 0

    for episode in range(TEST_EPISODES):

        result = run_episode(
            environment,
            agent=agent,
            fixed_controller=fixed_controller,
            training=False
        )

        result["episode"] = episode + 1

        results.append(result)

    if agent is not None:
        agent.epsilon = original_epsilon

    average_reward = np.mean([
        item["reward"] for item in results
    ])

    average_wait = np.mean([
        item["average_wait"] for item in results
    ])

    average_queue = np.mean([
        item["average_queue"] for item in results
    ])

    average_served = np.mean([
        item["served"] for item in results
    ])

    print(
        f"\n========== {controller_name} =========="
    )

    print(f"Average Reward: {average_reward:.2f}")

    print(
        f"Average Waiting-Time Proxy: "
        f"{average_wait:.2f}"
    )

    print(
        f"Average Queue Length: "
        f"{average_queue:.2f}"
    )

    print(
        f"Average Vehicles Served: "
        f"{average_served:.2f}"
    )

    return results


# ============================================================
# 8. TASK 2 - MDP FORMULATION
# ============================================================

def display_mdp_example():

    print("\n======================================")
    print("TASK 2 - MARKOV DECISION PROCESS")
    print("======================================")

    print("\nMDP = (S, A, P, R, gamma)")

    print("\nStates:")

    print("S1 = Low traffic")
    print("S2 = High North-South traffic")
    print("S3 = Medium traffic")

    print("\nActions:")

    print("A0 = North-South green")
    print("A1 = East-West green")
    print("A2 = Extend current green")
    print("A3 = Switch phase")

    print("\nIllustrative transition table:")

    print("--------------------------------------")
    print("Next State   Probability   Reward")
    print("--------------------------------------")

    print("S1           0.20          +5")
    print("S2           0.20          -2")
    print("S3           0.60          +2")

    print("--------------------------------------")

    probabilities = [0.2, 0.2, 0.6]

    rewards = [5, -2, 2]

    probability_sum = sum(probabilities)

    expected_reward = sum(
        p * r
        for p, r in zip(probabilities, rewards)
    )

    print("Probability sum:", probability_sum)

    print("Expected immediate reward:", expected_reward)

    print(
        "\nThese probabilities are illustrative "
        "assignment examples, not measured "
        "simulation transition probabilities."
    )

    print("\nDiscount factor:", DISCOUNT_FACTOR)


# ============================================================
# 9. TASK 3 - POLICY AND VALUE FUNCTIONS
# ============================================================

def display_policy_examples():

    print("\n======================================")
    print("TASK 3 - POLICIES AND VALUE FUNCTIONS")
    print("======================================")

    print("\nDeterministic policy example:")

    print(
        "For a given traffic state, always select "
        "North-South green."
    )

    print("\nStochastic policy example:")

    print("North-South green : 0.70")
    print("East-West green   : 0.20")
    print("Extend green      : 0.10")

    print("\nState-value function:")

    print("V_pi(s) = Expected discounted return from state s")

    print("\nAction-value function:")

    print(
        "Q_pi(s,a) = Expected discounted return "
        "after taking action a in state s"
    )

    print("\nBellman expectation equation:")

    print(
        "V_pi(s) = Sum_a pi(a|s) "
        "Sum_(s',r) P(s',r|s,a) "
        "[r + gamma V_pi(s')]"
    )

    print("\nBellman optimality equation:")

    print(
        "V*(s) = max_a "
        "Sum_(s',r) P(s',r|s,a) "
        "[r + gamma V*(s')]"
    )


# ============================================================
# 10. TASK 3 - NUMERICAL ACTION COMPARISON
# ============================================================

def compare_actions():

    print("\n======================================")
    print("TASK 3 - ACTION RETURN COMPARISON")
    print("======================================")

    gamma = 0.8

    reward_A = 4
    future_value_A = 8

    reward_B = 6
    future_value_B = 5

    return_A = (
        reward_A
        + gamma * future_value_A
    )

    return_B = (
        reward_B
        + gamma * future_value_B
    )

    print("\nAction A:")

    print(
        f"{reward_A} + ({gamma} * {future_value_A}) "
        f"= {return_A:.2f}"
    )

    print("\nAction B:")

    print(
        f"{reward_B} + ({gamma} * {future_value_B}) "
        f"= {return_B:.2f}"
    )

    if return_A > return_B:

        print("\nPreferred action: A")

    elif return_B > return_A:

        print("\nPreferred action: B")

    else:

        print("\nBoth actions have equal returns.")


# ============================================================
# 11. TASK 3 - BELLMAN ONE-STEP CALCULATION
# ============================================================

def bellman_return(
    probabilities,
    rewards,
    next_state_values,
    gamma
):

    total = 0.0

    for i in range(len(probabilities)):

        # TODO 1: Probability-weighted reward
        weighted_reward = (
            probabilities[i] * rewards[i]
        )

        # TODO 2: Discounted next-state value
        discounted_value = (
            probabilities[i]
            * gamma
            * next_state_values[i]
        )

        # TODO 3: Add contribution to total
        total += (
            weighted_reward
            + discounted_value
        )

    return total


def display_bellman_calculation():

    print("\n======================================")
    print("TASK 3 - BELLMAN ONE-STEP CALCULATION")
    print("======================================")

    gamma = 0.8

    probabilities = [0.6, 0.4]

    rewards = [5, -2]

    next_state_values = [10, 4]

    result = bellman_return(
        probabilities,
        rewards,
        next_state_values,
        gamma
    )

    print("Bellman return:", round(result, 2))

    print("\nExpected calculation:")

    print("0.6 * (5 + 0.8 * 10) = 7.80")

    print("0.4 * (-2 + 0.8 * 4) = 0.48")

    print("Total = 7.80 + 0.48 = 8.28")


# ============================================================
# 12. TASK 3 - POLICY EVALUATION COMPLEXITY
# ============================================================

def display_policy_evaluation_complexity():

    print("\n======================================")
    print("POLICY EVALUATION COMPLEXITY")
    print("======================================")

    print("\nIterative Bellman evaluation:")

    print(
        "Cost per sweep: O(|S| * |A| * K)"
    )

    print(
        "K = maximum possible next states "
        "for each state-action pair."
    )

    print("\nDirect evaluation:")

    print(
        "Dense matrix solution: O(|S|^3)"
    )

    print(
        "Dense matrix memory: O(|S|^2)"
    )

    print(
        "\nRepeated updates are needed because "
        "state values depend on future state values."
    )

    print(
        "For a discounted finite MDP, iterative "
        "policy evaluation converges to the "
        "value function under a fixed policy."
    )


# ============================================================
# 13. SAVE RESULTS
# ============================================================

def save_results(filename, results):

    path = os.path.join(
        OUTPUT_DIR,
        filename
    )

    fieldnames = [
        "episode",
        "reward",
        "average_wait",
        "average_queue",
        "arrivals",
        "served"
    ]

    with open(path, "w", newline="") as file:

        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames
        )

        writer.writeheader()

        writer.writerows(results)

    print("\nSaved:", path)


# ============================================================
# 14. TRAINING GRAPH
# ============================================================

def plot_training_rewards(rewards):

    plt.figure(figsize=(10, 5))

    plt.plot(
        rewards,
        label="Episode Reward"
    )

    if len(rewards) >= 25:

        moving_average = np.convolve(
            rewards,
            np.ones(25) / 25,
            mode="valid"
        )

        plt.plot(
            range(24, len(rewards)),
            moving_average,
            label="25-Episode Moving Average"
        )

    plt.xlabel("Episode")

    plt.ylabel("Total Reward")

    plt.title("Q-Learning Training Performance")

    plt.legend()

    plt.grid(True)

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            OUTPUT_DIR,
            "training_rewards.png"
        )
    )

    plt.show()


# ============================================================
# 15. RL VS FIXED-TIME COMPARISON GRAPH
# ============================================================

def plot_comparison(rl_results, fixed_results):

    labels = [
        "Average Reward",
        "Waiting-Time Proxy",
        "Average Queue",
        "Vehicles Served"
    ]

    rl_values = [
        np.mean([x["reward"] for x in rl_results]),

        np.mean([x["average_wait"] for x in rl_results]),

        np.mean([x["average_queue"] for x in rl_results]),

        np.mean([x["served"] for x in rl_results])
    ]

    fixed_values = [
        np.mean([x["reward"] for x in fixed_results]),

        np.mean([x["average_wait"] for x in fixed_results]),

        np.mean([x["average_queue"] for x in fixed_results]),

        np.mean([x["served"] for x in fixed_results])
    ]

    x = np.arange(len(labels))

    width = 0.35

    plt.figure(figsize=(11, 5))

    plt.bar(
        x - width / 2,
        rl_values,
        width,
        label="Q-Learning"
    )

    plt.bar(
        x + width / 2,
        fixed_values,
        width,
        label="Fixed-Time"
    )

    plt.xticks(
        x,
        labels,
        rotation=10
    )

    plt.ylabel("Measured Value")

    plt.title(
        "Q-Learning vs Fixed-Time Controller"
    )

    plt.legend()

    plt.grid(axis="y")

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            OUTPUT_DIR,
            "controller_comparison.png"
        )
    )

    plt.show()


# ============================================================
# 16. MAIN PROGRAM
# ============================================================

if __name__ == "__main__":

    print("\n")
    print("==============================================")
    print(" SMART TRAFFIC SIGNAL CONTROL USING RL")
    print(" MLA03 - REINFORCEMENT LEARNING")
    print(" ASSIGNMENT 1 - TASKS 1, 2 AND 3")
    print("==============================================")

    print("\nEnvironment: Four-way traffic intersection")

    print("State space: 810")

    print("Action space: 4")

    print("Algorithm: Q-Learning")

    print("Learning rate:", LEARNING_RATE)

    print("Discount factor:", DISCOUNT_FACTOR)

    # TASK 2
    display_mdp_example()

    # TASK 3
    display_policy_examples()

    compare_actions()

    display_bellman_calculation()

    display_policy_evaluation_complexity()

    # TASK 1 - Train Q-learning agent
    trained_agent, training_rewards = train_agent()

    # Evaluate RL controller
    rl_results = evaluate_controller(
        agent=trained_agent,
        controller_name="Q-LEARNING CONTROLLER"
    )

    # Evaluate fixed-time controller
    fixed_results = evaluate_controller(
        agent=None,
        controller_name="FIXED-TIME CONTROLLER"
    )

    # Save evaluation data
    save_results(
        "rl_evaluation.csv",
        rl_results
    )

    save_results(
        "fixed_time_evaluation.csv",
        fixed_results
    )

    # Generate graphs
    plot_training_rewards(
        training_rewards
    )

    plot_comparison(
        rl_results,
        fixed_results
    )

    print("\n==============================================")
    print(" PROGRAM EXECUTION COMPLETED")
    print("==============================================")

    print("\nGenerated files are available in:")

    print(OUTPUT_DIR)

    print("\nFiles:")

    print("1. q_table.npy")

    print("2. rl_evaluation.csv")

    print("3. fixed_time_evaluation.csv")

    print("4. training_rewards.png")

    print("5. controller_comparison.png")
