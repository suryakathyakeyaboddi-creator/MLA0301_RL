
"""
SMART TRAFFIC SIGNAL CONTROL USING REINFORCEMENT LEARNING
Assignment 1 – Task 1 and Task 2

Algorithm: Tabular Q-Learning
Environment: Four-way traffic intersection
State space: 810 states
Actions: 4
Baseline: Fixed-time traffic signal controller

Install dependencies:
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

SIMULATION_TIME = 600       # Seconds per episode
GREEN_STEP = 5              # Seconds per RL decision
MIN_GREEN = 10
MAX_GREEN = 30

YELLOW_TIME = 3
ALL_RED_TIME = 1
CLEARANCE_TIME = YELLOW_TIME + ALL_RED_TIME

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

        # Four approaches: North, South, East, West
        self.approaches = ["North", "South", "East", "West"]

        # Arrival rates: vehicles per second
        self.arrival_rates = [0.32, 0.30, 0.38, 0.34]

        # Service rates: vehicles per second
        self.service_rates = [0.45, 0.45, 0.45, 0.45]

        self.reset()

    def reset(self):

        self.queues = np.zeros(4, dtype=int)

        # 0 = North-South green
        # 1 = East-West green
        self.phase = 0

        self.elapsed_green = 0
        self.time = 0

        self.total_arrivals = 0
        self.total_served = 0
        self.total_waiting_time = 0

        return self.get_state()

    # --------------------------------------------------------
    # Queue classification
    # --------------------------------------------------------

    def queue_category(self, queue):

        if queue <= 4:
            return 0       # Low

        elif queue <= 9:
            return 1       # Medium

        return 2           # High

    # --------------------------------------------------------
    # State representation
    # --------------------------------------------------------

    def get_state(self):

        queue_states = [
            self.queue_category(q) for q in self.queues
        ]

        # Five elapsed-green categories
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

        # State = (QN, QS, QE, QW, phase, elapsed time)
        state = tuple(queue_states) + (
            self.phase,
            time_state
        )

        # Encode the state as an integer
        state_index = 0

        for value in state:
            state_index = state_index * 3 + value

        # The phase and time features have different ranges
        state_index = (
            ((queue_states[0] * 3 + queue_states[1]) * 3
             + queue_states[2]) * 3 + queue_states[3]
        )

        state_index = state_index * 2 + self.phase
        state_index = state_index * 5 + time_state

        return state_index

    # --------------------------------------------------------
    # Available actions and safety restrictions
    # --------------------------------------------------------

    def valid_actions(self):

        if self.elapsed_green < MIN_GREEN:

            # Do not allow a phase change before minimum green
            return [self.phase, 2]

        if self.elapsed_green >= MAX_GREEN:

            # Force a phase change at maximum green
            return [3]

        return [0, 1, 2, 3]

    # --------------------------------------------------------
    # Action execution
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

        # Safety controller: enforce minimum green time
        if (
            target_phase != self.phase
            and self.elapsed_green < MIN_GREEN
        ):
            target_phase = self.phase

        # Safety controller: enforce maximum green time
        if self.elapsed_green >= MAX_GREEN:
            target_phase = 1 - self.phase

        # Phase change requires yellow and all-red clearance
        if target_phase != old_phase:

            clearance = min(
                CLEARANCE_TIME,
                SIMULATION_TIME - self.time
            )

            for _ in range(clearance):
                if self.time >= SIMULATION_TIME:
                    break

                self.advance_second(active_phase=None)

            self.phase = target_phase
            self.elapsed_green = 0

        # Apply the selected green phase
        green_duration = min(
            GREEN_STEP,
            SIMULATION_TIME - self.time
        )

        served_this_step = 0
        waiting_this_step = 0
        queue_sum = 0

        for _ in range(green_duration):

            if self.time >= SIMULATION_TIME:
                break

            served, waiting, queue = self.advance_second(
                active_phase=self.phase
            )

            served_this_step += served
            waiting_this_step += waiting
            queue_sum += queue

        self.elapsed_green += green_duration

        # Average queue and waiting accumulation per second
        if green_duration > 0:

            average_queue = queue_sum / green_duration

            average_waiting = (
                waiting_this_step / green_duration
            )

        else:
            average_queue = 0
            average_waiting = 0

        # Reward: lower queues and waiting, higher throughput
        reward = (
            -0.5 * average_queue
            -0.1 * average_waiting
            +2.0 * served_this_step
        )

        next_state = self.get_state()

        done = self.time >= SIMULATION_TIME

        return next_state, reward, done, served_this_step

    # --------------------------------------------------------
    # One-second traffic simulation
    # --------------------------------------------------------

    def advance_second(self, active_phase):

        # Vehicles arriving at all four approaches
        arrivals = [
            np.random.poisson(rate)
            for rate in self.arrival_rates
        ]

        self.queues += arrivals

        total_arrivals = sum(arrivals)
        self.total_arrivals += total_arrivals

        # Waiting time accumulated by vehicles in queues
        waiting = int(np.sum(self.queues))

        self.total_waiting_time += waiting

        served = 0

        # North-South phase
        if active_phase == 0:
            active_approaches = [0, 1]

        # East-West phase
        elif active_phase == 1:
            active_approaches = [2, 3]

        # Clearance period: no approach receives green
        else:
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

        return served, waiting, int(np.sum(self.queues))


# ============================================================
# 3. Q-LEARNING AGENT
# ============================================================

class QLearningAgent:

    def __init__(self):

        # 3^4 queue combinations * 2 phases * 5 time bins
        self.state_size = (3 ** 4) * 2 * 5

        self.action_size = 4

        self.q_table = np.zeros(
            (self.state_size, self.action_size)
        )

        self.epsilon = EPSILON

    # --------------------------------------------------------
    # Epsilon-greedy action selection
    # --------------------------------------------------------

    def select_action(self, state, valid_actions):

        if random.random() < self.epsilon:

            return random.choice(valid_actions)

        q_values = self.q_table[state, valid_actions]

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

        # Switch after 20 seconds of green
        if environment.elapsed_green >= 20:
            return 3

        return 2


# ============================================================
# 5. RUN ONE EPISODE
# ============================================================

def run_episode(environment, agent=None, training=False):

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

    # Average waiting time estimate in seconds per arrival
    if environment.total_arrivals > 0:

        average_wait = (
            environment.total_waiting_time
            / environment.total_arrivals
        )

    else:
        average_wait = 0

    average_queue = (
        environment.total_waiting_time
        / SIMULATION_TIME
    )

    return {
        "reward": total_reward,
        "average_wait": average_wait,
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
            agent,
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

    print("========== TRAINING COMPLETED ==========")

    np.save(
        os.path.join(OUTPUT_DIR, "q_table.npy"),
        agent.q_table
    )

    return agent, rewards


# ============================================================
# 7. EVALUATION
# ============================================================

def evaluate_controller(agent=None, controller_name="RL"):

    environment = TrafficEnvironment()

    results = []

    # Disable exploration during RL evaluation
    if agent is not None:
        original_epsilon = agent.epsilon
        agent.epsilon = 0

    for episode in range(TEST_EPISODES):

        result = run_episode(
            environment,
            agent,
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

    print(f"\n========== {controller_name} RESULTS ==========")

    print(f"Average Reward       : {average_reward:.2f}")
    print(f"Average Waiting Time: {average_wait:.2f} seconds")
    print(f"Average Queue Length : {average_queue:.2f} vehicles")
    print(f"Average Vehicles Served: {average_served:.2f}")

    return results


# ============================================================
# 8. SAVE RESULTS
# ============================================================

def save_results(filename, results):

    path = os.path.join(OUTPUT_DIR, filename)

    with open(path, "w", newline="") as file:

        writer = csv.DictWriter(
            file,
            fieldnames=[
                "episode",
                "reward",
                "average_wait",
                "average_queue",
                "arrivals",
                "served"
            ]
        )

        writer.writeheader()
        writer.writerows(results)

    print(f"Results saved: {path}")


# ============================================================
# 9. MDP EXAMPLE FROM TASK 2
# ============================================================

def display_mdp_example():

    print("\n========== MDP FORMULATION ==========")

    print("MDP = (S, A, P, R, gamma)")

    print("\nSample states:")
    print("S1 = Low traffic on all approaches")
    print("S2 = High North-South traffic")
    print("S3 = Medium traffic condition")

    print("\nActions:")
    print("A0 = North-South green")
    print("A1 = East-West green")
    print("A2 = Extend current green")
    print("A3 = Switch phase")

    print("\nIllustrative transition probabilities:")
    print("-----------------------------------------")
    print("Current state: S2")
    print("Action: A2 (Extend current green)")
    print("-----------------------------------------")
    print("Next State     Probability     Reward")
    print("S1             0.20             +5")
    print("S2             0.20             -2")
    print("S3             0.60             +2")

    probability_sum = 0.20 + 0.20 + 0.60

    expected_reward = (
        0.20 * 5
        + 0.20 * (-2)
        + 0.60 * 2
    )

    print("-----------------------------------------")
    print("Probability sum:", probability_sum)
    print("Expected immediate reward:", expected_reward)

    # Numerical example with future state values
    gamma = 0.90

    V_s1 = 10
    V_s2 = 4

    expected_return = (
        expected_reward
        + gamma * (
            0.20 * V_s1
            + 0.20 * V_s2
            + 0.60 * 0
        )
    )

    print("Expected return with future values:", expected_return)

    print(
        "\nNote: These transition probabilities and rewards "
        "are illustrative examples for the MDP explanation. "
        "They are not learned from the traffic simulation."
    )


# ============================================================
# 10. VISUALIZATION
# ============================================================

def plot_training_rewards(rewards):

    plt.figure(figsize=(10, 5))

    plt.plot(
        rewards,
        label="Episode Reward"
    )

    # Moving average
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


def plot_comparison(rl_results, fixed_results):

    labels = [
        "Average Reward",
        "Average Waiting Time",
        "Average Queue Length",
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

    plt.figure(figsize=(10, 5))

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

    plt.xticks(x, labels, rotation=10)
    plt.ylabel("Measured Value")
    plt.title("RL Controller vs Fixed-Time Controller")

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
# 11. MAIN PROGRAM
# ============================================================

if __name__ == "__main__":

    print("SMART TRAFFIC SIGNAL REINFORCEMENT LEARNING")
    print("============================================")

    print("\nState space size: 810")
    print("Action space size: 4")
    print("Learning algorithm: Q-Learning")
    print("Discount factor:", DISCOUNT_FACTOR)

    display_mdp_example()

    # Train RL agent
    trained_agent, training_rewards = train_agent()

    # Evaluate trained RL controller
    rl_results = evaluate_controller(
        trained_agent,
        "Q-LEARNING CONTROLLER"
    )

    # Evaluate fixed-time baseline
    fixed_controller = FixedTimeController()

    fixed_results = evaluate_controller(
        None,
        "FIXED-TIME CONTROLLER"
    )

    # Save evaluation results
    save_results(
        "rl_evaluation.csv",
        rl_results
    )

    save_results(
        "fixed_time_evaluation.csv",
        fixed_results
    )

    # Generate plots
    plot_training_rewards(training_rewards)

    plot_comparison(
        rl_results,
        fixed_results
    )

    print("\n========== PROGRAM COMPLETED ==========")
    print("Q-table and evaluation files saved in:")
    print(OUTPUT_DIR)
