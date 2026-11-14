
"""
MLA03 - Reinforcement Learning
Assignment 1 - Task 1
Smart Traffic Signal Control using Q-Learning

Includes:
1. Traffic intersection simulation
2. RL framework: Agent, Environment, State, Action, Reward
3. Q-Learning algorithm
4. Safety constraints
5. Fixed-time traffic controller
6. Training and evaluation
7. Performance comparison
8. CSV and graph generation

Install:
pip install numpy matplotlib

Run:
python task1_traffic_rl.py
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

LEARNING_RATE = 0.1
DISCOUNT_FACTOR = 0.95

EPSILON = 1.0
EPSILON_MIN = 0.05
EPSILON_DECAY = 0.99

OUTPUT_DIR = "task1_results"

random.seed(SEED)
np.random.seed(SEED)

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================
# 2. TRAFFIC ENVIRONMENT
# ============================================================

class TrafficEnvironment:

    def __init__(self):

        self.rows = ["North", "South", "East", "West"]

        # Vehicle arrival rates per second
        self.arrival_rates = [
            0.32, 0.30, 0.38, 0.34
        ]

        self.service_capacity = 2

        self.reset(SEED)

    def reset(self, seed=None):

        self.rng = np.random.default_rng(seed)

        # Queue order: North, South, East, West
        self.queues = [0, 0, 0, 0]

        # 0 = North-South
        # 1 = East-West
        self.phase = 0

        self.elapsed_green = 0
        self.simulation_time = 0

        self.total_waiting_time = 0
        self.total_queue_time = 0

        self.total_arrivals = 0
        self.total_departures = 0

        self.steps = 0

        return self.get_state()

    # --------------------------------------------------------
    # STATE REPRESENTATION
    # --------------------------------------------------------

    def classify_queue(self, queue):

        if queue <= 4:
            return 0       # Low

        elif queue <= 9:
            return 1       # Medium

        else:
            return 2       # High

    def get_state(self):

        queue_states = [
            self.classify_queue(q)
            for q in self.queues
        ]

        # Five elapsed green categories
        if self.elapsed_green < 5:
            elapsed_state = 0
        elif self.elapsed_green < 10:
            elapsed_state = 1
        elif self.elapsed_green < 20:
            elapsed_state = 2
        elif self.elapsed_green < 30:
            elapsed_state = 3
        else:
            elapsed_state = 4

        # Total states = 3^4 * 2 * 5 = 810
        state = 0

        for q in queue_states:
            state = state * 3 + q

        state = state * 2 + self.phase
        state = state * 5 + elapsed_state

        return state

    # --------------------------------------------------------
    # ACTION SPACE
    # --------------------------------------------------------

    # 0 = Select North-South green
    # 1 = Select East-West green
    # 2 = Extend current green
    # 3 = Switch to next phase

    def valid_actions(self):

        valid = []

        # Select North-South
        if self.phase == 0:
            if self.elapsed_green + GREEN_STEP <= MAX_GREEN:
                valid.append(0)

        elif self.elapsed_green >= MIN_GREEN:
            valid.append(0)

        # Select East-West
        if self.phase == 1:
            if self.elapsed_green + GREEN_STEP <= MAX_GREEN:
                valid.append(1)

        elif self.elapsed_green >= MIN_GREEN:
            valid.append(1)

        # Extend current green
        if self.elapsed_green + GREEN_STEP <= MAX_GREEN:
            valid.append(2)

        # Switch to opposite phase
        if self.elapsed_green >= MIN_GREEN:
            valid.append(3)

        return valid

    # --------------------------------------------------------
    # VEHICLE ARRIVALS
    # --------------------------------------------------------

    def generate_arrivals(self, duration):

        for i in range(4):

            arrivals = self.rng.poisson(
                self.arrival_rates[i] * duration
            )

            self.queues[i] += arrivals
            self.total_arrivals += arrivals

    # --------------------------------------------------------
    # WAITING TIME CALCULATION
    # --------------------------------------------------------

    def advance_time(self, duration):

        queue_total = sum(self.queues)

        # Vehicle-seconds accumulated during this interval
        waiting = queue_total * duration

        self.total_waiting_time += waiting
        self.total_queue_time += waiting

        self.simulation_time += duration

        self.generate_arrivals(duration)

        return waiting

    # --------------------------------------------------------
    # VEHICLE SERVICE
    # --------------------------------------------------------

    def serve_vehicles(self):

        served = 0

        if self.phase == 0:

            # North-South green
            approaches = [0, 1]

        else:

            # East-West green
            approaches = [2, 3]

        for road in approaches:

            vehicles = min(
                self.queues[road],
                self.service_capacity
            )

            self.queues[road] -= vehicles
            served += vehicles

        self.total_departures += served

        return served

    # --------------------------------------------------------
    # REWARD FUNCTION
    # --------------------------------------------------------

    def calculate_reward(self, waiting, served):

        total_queue = sum(self.queues)

        # Reward:
        # -0.5 * queue
        # -0.02 * average waiting vehicle count
        # +2 * vehicles served

        reward = (
            -0.5 * total_queue
            -0.02 * (waiting / GREEN_STEP)
            +2 * served
        )

        return reward

    # --------------------------------------------------------
    # ENVIRONMENT STEP
    # --------------------------------------------------------

    def step(self, action):

        if action not in self.valid_actions():

            raise ValueError(
                "Unsafe or invalid traffic signal action"
            )

        old_phase = self.phase

        # Determine selected phase
        if action == 0:
            target_phase = 0

        elif action == 1:
            target_phase = 1

        elif action == 2:
            target_phase = self.phase

        else:
            target_phase = 1 - self.phase

        # Apply yellow/all-red clearance before switching
        if target_phase != old_phase:

            self.advance_time(YELLOW_TIME)

            self.phase = target_phase
            self.elapsed_green = 0

        # Apply green signal
        waiting = self.advance_time(GREEN_STEP)

        # Serve vehicles on the active approaches
        served = self.serve_vehicles()

        self.elapsed_green += GREEN_STEP
        self.steps += 1

        reward = self.calculate_reward(
            waiting,
            served
        )

        done = (
            self.simulation_time >= SIMULATION_TIME
        )

        next_state = self.get_state()

        return next_state, reward, done


# ============================================================
# 3. Q-LEARNING AGENT
# ============================================================

class QLearningAgent:

    def __init__(self):

        self.q_table = np.zeros((810, 4))

        self.alpha = LEARNING_RATE
        self.gamma = DISCOUNT_FACTOR

        self.epsilon = EPSILON

    # --------------------------------------------------------
    # ACTION SELECTION
    # --------------------------------------------------------

    def select_action(self, state, valid_actions):

        # Exploration
        if random.random() < self.epsilon:

            return random.choice(valid_actions)

        # Exploitation
        values = self.q_table[state, valid_actions]

        best_value = np.max(values)

        best_actions = [
            action
            for action in valid_actions
            if self.q_table[state, action] == best_value
        ]

        return random.choice(best_actions)

    # --------------------------------------------------------
    # Q-LEARNING UPDATE
    # --------------------------------------------------------

    def learn(
        self,
        state,
        action,
        reward,
        next_state,
        done,
        next_valid_actions
    ):

        current_value = self.q_table[state, action]

        if done or not next_valid_actions:

            target = reward

        else:

            future_value = max(
                self.q_table[
                    next_state, next_valid_actions
                ]
            )

            target = (
                reward
                + self.gamma * future_value
            )

        # Bellman Q-learning update
        self.q_table[state, action] = (
            current_value
            + self.alpha * (
                target - current_value
            )
        )

    # --------------------------------------------------------
    # EPSILON DECAY
    # --------------------------------------------------------

    def update_epsilon(self):

        self.epsilon = max(
            EPSILON_MIN,
            self.epsilon * EPSILON_DECAY
        )


# ============================================================
# 4. FIXED-TIME TRAFFIC CONTROLLER
# ============================================================

class FixedTimeController:

    def select_action(self, environment):

        # Fixed green duration = 20 seconds

        if environment.elapsed_green >= 20:

            # Switch to the opposite phase
            return 3

        # Continue current green
        return 2


# ============================================================
# 5. EPISODE SIMULATION
# ============================================================

def run_episode(
    environment,
    controller,
    seed,
    training=False
):

    state = environment.reset(seed)

    total_reward = 0
    done = False

    while not done:

        valid_actions = environment.valid_actions()

        if training:

            action = controller.select_action(
                state,
                valid_actions
            )

        else:

            # Fixed-time controller
            if isinstance(
                controller,
                FixedTimeController
            ):

                action = controller.select_action(
                    environment
                )

            # Greedy Q-learning evaluation
            else:

                values = controller.q_table[
                    state, valid_actions
                ]

                best_value = np.max(values)

                best_actions = [
                    a for a in valid_actions
                    if controller.q_table[state, a]
                    == best_value
                ]

                action = random.choice(best_actions)

        next_state, reward, done = environment.step(
            action
        )

        total_reward += reward

        if training:

            next_valid_actions = (
                environment.valid_actions()
            )

            controller.learn(
                state,
                action,
                reward,
                next_state,
                done,
                next_valid_actions
            )

        state = next_state

    # Calculate evaluation metrics
    average_wait = (
        environment.total_waiting_time
        / max(environment.total_arrivals, 1)
    )

    average_queue = (
        environment.total_queue_time
        / max(environment.simulation_time, 1)
    )

    return {
        "reward": total_reward,
        "average_wait": average_wait,
        "average_queue": average_queue,
        "arrivals": environment.total_arrivals,
        "served": environment.total_departures,
        "steps": environment.steps,
        "simulation_time": environment.simulation_time
    }


# ============================================================
# 6. TRAIN Q-LEARNING AGENT
# ============================================================

def train_agent():

    print("\n====================================")
    print("      Q-LEARNING TRAINING")
    print("====================================")

    environment = TrafficEnvironment()
    agent = QLearningAgent()

    training_rewards = []

    for episode in range(EPISODES):

        metrics = run_episode(
            environment,
            agent,
            seed=SEED + episode,
            training=True
        )

        training_rewards.append(
            metrics["reward"]
        )

        agent.update_epsilon()

        if (episode + 1) % 25 == 0:

            recent_average = np.mean(
                training_rewards[-25:]
            )

            print(
                f"Episode {episode + 1}/{EPISODES} | "
                f"Average Reward: {recent_average:.2f} | "
                f"Epsilon: {agent.epsilon:.3f}"
            )

    # Save Q-table
    np.save(
        os.path.join(OUTPUT_DIR, "q_table.npy"),
        agent.q_table
    )

    print("\nTraining completed.")
    print("Q-table saved successfully.")

    return agent, training_rewards


# ============================================================
# 7. EVALUATE BOTH CONTROLLERS
# ============================================================

def evaluate_controllers(agent):

    print("\n====================================")
    print("       PERFORMANCE EVALUATION")
    print("====================================")

    rl_environment = TrafficEnvironment()
    fixed_environment = TrafficEnvironment()

    fixed_controller = FixedTimeController()

    rl_results = []
    fixed_results = []

    for episode in range(TEST_EPISODES):

        # Use identical random seeds for comparison
        seed = 10000 + episode

        rl_metrics = run_episode(
            rl_environment,
            agent,
            seed,
            training=False
        )

        fixed_metrics = run_episode(
            fixed_environment,
            fixed_controller,
            seed,
            training=False
        )

        rl_results.append(rl_metrics)
        fixed_results.append(fixed_metrics)

    return rl_results, fixed_results


# ============================================================
# 8. DISPLAY PERFORMANCE
# ============================================================

def calculate_average(results, key):

    return np.mean([
        result[key] for result in results
    ])


def display_comparison(rl_results, fixed_results):

    metrics = [
        "reward",
        "average_wait",
        "average_queue",
        "steps"
    ]

    print("\n====================================")
    print("       FINAL PERFORMANCE")
    print("====================================")

    print(
        f"{'Metric':<25}"
        f"{'Q-Learning':>15}"
        f"{'Fixed-Time':>15}"
    )

    print("-" * 55)

    for metric in metrics:

        rl_value = calculate_average(
            rl_results, metric
        )

        fixed_value = calculate_average(
            fixed_results, metric
        )

        print(
            f"{metric:<25}"
            f"{rl_value:>15.2f}"
            f"{fixed_value:>15.2f}"
        )


# ============================================================
# 9. SAVE EXPERIMENT RESULTS
# ============================================================

def save_results(rl_results, fixed_results):

    filename = os.path.join(
        OUTPUT_DIR,
        "experiment_results.csv"
    )

    fields = [
        "controller",
        "episode",
        "reward",
        "average_wait",
        "average_queue",
        "arrivals",
        "served",
        "steps"
    ]

    with open(
        filename,
        "w",
        newline=""
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=fields
        )

        writer.writeheader()

        for name, results in [
            ("Q-Learning", rl_results),
            ("Fixed-Time", fixed_results)
        ]:

            for i, result in enumerate(results):

                row = {
                    "controller": name,
                    "episode": i + 1,
                    **result
                }

                writer.writerow(row)

    print("\nCSV results saved:", filename)


# ============================================================
# 10. GENERATE GRAPHS
# ============================================================

def generate_graphs(
    training_rewards,
    rl_results,
    fixed_results
):

    # Training reward graph
    plt.figure(figsize=(9, 5))

    plt.plot(
        training_rewards,
        label="Training Reward"
    )

    plt.xlabel("Training Episodes")
    plt.ylabel("Cumulative Reward")
    plt.title("Q-Learning Training Performance")

    plt.legend()
    plt.grid(True)

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            OUTPUT_DIR,
            "training_reward.png"
        )
    )

    plt.show()

    # Comparison graph
    labels = [
        "Average Reward",
        "Average Waiting Time",
        "Average Queue Length"
    ]

    rl_values = [
        calculate_average(rl_results, "reward"),
        calculate_average(rl_results, "average_wait"),
        calculate_average(rl_results, "average_queue")
    ]

    fixed_values = [
        calculate_average(fixed_results, "reward"),
        calculate_average(fixed_results, "average_wait"),
        calculate_average(fixed_results, "average_queue")
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

    plt.xticks(x, labels)
    plt.ylabel("Measured Value")

    plt.title(
        "Q-Learning vs Fixed-Time Traffic Control"
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

    print("\nGraphs saved in:", OUTPUT_DIR)


# ============================================================
# 11. MAIN EXECUTION
# ============================================================

def main():

    print("\n")
    print("============================================")
    print(" SMART TRAFFIC SIGNAL CONTROL USING RL")
    print(" MLA03 - ASSIGNMENT 1 - TASK 1")
    print("============================================")

    print("\nTraffic intersection:")
    print("North, South, East, West")

    print("\nState space: 810")
    print("Action space: 4")
    print("Algorithm: Q-Learning")

    print("\nReward Function:")
    print("R = -0.5 * Queue")
    print("    -0.02 * Average Waiting Queue")
    print("    +2 * Vehicles Served")

    print("\nDiscount Factor:", DISCOUNT_FACTOR)

    print("Learning Rate:", LEARNING_RATE)

    print("Episodes:", EPISODES)

    print("Test Episodes:", TEST_EPISODES)

    # Step 1: Train agent
    agent, training_rewards = train_agent()

    # Step 2: Evaluate controllers
    rl_results, fixed_results = evaluate_controllers(
        agent
    )

    # Step 3: Display comparison
    display_comparison(
        rl_results,
        fixed_results
    )

    # Step 4: Save results
    save_results(
        rl_results,
        fixed_results
    )

    # Step 5: Generate graphs
    generate_graphs(
        training_rewards,
        rl_results,
        fixed_results
    )

    print("\n============================================")
    print("          EXECUTION COMPLETED")
    print("============================================")

    print("\nGenerated files:")

    print("1. q_table.npy")
    print("2. experiment_results.csv")
    print("3. training_reward.png")
    print("4. controller_comparison.png")

    print("\nOutput directory:", OUTPUT_DIR)


if __name__ == "__main__":
    main()
