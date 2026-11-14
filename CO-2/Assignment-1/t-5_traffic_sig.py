
# ============================================================
# TASK 5: EXPERIMENTAL EVALUATION
# Fixed-Time vs Greedy Bandit vs Epsilon-Greedy Bandit
# ============================================================

import numpy as np

def run_task5_experiment():

    # Experiment configuration
    SEED = 42
    CYCLES = 120
    STEP = 5
    CYCLE_DURATION = 60
    SERVICE_RATE = 1.2
    EPSILON = 0.1

    ACTIONS = [15, 25, 30, 35, 45]
    N_ARMS = len(ACTIONS)

    # --------------------------------------------------------
    # Generate common traffic arrivals for all strategies
    # --------------------------------------------------------

    base_rng = np.random.default_rng(SEED)
    traffic_arrivals = []

    for cycle in range(CYCLES):

        # Change traffic demand halfway through the experiment
        if cycle < CYCLES // 2:
            rates = (2.8, 1.8)
        else:
            rates = (1.8, 2.8)

        cycle_data = []

        for interval in range(12):
            ns_arrivals = base_rng.poisson(rates[0])
            ew_arrivals = base_rng.poisson(rates[1])

            cycle_data.append(
                (ns_arrivals, ew_arrivals)
            )

        traffic_arrivals.append(cycle_data)

    # --------------------------------------------------------
    # Evaluate one control strategy
    # --------------------------------------------------------

    def evaluate(strategy, random_seed=None):

        rng = (
            np.random.default_rng(random_seed)
            if random_seed is not None
            else None
        )

        estimates = np.zeros(N_ARMS)
        counts = np.zeros(N_ARMS, dtype=int)

        total_reward = 0.0
        total_queue_area = 0.0
        total_arrivals = 0

        for cycle in range(CYCLES):

            # Action selection
            if strategy == "fixed":
                action = 2  # 30-second NS green

            elif strategy == "greedy":
                action = int(np.argmax(estimates))

            elif strategy == "epsilon_greedy":

                if rng.random() < EPSILON:
                    action = int(rng.integers(N_ARMS))
                else:
                    action = int(np.argmax(estimates))

            else:
                raise ValueError("Unknown strategy")

            counts[action] += 1

            # Queue initialization for this trial
            q_ns = 0
            q_ew = 0

            green_ns = ACTIONS[action]
            green_ew = CYCLE_DURATION - green_ns

            cycle_queue_sum = 0
            cycle_queue_area = 0
            cycle_arrivals = 0

            # Simulate 12 intervals of 5 seconds
            for t, (arr_ns, arr_ew) in enumerate(
                traffic_arrivals[cycle]
            ):

                q_ns += arr_ns
                q_ew += arr_ew

                cycle_arrivals += arr_ns + arr_ew

                # Calculate green time in this interval
                ns_green = max(
                    0,
                    min(STEP, green_ns - t * STEP)
                )

                ew_green = max(
                    0,
                    min(STEP, green_ew - t * STEP)
                )

                # Serve vehicles
                served_ns = min(
                    q_ns,
                    int(SERVICE_RATE * ns_green)
                )

                served_ew = min(
                    q_ew,
                    int(SERVICE_RATE * ew_green)
                )

                q_ns -= served_ns
                q_ew -= served_ew

                current_queue = q_ns + q_ew

                cycle_queue_sum += current_queue

                cycle_queue_area += (
                    current_queue * STEP
                )

            # Reward: negative average queue
            reward = -(cycle_queue_sum / 12)

            total_reward += reward
            total_queue_area += cycle_queue_area
            total_arrivals += cycle_arrivals

            # Incremental mean update for Bandit
            estimates[action] += (
                reward - estimates[action]
            ) / counts[action]

        average_reward = total_reward / CYCLES

        average_waiting_time = (
            total_queue_area / max(total_arrivals, 1)
        )

        average_queue = (
            total_queue_area /
            (CYCLES * CYCLE_DURATION)
        )

        return (
            average_reward,
            average_waiting_time,
            average_queue,
            counts
        )

    # --------------------------------------------------------
    # Run all three experiments
    # --------------------------------------------------------

    fixed = evaluate("fixed")
    greedy = evaluate("greedy")
    epsilon_greedy = evaluate(
        "epsilon_greedy",
        random_seed=44
    )

    # --------------------------------------------------------
    # Display results
    # --------------------------------------------------------

    results = {
        "Fixed-Time": fixed,
        "Greedy Bandit": greedy,
        "Epsilon-Greedy": epsilon_greedy
    }

    print("\n========== TASK 5 EXPERIMENT RESULTS ==========")

    for name, result in results.items():

        reward, waiting, queue, counts = result

        print(f"\n{name}")
        print(f"Average Reward: {reward:.3f}")
        print(f"Average Waiting Time: {waiting:.3f} seconds")
        print(f"Average Queue Length: {queue:.3f}")
        print(f"Action Counts: {counts.tolist()}")

    print("\nAction order:", ACTIONS)
    print("Experiment completed.")


if __name__ == "__main__":
    run_task5_experiment()
