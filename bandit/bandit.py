"""
Multi-Armed Bandit - a first example of reinforcement learning

There are 4 slot machines.
Each machine has a different chance to win.
The agent does not know these chances.
The agent must pull the levers and find the best machine.

The big problem: exploration vs exploitation
  exploitation: pull the machine that looks best right now
  exploration:  try other machines to learn about them

Run this file:  python3 bandit.py
You can change: TRUE_RATES, EPSILONS, STEPS
"""

import random

# The real win chances. The agent can never see this list.
TRUE_RATES = [0.25, 0.45, 0.30, 0.55]
N_ARMS = len(TRUE_RATES)
BEST_ARM = TRUE_RATES.index(max(TRUE_RATES))

STEPS = 1000      # how many pulls in one run
RUNS = 300        # how many runs to average
EPSILONS = [0.0, 0.01, 0.1, 0.3, 1.0]


def pull(arm):
    """Pull one lever. Return 1 if we win. Return 0 if we lose."""
    return 1 if random.random() < TRUE_RATES[arm] else 0


def argmax(xs):
    """Find where the biggest value is. If there is a tie, pick one at random.

    Why random? At the start all estimates are 0.0.
    So every arm is tied.
    Without this, the agent would always pick arm 0.
    That would make the test unfair.
    """
    best = max(xs)
    return random.choice([i for i, x in enumerate(xs) if x == best])


def run(epsilon, steps=STEPS):
    """
    Play many pulls. Use the epsilon-greedy rule.
    epsilon = the chance to pull a random arm. This is the explore rate.
    """
    counts = [0] * N_ARMS      # how many times we pulled each arm
    values = [0.0] * N_ARMS    # what the agent THINKS each win chance is
    total_reward = 0

    for _ in range(steps):
        # --- 1. Pick an arm ---
        if random.random() < epsilon:
            arm = random.randrange(N_ARMS)       # explore: pick any arm
        else:
            arm = argmax(values)                 # exploit: pick the best one

        # --- 2. Get a reward from the environment ---
        reward = pull(arm)

        # --- 3. Update the estimate. This part is the heart of RL. ---
        # new estimate = old estimate + step size * (what we got - what we expected)
        # The part inside the brackets is the prediction error.
        # If we got more than we expected, the estimate goes up.
        # If we got less, it goes down.
        # Q-learning uses this same idea.
        counts[arm] += 1
        values[arm] += (reward - values[arm]) / counts[arm]

        total_reward += reward

    return counts, values, total_reward


def demo_single_run():
    """Look at one run. Here the agent never explores."""
    random.seed(7)
    counts, values, total = run(0.0)

    print("\n[1] One run with no exploration (epsilon = 0.0)")
    print("-" * 52)
    print(f"  {'Arm':<6}{'Real':>10}{'Guess':>10}{'Pulls':>10}")
    for i in range(N_ARMS):
        mark = "  <- the best one" if i == BEST_ARM else ""
        print(f"  {i:<6}{TRUE_RATES[i]:>10.2f}{values[i]:>10.2f}"
              f"{counts[i]:>10}{mark}")
    print(f"\n  Total reward: {total} / {STEPS}")
    print("  The agent used only one machine.")
    print("  It never tried the others, so their guess stays 0.00 forever.")


def compare_epsilons():
    """Play many runs and take the average. This takes luck out."""
    print(f"\n[2] Compare epsilon values (average of {RUNS} runs)")
    print("-" * 52)
    print(f"  {'epsilon':<10}{'reward':>10}{'best arm':>12}")

    for eps in EPSILONS:
        random.seed(42)
        rewards, best_hits = 0, 0
        for _ in range(RUNS):
            counts, _, total = run(eps)
            rewards += total
            best_hits += counts[BEST_ARM]

        avg_reward = rewards / RUNS
        best_ratio = best_hits / (RUNS * STEPS) * 100
        bar = "#" * round(best_ratio / 4)

        print(f"  {eps:<10}{avg_reward:>10.0f}{best_ratio:>11.1f}%  {bar}")

    print(f"\n  Best possible: {max(TRUE_RATES) * STEPS:.0f}"
          f"   /  Random guessing: {sum(TRUE_RATES) / N_ARMS * STEPS:.0f}")


if __name__ == "__main__":
    print("=" * 52)
    print(f"  Multi-armed bandit - {N_ARMS} machines, {STEPS} pulls per run")
    print("=" * 52)

    demo_single_run()
    compare_epsilons()

    print("\n" + "=" * 52)
    print("  0.0  never explores. It gets stuck on one lucky machine.")
    print("  0.1  mostly exploits, explores a little. This works best.")
    print("  1.0  only random. It never uses what it learned.")
    print("=" * 52)
    print("  Try this: make the numbers close, like [0.50, 0.52, 0.48, 0.51].")
    print("           Then the agent needs much more exploration.")
    print("=" * 52)