"""
Decaying epsilon - explore early, exploit later

In bandit.py, epsilon never changed. That is a problem.

At the first pull, the agent knows nothing. So it should explore a lot.
At pull 1000, the agent already knows the best arm.
But if epsilon is 0.1, it still wastes 1 pull out of every 10.

The idea is simple.
Make epsilon big at the start. Make it small later.

This file runs two experiments.
  [1] fixed epsilon vs decaying epsilon, over 1000 pulls
  [2] the same strategies, but with more pulls and with fewer pulls

Run this file:  python3 decay.py
"""

import math
import random

TRUE_RATES = [0.25, 0.45, 0.30, 0.55]
N_ARMS = len(TRUE_RATES)
BEST_ARM = TRUE_RATES.index(max(TRUE_RATES))

STEPS = 1000
RUNS = 300
WINDOW = 100      # the first 100 pulls are "early". the last 100 are "late"
MIN_EPS = 0.01    # epsilon never goes lower than this


def pull(arm):
    """Pull one lever. Return 1 if we win. Return 0 if we lose."""
    return 1 if random.random() < TRUE_RATES[arm] else 0


def argmax(xs):
    """Return the place of the biggest value.
    If two values are the same, pick one of them at random."""
    best = max(xs)
    return random.choice([i for i, x in enumerate(xs) if x == best])


def bar_for(reward, steps):
    """Make a bar out of '#' marks.

    The bar does not start at zero.
    It starts at "random guessing" and it ends at "best possible".

    Every strategy is better than random guessing.
    So a bar that starts at zero would make them all look the same.
    This bar shows only the part that we care about.
    """
    low = sum(TRUE_RATES) / N_ARMS * steps      # random guessing
    high = max(TRUE_RATES) * steps              # always pick the best arm
    frac = (reward - low) / (high - low)
    return "#" * round(frac * 30)


# --- epsilon schedules -------------------------------------------------
# Each schedule is a function of t. t is the pull number (t = 0, 1, 2, ...).

def fixed(value):
    """Return a schedule that never changes. bandit.py used this one."""
    return lambda t: value


def decay_fast(t):
    """Epsilon goes 1.00, 0.50, 0.33, 0.25 ... and then stops at MIN_EPS.
    This goes down very fast.
    After about 100 pulls it is already at the floor."""
    return max(MIN_EPS, 1.0 / (t + 1))


def decay_slow(t):
    """Epsilon goes 1.00, 0.71, 0.58, 0.50 ... and then keeps going down.
    This goes down more slowly.
    So the agent explores for a longer time."""
    return max(MIN_EPS, 1.0 / math.sqrt(t + 1))


SCHEDULES = [
    ("fixed 0.01", fixed(0.01)),
    ("fixed 0.1", fixed(0.1)),
    ("fixed 0.3", fixed(0.3)),
    ("decay 1/t", decay_fast),
    ("decay 1/sqrt(t)", decay_slow),
]


# --- the agent ---------------------------------------------------------

def run(eps_fn, steps=STEPS):
    """
    Play the game one time. Pull the levers `steps` times.
    eps_fn(t) tells us which epsilon to use at pull number t.

    The learning part is the same as bandit.py.
    Only epsilon is different.
    """
    counts = [0] * N_ARMS
    values = [0.0] * N_ARMS
    total_reward = 0
    early_hits = 0    # we picked the best arm during the first WINDOW pulls
    late_hits = 0     # we picked the best arm during the last WINDOW pulls

    for t in range(steps):
        epsilon = eps_fn(t)

        if random.random() < epsilon:
            arm = random.randrange(N_ARMS)
        else:
            arm = argmax(values)

        reward = pull(arm)

        counts[arm] += 1
        values[arm] += (reward - values[arm]) / counts[arm]
        total_reward += reward

        if arm == BEST_ARM:
            if t < WINDOW:
                early_hits += 1
            if t >= steps - WINDOW:
                late_hits += 1

    return total_reward, early_hits, late_hits


def average(eps_fn, steps, runs):
    """Play the game many times. Then take the average.
    This way, good luck and bad luck do not change the answer."""
    random.seed(42)
    total, early, late = 0, 0, 0
    for _ in range(runs):
        r, e, l = run(eps_fn, steps)
        total += r
        early += e
        late += l
    return total / runs, early / (runs * WINDOW) * 100, late / (runs * WINDOW) * 100


# --- experiments -------------------------------------------------------

def experiment_1():
    print(f"\n[1] Fixed vs decaying, over {STEPS} pulls ({RUNS} runs)")
    print("-" * 60)
    print(f"  {'strategy':<18}{'reward':>9}{'early':>11}{'late':>11}")
    print(f"  {'':<18}{'':>9}{'first 100':>11}{'last 100':>11}")
    for name, fn in SCHEDULES:
        reward, early, late = average(fn, STEPS, RUNS)
        bar = bar_for(reward, STEPS)
        print(f"  {name:<18}{reward:>9.0f}{early:>10.0f}%{late:>10.0f}%  {bar}")
    print(f"\n  Best possible: {max(TRUE_RATES) * STEPS:.0f}"
          f"   /  Random guessing: {sum(TRUE_RATES) / N_ARMS * STEPS:.0f}")
    print("  The bar starts at random guessing. It ends at best possible.")


def experiment_2():
    """Reward for one pull.
    This lets us compare a short game and a long game."""
    horizons = [(200, 200), (1000, 200), (10000, 50), (50000, 20)]

    print("\n[2] Reward for one pull, with different numbers of pulls")
    print("-" * 60)
    header = "".join(f"{n:>10}" for n, _ in horizons)
    print(f"  {'strategy':<18}{header}")
    for name, fn in SCHEDULES:
        row = ""
        for steps, runs in horizons:
            reward, _, _ = average(fn, steps, runs)
            row += f"{reward / steps:>10.3f}"
        print(f"  {name:<18}{row}")
    print(f"\n  Best possible: {max(TRUE_RATES):.3f}"
          f"   /  Random guessing: {sum(TRUE_RATES) / N_ARMS:.3f}")


if __name__ == "__main__":
    print("=" * 60)
    print("  Epsilon schedules for the multi-armed bandit")
    print("=" * 60)

    experiment_1()
    experiment_2()

    print("\n" + "=" * 60)
    print("  What the numbers say")
    print("=" * 60)
    print("  1. Decaying is not always better.")
    print("     'decay 1/t' goes down too fast.")
    print("     After about 100 pulls it almost never explores.")
    print("     So it can get stuck on a bad arm.")
    print("     Look at epsilon = 0.0 in bandit.py. It is the same problem.")
    print()
    print("  2. 'decay 1/sqrt(t)' explores a lot early. Then it slowly stops.")
    print("     Check if it wins at every length that we test here.")
    print()
    print("  3. A fixed epsilon has a ceiling.")
    print("     If epsilon is 0.1, then 1 pull out of 10 is random forever.")
    print("     So it can never get close to 0.550.")
    print("     A longer game does not help. Look at experiment 2.")
    print()
    print("  4. The best fixed value depends on the length of the game.")
    print("     In a short game, 0.1 may beat 0.01.")
    print("     In a long game, 0.01 may beat 0.1.")
    print("     A decay schedule does not need to know the length first.")
    print("=" * 60)