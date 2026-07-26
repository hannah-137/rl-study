"""
UCB - Upper Confidence Bound

epsilon-greedy explores by chance.
It rolls a dice, and sometimes it picks a random arm.
This is not smart. It can pick an arm that it already knows very well.

UCB explores on purpose.
It gives a bonus to every arm that we have not pulled many times.
Then it picks the arm with the highest score.

    score = guess + bonus

    bonus = c * sqrt( log(t + 1) / counts[arm] )

    counts[arm] is small  ->  the bonus is big
    counts[arm] is big    ->  the bonus is small
    t goes up             ->  every bonus goes up a little

So an arm that we ignored for a long time slowly looks better again.
UCB never uses epsilon. There is no random choice at all.

`c` decides how big the bonus is.
A big c means more exploration.

This file compares epsilon-greedy and UCB.

Run this file:  python3 ucb.py
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
    The bar starts at "random guessing" and ends at "best possible"."""
    low = sum(TRUE_RATES) / N_ARMS * steps
    high = max(TRUE_RATES) * steps
    frac = (reward - low) / (high - low)
    return "#" * round(frac * 30)


# --- strategies --------------------------------------------------------
# A strategy is a function. We give it what the agent knows.
# It gives back the arm to pull.
#
#   values = the guess for each arm
#   counts = how many times we pulled each arm
#   t      = the pull number (t = 0, 1, 2, ...)

def epsilon_greedy(eps_fn):
    """Pick a random arm sometimes. Pick the best arm the rest of the time.
    eps_fn(t) gives the epsilon to use at pull number t."""
    def pick(values, counts, t):
        if random.random() < eps_fn(t):
            return random.randrange(N_ARMS)
        return argmax(values)
    return pick


def ucb(c):
    """Give a bonus to arms that we did not pull often. Then pick the best score.

    At the start, counts[arm] is 0 for every arm.
    We cannot divide by 0, so we use a bonus of infinity.
    This makes the agent pull every arm one time first.
    """
    def pick(values, counts, t):
        scores = []
        for arm in range(N_ARMS):
            if counts[arm] == 0:
                scores.append(float("inf"))
            else:
                bonus = c * math.sqrt(math.log(t + 1) / counts[arm])
                scores.append(values[arm] + bonus)
        return argmax(scores)
    return pick


def fixed(value):
    """An epsilon that never changes."""
    return lambda t: value


def decay_slow(t):
    """An epsilon that goes down slowly: 1.00, 0.71, 0.58, 0.50 ..."""
    return max(MIN_EPS, 1.0 / math.sqrt(t + 1))


STRATEGIES = [
    ("epsilon 0.1", epsilon_greedy(fixed(0.1))),
    ("epsilon 1/sqrt(t)", epsilon_greedy(decay_slow)),
    ("UCB c=0.5", ucb(0.5)),
    ("UCB c=1", ucb(1.0)),
    ("UCB c=2", ucb(2.0)),
]


# --- the agent ---------------------------------------------------------

def run(strategy, steps=STEPS):
    """Play the game one time. Pull the levers `steps` times.

    Only the first line inside the loop is new.
    The learning part is the same as bandit.py.
    """
    counts = [0] * N_ARMS
    values = [0.0] * N_ARMS
    total_reward = 0
    early_hits = 0
    late_hits = 0

    for t in range(steps):
        arm = strategy(values, counts, t)

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


def average(strategy, steps, runs):
    """Play the game many times. Then take the average.
    This way, good luck and bad luck do not change the answer."""
    random.seed(42)
    total, early, late = 0, 0, 0
    for _ in range(runs):
        r, e, l = run(strategy, steps)
        total += r
        early += e
        late += l
    return total / runs, early / (runs * WINDOW) * 100, late / (runs * WINDOW) * 100


# --- experiments -------------------------------------------------------

def experiment_1():
    print(f"\n[1] epsilon-greedy vs UCB, over {STEPS} pulls ({RUNS} runs)")
    print("-" * 60)
    print(f"  {'strategy':<18}{'reward':>9}{'early':>11}{'late':>11}")
    print(f"  {'':<18}{'':>9}{'first 100':>11}{'last 100':>11}")
    for name, strategy in STRATEGIES:
        reward, early, late = average(strategy, STEPS, RUNS)
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
    for name, strategy in STRATEGIES:
        row = ""
        for steps, runs in horizons:
            reward, _, _ = average(strategy, steps, runs)
            row += f"{reward / steps:>10.3f}"
        print(f"  {name:<18}{row}")
    print(f"\n  Best possible: {max(TRUE_RATES):.3f}"
          f"   /  Random guessing: {sum(TRUE_RATES) / N_ARMS:.3f}")


if __name__ == "__main__":
    print("=" * 60)
    print("  UCB vs epsilon-greedy on the multi-armed bandit")
    print("=" * 60)

    experiment_1()
    experiment_2()

    print("\n" + "=" * 60)
    print("  What to look at")
    print("=" * 60)
    print("  1. UCB uses no epsilon and no dice.")
    print("     It only uses the bonus. Check the 'early' column.")
    print("     Does UCB find the best arm faster?")
    print()
    print("  2. 'c' sets the size of the bonus.")
    print("     A big c explores more. A small c explores less.")
    print("     Check which c is best here.")
    print()
    print("  3. Look at the long game in experiment 2.")
    print("     A fixed epsilon stops going up, because it wastes")
    print("     the same share of pulls forever.")
    print("     Does UCB stop too, or does it keep going up?")
    print("=" * 60)
