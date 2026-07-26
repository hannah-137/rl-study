# Multi-Armed Bandit

A first example of **reinforcement learning**.
It is written in plain Python. No libraries needed.

There are 4 slot machines. Each one has a different chance to win.
The agent does not know these chances.
It can only pull a lever and see the result: win (1) or lose (0).
From this, it must find the best machine.

```
TRUE_RATES = [0.25, 0.45, 0.30, 0.55]   # the agent cannot see this
```

This is the **exploration vs exploitation** problem:

- **exploitation**: pull the machine that looks best right now
- **exploration**: try other machines to learn about them

If you only exploit, you may miss a better machine.
If you only explore, you never use what you learned.

## How to run

```bash
python3 bandit.py
```

You only need Python 3.

## Results

### Test 1 - no exploration (epsilon = 0.0)

```
  Arm         Real     Guess     Pulls
  0           0.25      0.00         0
  1           0.45      0.45      1000
  2           0.30      0.00         0
  3           0.55      0.00         0  <- the best one

  Total reward: 455 / 1000
```

Arm 1 won on the very first pull.
After that, the agent used arm 1 for all 1000 pulls.

Arm 3 is the best machine, but the agent never tried it.
So its guess stays at 0.00.

Here 0.00 does not mean "this arm is bad".
It means "I do not know".
But the code cannot tell the difference.
So it thinks the arm is bad, never tries it, and never learns.

The code is not broken.
It followed the rule "pick the best one" every single time.
Only we can see the mistake, because only we can see `TRUE_RATES`.

### Test 2 - different epsilon values (average of 300 runs)

```
  epsilon       reward    best arm
  0.0              419       35.6%  #########
  0.01             478       56.9%  ##############
  0.1              513       75.0%  ###################
  0.3              493       69.7%  #################
  1.0              389       24.9%  ######

  Best possible: 550   /  Random guessing: 388
```

| epsilon | what it does | result |
|---|---|---|
| 0.0 | never explores | gets stuck (419) |
| 0.1 | explores a little | **best (513)** |
| 1.0 | always random | same as guessing (389) |

Both extremes are bad.
The agent only needs to try something new **1 time out of 10**.
That small change moves the score from 455 up to 513.

## The main code

The learning part is only 5 lines.

```python
# 1. Pick an arm
if random.random() < epsilon:
    arm = random.randrange(N_ARMS)   # explore: pick any arm
else:
    arm = argmax(values)             # exploit: pick the best one

# 2. Get a reward
reward = pull(arm)

# 3. Update the guess
counts[arm] += 1
values[arm] += (reward - values[arm]) / counts[arm]
```

In step 3, `(reward - values[arm])` is the **prediction error**.

- We got more than we expected -> the guess goes up
- We got less than we expected -> the guess goes down

We divide by `counts[arm]`.
So an arm we pulled many times does not move much from one new result.

Q-learning uses this same idea.

## Does the guess become correct?

I ran the code with more and more pulls (epsilon = 0.1).

| pulls | arm 0 | arm 1 | arm 2 | arm 3 |
|---|---|---|---|---|
| **real** | 0.25 | 0.45 | 0.30 | **0.55** |
| 1,000 | 0.250 | 0.559 | 0.292 | 0.569 |
| 10,000 | 0.241 | 0.478 | 0.297 | 0.552 |
| 100,000 | 0.255 | 0.467 | 0.292 | 0.549 |
| 1,000,000 | 0.249 | 0.449 | 0.304 | 0.550 |

Yes, but slowly, and not at the same speed for every arm.
The best arm gets over 90% of the pulls, so its guess gets good fast.
The other arms share the small epsilon, so they stay wrong for a long time.

But the agent does not need perfect numbers.
It only needs the **order** to be right.
At 1,000 pulls, the guess for arm 1 (0.559) was far from the real value (0.45).
That was fine. Arm 3 was still number one, so the choice was still correct.

## Files

```
.
├── README.md
└── bandit.py
```

## Next steps

- [ ] Make epsilon smaller over time (explore early, exploit later)
- [ ] Compare with UCB (Upper Confidence Bound)
- [ ] Move on to Gridworld Q-learning, which adds **states**

## Reference

- Sutton & Barto, *Reinforcement Learning: An Introduction*, Chapter 2