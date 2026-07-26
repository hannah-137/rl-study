# Bandit: UCB (Upper Confidence Bound)

epsilon-greedy explores with a dice roll.
The dice does not look at anything.
It can pick an arm that we already pulled 800 times.
That is a waste.

UCB explores on purpose.
It gives a **bonus** to every arm that we did not pull often.

```
score = guess + bonus

bonus = c * sqrt( log(t + 1) / counts[arm] )
```

Then it picks the arm with the highest score.
There is no dice here. UCB never picks at random.

A small count makes a big bonus.
So an arm that we barely know looks better than it really is.
The agent tries it, the count goes up, and the bonus goes down.

`c` sets the size of the bonus. A big `c` gives more exploration.

### A small example

This is what the numbers can look like after 100 pulls.

| arm | guess | pulls | bonus | **score** |
|---|---|---|---|---|
| arm 1 | 0.50 | 80 | 0.24 | 0.74 |
| arm 3 | 0.45 | 5 | 0.96 | **1.41** |

The guess for arm 1 is higher.
But UCB picks arm 3, because we only pulled it 5 times.
We are not sure about arm 3 yet, so we go and check it.

## The setup

There are 4 slot machines. These are the real win chances:

```
TRUE_RATES = [0.25, 0.45, 0.30, 0.55]
```

The agent cannot see this list. Arm 3 is the best arm.

I compared 5 strategies.

| name | how it explores |
|---|---|
| `epsilon 0.1` | dice, always the same chance |
| `epsilon 1/sqrt(t)` | dice, but the chance goes down over time |
| `UCB c=0.5` | bonus, small |
| `UCB c=1` | bonus, medium |
| `UCB c=2` | bonus, big |

## Two numbers that are easy to mix up

One **game** is 1000 pulls.
I played the game 300 times and took the average.

The agent forgets everything between games.
So 300 is not extra learning. It only takes luck out of the answer.

## How to run

```bash
python3 ucb.py
```

You only need Python 3. There is nothing to install.

## Test 1: one game is 1000 pulls

```
[1] epsilon-greedy vs UCB, over 1000 pulls (300 runs)
------------------------------------------------------------
  strategy             reward      early       late
                               first 100   last 100
  epsilon 0.1             513        49%        84%  #######################
  epsilon 1/sqrt(t)       520        53%        85%  ########################
  UCB c=0.5               532        56%        96%  ###########################
  UCB c=1                 505        44%        84%  ######################
  UCB c=2                 466        35%        63%  ##############

  Best possible: 550   /  Random guessing: 388
```

- **early** = how often the agent picked the best arm in the first 100 pulls.
- **late** = how often it picked the best arm in the last 100 pulls.

`UCB c=0.5` got 532. That is the best score in the table.

Look at the **late** column. `UCB c=0.5` reached 96%.
The two epsilon strategies stopped near 84% and 85%.

This is the big difference.
An epsilon strategy keeps rolling the dice, so it never stops exploring.
`UCB c=0.5` was sure about arm 3, so it almost stopped exploring.

But a bigger `c` is worse, not better.
`UCB c=2` got only 466. Every epsilon strategy beat it.
A big bonus means the agent keeps trying bad arms.

## Test 2: games of different lengths

```
[2] Reward for one pull, with different numbers of pulls
------------------------------------------------------------
  strategy                 200      1000     10000     50000
  epsilon 0.1            0.477     0.514     0.533     0.534
  epsilon 1/sqrt(t)      0.491     0.522     0.539     0.546
  UCB c=0.5              0.503     0.532     0.548     0.550
  UCB c=1                0.466     0.505     0.541     0.548
  UCB c=2                0.431     0.467     0.522     0.541

  Best possible: 0.550   /  Random guessing: 0.388
```

Here I divide the reward by the number of pulls.
Now a short game and a long game are on the same scale.

Read this table from left to right.

`epsilon 0.1` goes 0.477, 0.514, 0.533, 0.534. Then it stops.
It rolls the dice 1 time out of 10 forever, so it can never reach 0.550.

**All three UCB rows keep going up.**
`UCB c=0.5` reached 0.550 at 50000 pulls. That is the top score.
The bonus gets smaller and smaller, so the waste goes away.

`UCB c=2` is slow, but it is going up too: 0.431, 0.467, 0.522, 0.541.
A big `c` is not broken. It just needs a lot more pulls.

## What I learned

**1. UCB explores better than a dice roll.**
`UCB c=0.5` got 532. The best epsilon strategy got 520.
UCB looks at `counts`, so it does not waste pulls on arms it already knows.

**2. A big bonus is not a good bonus.**
`UCB c=2` got 466. Even `epsilon 0.1` beat that.
The best value here was the smallest one I tried, `c=0.5`.

**3. UCB has no ceiling. A fixed epsilon does.**
In a 50000 pull game, `UCB c=0.5` reached 0.550, the top score.
`epsilon 0.1` stopped at 0.534 and stayed there.
