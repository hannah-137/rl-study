# Bandit: Decaying Epsilon

`epsilon` is the chance that the agent explores.
In the basic bandit, epsilon never changes. That is a problem.

At the first pull, the agent knows nothing. So it should explore a lot.
At pull 1000, the agent already knows the best arm.
But if epsilon is 0.1, it still wastes 1 pull out of every 10.

The idea is simple.
Make epsilon big at the start. Make it small later.

## The setup

There are 4 slot machines. These are the real win chances:

```
TRUE_RATES = [0.25, 0.45, 0.30, 0.55]
```

The agent cannot see this list. Arm 3 is the best arm.

I tested 5 ways to choose epsilon.

| name | what it does |
|---|---|
| `fixed 0.01` | epsilon is always 0.01 |
| `fixed 0.1` | epsilon is always 0.1 |
| `fixed 0.3` | epsilon is always 0.3 |
| `decay 1/t` | epsilon starts at 1.00 and goes down fast |
| `decay 1/sqrt(t)` | epsilon starts at 1.00 and goes down slowly |

Everything else is the same. Only epsilon changes.

## Two numbers that are easy to mix up

One **game** is 1000 pulls.
I played the game 300 times, and then I took the average.

The agent forgets everything between games.
So 300 is not extra learning. It only takes luck out of the answer.

## How to run

```bash
python3 decay.py
```

You only need Python 3. There is nothing to install.

## Test 1: one game is 1000 pulls

```
[1] Fixed vs decaying, over 1000 pulls (300 runs)
------------------------------------------------------------
  strategy             reward      early       late
                               first 100   last 100
  fixed 0.01              478        38%        68%  #################
  fixed 0.1               513        49%        84%  #######################
  fixed 0.3               493        49%        76%  ###################
  decay 1/t               491        44%        68%  ###################
  decay 1/sqrt(t)         520        53%        85%  ########################

  Best possible: 550   /  Random guessing: 388
```

Two words in this table:

- **early** = how often the agent picked the best arm in the first 100 pulls.
  This shows how fast it learned.
- **late** = how often it picked the best arm in the last 100 pulls.
  This shows how well it used what it learned.

`fixed 0.01` went from 38% to 68%. It learned late.

`decay 1/sqrt(t)` went from 53% to 85%. It learned fast, and then it used it.

**Decay is not always better.**
`decay 1/t` got 491, but `fixed 0.1` got 513.
`decay 1/t` goes down too fast.
After about 100 pulls, epsilon is already at the floor.
Then the agent almost stops exploring, so it can get stuck on a bad arm.

## Test 2: games of different lengths

```
[2] Reward for one pull, with different numbers of pulls
------------------------------------------------------------
  strategy                 200      1000     10000     50000
  fixed 0.01             0.440     0.474     0.526     0.542
  fixed 0.1              0.477     0.514     0.533     0.534
  fixed 0.3              0.473     0.493     0.502     0.502
  decay 1/t              0.468     0.493     0.525     0.535
  decay 1/sqrt(t)        0.491     0.522     0.539     0.546

  Best possible: 0.550   /  Random guessing: 0.388
```

Here I divide the reward by the number of pulls.
Now a short game and a long game are on the same scale.

Read this table from left to right.

`fixed 0.1` goes 0.477, 0.514, 0.533, 0.534. Then it stops.
It throws away 1 pull out of 10 forever. So it can never reach 0.550.

`fixed 0.01` goes 0.440, 0.474, 0.526, 0.542. It keeps going up.
It learns late, but it wastes very little.

So the best fixed value is not always the same one.
In a short game (200 pulls), 0.1 beats 0.01.
In a long game (50000 pulls), 0.01 beats 0.1.

`decay 1/sqrt(t)` is first in all four columns.

## What I learned

**1. A fixed epsilon can never reach the top score.**
If epsilon is 0.1, then 1 pull out of 10 is random forever.
That cost never goes away. `fixed 0.1` stopped at 0.534, not 0.550.

**2. Decay is not always better. The speed matters.**
`decay 1/t` got 491, but `fixed 0.1` got 513.
`1/t` hits the floor after about 100 pulls. Then the agent gets stuck.

**3. A fixed value only works if you know the game length.**
In a 200 pull game, 0.1 wins. In a 50000 pull game, 0.01 wins.
A decay schedule works without knowing the length.
