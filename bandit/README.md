# Multi-Armed Bandit

There are 4 slot machines. Each machine has a different chance to win.
The agent does not know these chances.

The agent can only pull a lever and see what happens.
It wins (1) or it loses (0).
From this, the agent must find the best machine.

## The setup

These are the real win chances:

```
TRUE_RATES = [0.25, 0.45, 0.30, 0.55]
```

The agent cannot see this list. Arm 3 is the best arm.

At every pull, the agent has two choices.

- **exploit**: pull the machine that looks best right now
- **explore**: pull another machine to learn about it

`epsilon` is the chance to explore.
If epsilon is 0.1, the agent explores 1 time out of 10.
If epsilon is 0.0, the agent never explores.

## Two numbers that are easy to mix up

One **game** is 1000 pulls.
In Test 2 below, I played the game 300 times and took the average.

The agent forgets everything between games.
So 300 is not extra learning. It only takes luck out of the answer.

## How to run

```bash
python3 bandit.py
```

You only need Python 3. There is nothing to install.

## Test 1: one game with no exploration

This is one single game, not an average.
I set epsilon to 0.0, so the agent never explores.

```
[1] One run with no exploration (epsilon = 0.0)
----------------------------------------------------
  Arm         Real     Guess     Pulls
  0           0.25      0.00         0
  1           0.45      0.45      1000
  2           0.30      0.00         0
  3           0.55      0.00         0  <- the best one

  Total reward: 455 / 1000
```

Arm 1 won on the first pull. After that, the agent used arm 1 every time.
It pulled arm 1 for all 1000 pulls.

Arm 3 is the best machine, but the agent never tried it.
So the guess for arm 3 stays at 0.00.

A guess of 0.00 does not mean "this arm is bad".
It means "I do not know this arm".
But the code cannot see the difference between those two.
So it thinks the arm is bad, it never tries it, and it never learns.

The code is not broken.
It followed the rule "pick the best one" every single time.
Only we can see the mistake, because only we can see `TRUE_RATES`.

## Test 2: different epsilon values

Here I played 300 games for each epsilon and took the average.

```
[2] Compare epsilon values (average of 300 runs)
----------------------------------------------------
  epsilon       reward    best arm
  0.0              419       35.6%  #########
  0.01             478       56.9%  ##############
  0.1              513       75.0%  ###################
  0.3              493       69.7%  #################
  1.0              389       24.9%  ######

  Best possible: 550   /  Random guessing: 388
```

**reward** = points from one game of 1000 pulls.

**best arm** = how often the agent picked arm 3, the best one.

Both ends of the table are bad.
`0.0` never explores. `1.0` never uses what it learned.
The good values are in the middle.

## What I learned

**1. No exploration means the agent can get stuck.**
With epsilon 0.0, three arms were never pulled even one time.
Their guess stayed at 0.00, so the agent never had a reason to try them.

**2. Only exploring is just as bad.**
With epsilon 1.0, the score was 389.
Random guessing is 388. So all that learning was worth almost nothing.

**3. A little exploration is enough.**
With epsilon 0.1, the score was 513, the best in the table.
The agent only needs to try something new 1 time out of 10.
