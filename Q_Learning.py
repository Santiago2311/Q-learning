from enum import Enum
import random
import numpy as np
episodes = 2000

initialX = 0
initialY = 2
targetX = 3
targetY = 1
maxx = 4
maxy = 3
x = 0
y = 0

learning_rate_a = 0.9  # alpha/learning rate
discount_factor_g = 0.9  # gamma/discount factor.

epsilon = 1  # 1 = 100% Random walk
epsilon_decay_rate = 0.0001

rewards_per_episode = np.zeros(episodes)


Dir = Enum("Dir", ["UP", "DOWN", "LEFT", "RIGHT"])

actions = [Dir.UP, Dir.DOWN, Dir.LEFT, Dir.RIGHT]

q = {
    (0, 0): {Dir.UP: 0, Dir.DOWN: 0, Dir.LEFT: 0, Dir.RIGHT: 0},
    (0, 1): {Dir.UP: 0, Dir.DOWN: 0, Dir.LEFT: 0, Dir.RIGHT: 0},
    (0, 2): {Dir.UP: 0, Dir.DOWN: 0, Dir.LEFT: 0, Dir.RIGHT: 0},
    (1, 0): {Dir.UP: 0, Dir.DOWN: 0, Dir.LEFT: 0, Dir.RIGHT: 0},
    (1, 1): {Dir.UP: 0, Dir.DOWN: 0, Dir.LEFT: 0, Dir.RIGHT: 0},
    (1, 2): {Dir.UP: 0, Dir.DOWN: 0, Dir.LEFT: 0, Dir.RIGHT: 0},
    (2, 0): {Dir.UP: 0, Dir.DOWN: 0, Dir.LEFT: 0, Dir.RIGHT: 0},
    (2, 1): {Dir.UP: 0, Dir.DOWN: 0, Dir.LEFT: 0, Dir.RIGHT: 0},
    (2, 2): {Dir.UP: 0, Dir.DOWN: 0, Dir.LEFT: 0, Dir.RIGHT: 0},
    (3, 0): {Dir.UP: 0, Dir.DOWN: 0, Dir.LEFT: 0, Dir.RIGHT: 0},
    (3, 1): {Dir.UP: 0, Dir.DOWN: 0, Dir.LEFT: 0, Dir.RIGHT: 0},
    (3, 2): {Dir.UP: 0, Dir.DOWN: 0, Dir.LEFT: 0, Dir.RIGHT: 0}

}

ACTIONS = {
    Dir.UP: (0, -1),
    Dir.DOWN: (0, 1),
    Dir.LEFT: (-1, 0),
    Dir.RIGHT: (1, 0)
}


def reset():
    global x, y
    x = initialX
    y = initialY
    return (x, y)


def in_target(newx, newy):
    if newx == targetX and newy == targetY:
        return True
    else:
        return False


def in_bound(newx, newy):
    if newx >= 0 and newx < maxx and newy >= 0 and newy < maxy:
        return True
    else:
        return False


def get_reward(newx, newy):
    if not in_bound(newx, newy):
        return -1.0
    if in_target(newx, newy):
        return 1.0
    else:
        return -0.01


def step(action):
    global x, y
    xx, yy = ACTIONS[action]
    done = False
    next_x = x + xx
    next_y = y + yy

    reward = get_reward(next_x, next_y)

    if in_bound(next_x, next_y):
        x, y = next_x, next_y
        if in_target(next_x, next_y):
            done = True

    return (x, y), reward, done


k = 0
for i in range(episodes):
    state = reset()
    terminated = False

    max_q_val = 0

    while (not terminated):
        k += 1

        if np.random.uniform(0, 1) <= epsilon:
            action = random.choice(actions)
        else:
            for a in actions:
                q_val = q[state][a]
                if q_val >= max_q_val:
                    action = a
                    max_q_val = q_val

        new_state, reward, terminated = step(action)

        q[state][action] = q[state][action] + learning_rate_a * (
            reward + discount_factor_g *
            np.max([v for k, v in q[new_state].items()]) - q[state][action]
        )

        state = new_state

        epsilon = max(epsilon - epsilon_decay_rate, 0)

    if (epsilon == 0):
        learning_rate_a = 0.0001

    if reward == 1:
        rewards_per_episode[i] = 1

    if i % 100 == 0:
        k2 = k/100
        print(i, k, k2)
        k = 0
    if i % 500 == 0:
        print("\n\n", q, "\n\n")


print("done")
print(rewards_per_episode)
