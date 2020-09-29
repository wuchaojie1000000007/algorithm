import numpy as np
import matplotlib.pyplot as plt


# do not use
class CoinEnv:
    def __init__(self, win_rate):
        self.coin = 1
        self.win_rate = win_rate

    def win(self):
        return np.random.random() < self.win_rate

    def action(self, num_to_bet):
        win = num_to_bet
        if win:
            self.coin += num_to_bet
        else:
            self.coin -= num_to_bet

    def reward(self):
        if self.coin == 100:
            return 1
        else:
            return 0


win_rate = 0.55
threshold = 0.01
gamma = 0.9
action_value = np.zeros(101)
action_value[100] = 0


def value_iterate():
    delta = 100
    while delta > threshold:
        d = 0
        # update each state
        for s in range(1, 100):
            old_value = action_value[s]
            new_value = -999999999

            for a in range(1, min(s, 100 - s) + 1):
                r = 1 if (s + a) == 100 else 0
                v = (1 - win_rate) * (0 + action_value[s - a]) + win_rate * (r + gamma * action_value[s + a])
                new_value = max(new_value, v)

            action_value[s] = new_value
            d = max(d, abs(old_value - new_value))

        delta = d


def policy():
    ans = np.zeros(100)
    for s in range(1, 100):
        best_action = -1
        best_value = -999999999
        for a in range(1, min(s, 100 - s) + 1):
            r = 1 if (s + a) == 100 else 0
            v = (1 - win_rate) * (0 + action_value[0]) + win_rate * (r + action_value[s + a])
            if v > best_value:
                best_value = v
                best_action = a
        ans[s] = best_action
    return ans


value_iterate()
ans = policy()
plt.xkcd()
plt.plot(np.arange(101), action_value, label="action_value")
# plt.plot(np.arange(100), ans, label="policy")

plt.show()