import numpy as np
from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt


class BlackJackEnv:
    def __init__(self):
        self.player = np.array(self.fill_hand())
        self.dealer = np.array(self.fill_hand())

    def draw(self):
        cards = np.array((1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 10, 10, 10))
        card = np.random.choice(cards, p=np.ones_like(cards) / cards.size)
        return card

    def fill_hand(self):
        return self.draw(), self.draw()

    # change 1 to 11 without burst
    def usable_ace(self, person):
        if person == 0:
            if 1 in self.player and sum(self.player) <= 11:
                return True
            else:
                return False
        if person == 1:
            if 1 in self.dealer and sum(self.dealer) <= 11:
                return True
            else:
                return False

    def burst(self, person):
        if person == 0:
            return sum(self.player) > 21
        elif person == 1:
            return sum(self.dealer) > 21

    # person: 0 denote player, 1 denote dealer
    def hit(self, person):
        if person == 0:
            self.player = np.append(self.player, self.draw())
        elif person == 1:
            self.dealer = np.append(self.dealer, self.draw())

    def nature(self):
        nature = self.player.size == 2 and 1 in self.player and 10 in self.player
        return nature

    def dealer_draw(self):
        while sum(self.dealer) < 17:
            self.hit(1)

    def dealer_draw_with_ace(self):
        while sum(self.dealer) < 27:
            self.hit(1)

    def dealer_sum(self):
        return sum(self.dealer)

    def dealer_showing(self):
        return self.dealer[0]

    def player_sum(self):
        return sum(self.player)

    def compare(self, player, dealer):
        if player > dealer:
            return 1
        elif player == dealer:
            return 0
        else:
            return -1

    def simulate(self):
        # current sum [12..21], dealer showing [1..10], usable ace [yes, no]

        self.player = np.array(self.fill_hand())
        self.dealer = np.array(self.fill_hand())

        episode = []
        player_usable_ace = self.usable_ace(0)
        if player_usable_ace:
            while self.player_sum() < 10:
                episode.append([self.player_sum(), self.dealer_showing(), player_usable_ace])
                # action
                episode.append(True)
                self.hit(0)
                episode.append(0)
            if self.player_sum() > 11:
                player_usable_ace = False
        if player_usable_ace:
            episode.append([self.player_sum() + 10, self.dealer_showing(), player_usable_ace])
            episode.append(False)
        if self.player_sum() == 20 or self.player_sum() == 21:
            episode.append([self.player_sum(), self.dealer_showing(), player_usable_ace])
            episode.append(False)
        while not player_usable_ace and self.player_sum() < 20:
            episode.append([self.player_sum(), self.dealer_showing(), player_usable_ace])
            episode.append(True)
            self.hit(0)
            if self.burst(0):
                episode.append(-1)
                return episode
            if self.player_sum() != 20 and self.player_sum() != 21:
                episode.append(0)
        # player is ok

        dealer_usable_ace = self.usable_ace(1)
        if dealer_usable_ace:
            while self.dealer_sum() < 7:
                self.hit(1)
            if self.dealer_sum() > 11:
                dealer_usable_ace = False
        if not dealer_usable_ace and self.dealer_sum() < 17:
            self.dealer_draw()
            if self.burst(1):
                episode.append(1)
                return episode
        # dealer is ok

        # compare
        if player_usable_ace and dealer_usable_ace:
            episode.append(self.compare(self.player_sum(), self.dealer_sum()))
        elif player_usable_ace:
            episode.append(self.compare(self.player_sum() + 10, self.dealer_sum()))
        elif dealer_usable_ace:
            episode.append(self.compare(self.player_sum(), self.dealer_sum() + 10))
        else:
            episode.append(self.compare(self.player_sum(), self.dealer_sum()))

        return episode


action_value = np.random.randn(10, 10, 2)
count = np.zeros(action_value.shape)
runs = 500000
gamma = 0.9
env = BlackJackEnv()
for i in range(runs):
    res = env.simulate()
    index = len(res) - 1
    expectation = 0
    while index > 0:
        value = res[index]
        index -= 2
        (player_sum, dealer_show, usable_ace) = res[index]
        if usable_ace:
            usable_ace = 0
        else:
            usable_ace = 1
        expectation = gamma * expectation + value
        cnt = count[player_sum - 12][dealer_show - 1][usable_ace]
        action_value[player_sum - 12][dealer_show - 1][usable_ace] = \
            (action_value[player_sum - 12][dealer_show - 1][usable_ace] * cnt + expectation) / (cnt + 1)
        count[player_sum - 12][dealer_show - 1][usable_ace] += 1
        index -= 1

res = np.zeros((10, 10))
for j in range(10):
    for k in range(10):
        res[j][k] = action_value[j][k][0]
print(res)