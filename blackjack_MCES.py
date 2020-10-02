import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

sns.set_theme()


# person: 0 denote player, 1 denote dealer
# action: 0 is stay, 1 is hit
# policy: current sum [12..21], dealer showing [1..10], usable ace [0..1]
# usable ace: 0 no, 1 have
class BlackJackEnv:
    def __init__(self):
        self.player = np.array(self.fill_hand())
        self.dealer = np.array(self.fill_hand())
        self.policy = np.zeros((10, 10, 2))
        self.init_policy()
        self.init_player()

    def init_policy(self):
        # when sum less than 20, keep hit, do no care other things
        self.policy[:-2][:][:] = 1

    def init_player(self):
        while self.player_sum() < 12:
            self.hit(0)

    def draw(self):
        cards = np.array((1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 10, 10, 10))
        card = np.random.choice(cards, p=np.ones_like(cards) / cards.size)
        return card

    def fill_hand(self):
        return self.draw(), self.draw()

    def usable_ace(self, person):
        if person == 0:
            if 1 in self.player and sum(self.player) <= 11:
                return 1
            else:
                return 0
        if person == 1:
            if 1 in self.dealer and sum(self.dealer) <= 11:
                return 1
            else:
                return 0

    def burst(self, person):
        if person == 0:
            return sum(self.player) > 21
        elif person == 1:
            return sum(self.dealer) > 21

    def hit(self, person):
        if person == 0:
            self.player = np.append(self.player, self.draw())
        elif person == 1:
            self.dealer = np.append(self.dealer, self.draw())

    def nature(self):
        nature = self.player.size == 2 and 1 in self.player and 10 in self.player
        return nature

    def dealer_draw(self):
        while self.dealer_sum() < 17:
            self.hit(1)

    def dealer_sum(self):
        if self.usable_ace(1):
            return sum(self.dealer) + 10
        else:
            return sum(self.dealer)

    def dealer_showing(self):
        return self.dealer[0]

    def player_sum(self):
        if self.usable_ace(0):
            return sum(self.player) + 10
        else:
            return sum(self.player)

    def compare(self):
        player_sum = self.player_sum()
        dealer_sum = self.dealer_sum()
        assert player_sum <= 21
        assert dealer_sum <= 21
        if player_sum > dealer_sum:
            return 1
        elif player_sum == dealer_sum:
            return 0
        else:
            return -1

    def get_action_follow_policy(self):
        action = self.policy[self.player_sum() - 12][self.dealer_showing() - 1][self.usable_ace(0)]
        return int(action)

    def get_state(self):
        return [self.player_sum(), self.dealer_showing(), self.usable_ace(0)]

    def simulate(self):
        self.player = np.array(self.fill_hand())
        self.dealer = np.array(self.fill_hand())

        self.init_player()
        assert self.player_sum() >= 12

        episode = []
        es_and_hit = True
        episode.append(self.get_state())
        # explore start, for each state-action pair, then follow policy
        if np.random.rand() < 0.5:
            episode.append(0)
            es_and_hit = False
        else:
            episode.append(1)
            self.hit(0)
            if self.burst(0):
                episode.append(-1)
                return episode
            else:
                # do not burst
                episode.append(0)

        # alive, choose to action [keep or hit] until burst or keep
        while not self.burst(0) and es_and_hit:
            # append state
            episode.append(self.get_state())
            action = self.get_action_follow_policy()
            # append action
            episode.append(action)

            if action == 1:
                self.hit(0)
                if self.burst(0):
                    episode.append(-1)
                    return episode
                else:
                    # do not burst
                    episode.append(0)

            elif action == 0:
                # do not move, watch dealer action
                break

        self.dealer_draw()
        if self.burst(1):
            episode.append(1)
            return episode
        else:
            # dealer not burst should compare result
            episode.append(self.compare())
            return episode


# 10, 10, 2 for state, 2 for keep or hit
state_action_value = np.random.rand(10, 10, 2, 2)

# for uodate state action value
moving_average = 0.98
# for future reward
gamma = 0.9
# init env
env = BlackJackEnv()


# runs  int -> how many episode need generate
def run(runs):
    for i in range(runs):
        episode = env.simulate()
        index = len(episode) - 1
        expectation = 0
        while index > 0:
            value = episode[index]
            index -= 1
            action = episode[index]
            index -= 1
            (player_sum, dealer_show, usable_ace) = episode[index]
            index -= 1

            expectation = gamma * expectation + value

            assert 21 >= player_sum >= 12
            assert 1 <= dealer_show <= 10
            assert usable_ace == 1 or usable_ace == 0
            assert action == 0 or action == 1

            state_action_value[player_sum - 12][dealer_show - 1][usable_ace][action] = \
                moving_average * state_action_value[player_sum - 12][dealer_show - 1][usable_ace][action] + \
                (1 - moving_average) * expectation

            greedy = (1 if state_action_value[player_sum - 12][dealer_show - 1][usable_ace][0] <
                           state_action_value[player_sum - 12][dealer_show - 1][usable_ace][1] else 0)
            if state_action_value[player_sum - 12][dealer_show - 1][usable_ace][0] < \
                    state_action_value[player_sum - 12][dealer_show - 1][usable_ace][1]:
                assert greedy == 1

            env.policy[player_sum - 12][dealer_show - 1][usable_ace] = greedy


# show_policy  true -> show policy, false -> show state value
# with_usable_ace  0 -> without, 1 -> with
def show_policy_or_state_value(show_policy, with_usable_ace):
    value = np.zeros((10, 10))
    for j in range(10):
        for k in range(10):
            if show_policy:
                value[j][k] = env.policy[j][k][with_usable_ace]
            else:
                value[j][k] = max(state_action_value[j][k][with_usable_ace][0],
                                  state_action_value[j][k][with_usable_ace][1])
    ax = sns.heatmap(value, xticklabels=np.arange(10) + 1,
                     yticklabels=np.arange(10) + 12, annot=True)
    plt.xlabel("Dealer showing")
    plt.ylabel("Player sum")
    if show_policy:
        if with_usable_ace == 1:
            plt.title("Best policy with usable ace [0 mean stick while 1 mean hit]")
        else:
            plt.title("Best policy without usable ace [0 mean stick while 1 mean hit]")
    else:
        if with_usable_ace == 1:
            plt.title("State value with usable ace")
        else:
            plt.title("State value without usable ace")
    plt.show()


run(100000)
show_policy_or_state_value(False, 1)
