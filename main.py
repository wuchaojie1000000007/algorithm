import numpy as np
import matplotlib.pyplot as plt


class TestBed:
    # underlying value
    mean = [0.2, -0.4, 1.5, 0.3, 1.2, -1.5, -0.1, -1, 0.8, -0.4]
    # store action value, update frequently, init 0.0
    action_value = [0.0] * 10
    # store how many time action be chosen
    action_choose_time = [0] * 10
    # accumulate reward, for compare model
    total_reward = 0
    # optimal action count (when action = 2)
    optimal_action = np.zeros(1000)
    optimal_cnt = 0

    def __init__(self, epsilon=0.0):
        self.epsilon = epsilon

    def increase_choose_time(self, action):
        self.action_choose_time[action] += 1

    # action in [0, 9]
    def compute_reward(self, action):
        return np.random.randn() + self.mean[action]

    # compute all reward got through choose action "action" till now
    def all_reward(self, action):
        return self.action_value[action] * self.action_choose_time[action]

    # return action's chosen time plus one
    def step_plus_one(self, action):
        return self.action_choose_time[action] + 1

    # update action value after one step
    def update_action_value(self, action, value):
        self.action_value[action] = (self.all_reward(action) + value) / self.step_plus_one(action)

    # update total reward
    def update_reward(self, reward):
        self.total_reward += reward

    # update optimal action percentage and count
    def update_optimal_action_count(self, action, time_step):
        if action == 2:
            self.optimal_cnt += 1
        self.optimal_action[time_step] = self.optimal_cnt / (time_step + 1)

    def step(self, time):
        # find action which lead to max value
        max_value = max(self.action_value)
        max_value_action = self.action_value.index(max_value)

        # probability epsilon random choose action
        action = max_value_action
        if np.random.random() < self.epsilon:
            # [low, high)
            action = np.random.randint(0, 10)

        # do action, get reward
        self.increase_choose_time(action)
        self.update_optimal_action_count(action, time)
        reward = self.compute_reward(action)

        # update total reward
        self.update_reward(reward)
        # update the reward function
        self.update_action_value(action, reward)

        return self.total_reward

    def run(self):
        return [self.step(x) / (x + 1) for x in range(1000)]


# init array
optimal_0 = np.zeros(1000)
optimal_01 = np.zeros(1000)
optimal_001 = np.zeros(1000)
reward_0 = np.zeros(1000)
reward_01 = np.zeros(1000)
reward_001 = np.zeros(1000)

# perform 2000 runs, each run contains 1000 step.
run_time = 2000

for i in range(run_time):
    epsilon_0 = TestBed()
    epsilon_01 = TestBed(0.1)
    epsilon_001 = TestBed(0.01)

    reward_0 += epsilon_0.run()
    optimal_0 += epsilon_0.optimal_action
    reward_01 += epsilon_01.run()
    optimal_01 += epsilon_01.optimal_action
    reward_001 += epsilon_001.run()
    optimal_001 += epsilon_001.optimal_action

step = [x + 1 for x in range(1000)]


def show_average_reward():
    plt.xkcd()

    plt.plot(step, reward_0 / run_time, label="greedy")
    plt.plot(step, reward_01 / run_time, label="epsilon = 0.1")
    plt.plot(step, reward_001 / run_time, label="epsilon = 0.01")

    plt.xlabel("time step")
    plt.ylabel("Average reward")
    plt.title(f"10-armed bandit, average on {run_time} runs")

    plt.legend()

    plt.tight_layout()
    # plt.savefig("reward.png")
    plt.show()


def show_optimal_action():
    # plt.xkcd()
    plt.style.use("ggplot")

    plt.plot(step, optimal_0 / run_time, label="greedy")
    plt.plot(step, optimal_01 / run_time, label="epsilon = 0.1")
    plt.plot(step, optimal_001 / run_time, label="epsilon = 0.01")

    plt.xlabel("time step")
    plt.ylabel("optimal action percentage")
    plt.title(f"10-armed bandit, average on {run_time} runs")

    plt.legend()

    plt.tight_layout()
    plt.show()


# show_average_reward()
show_optimal_action()
