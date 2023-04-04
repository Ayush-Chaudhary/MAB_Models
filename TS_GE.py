import numpy as np
from config import config
from matplotlib import pyplot as plt

class TS_GE:
    def __init__(self, config):
        self.episodes = config.episodes
        self.T = config.tsge_time_horizon
        self.time = 0
        self.chng_time = config.tsge_chng_time
        self.chng_arms = config.chng_arms
        self.time_ts = config.ts_episode_time
        self.time_bp = config.bp_episode_time
        self.time_ge = config.ge_episode_time
        self.n_arms = config.n_arms_tsge
        self.n_etc = config.n_etc_tsge
        self.mean = config.mean
        self.std = config.std
        self.alpha = np.ones(self.n_arms)
        self.beta = np.ones(self.n_arms)
        self.delta_bp = config.delta_bp
        self.max_reward = config.max_reward
        self.arm_rewards = [[] for i in range(self.n_arms)]
        self.regret = []

    def update_arm(self):
        self.time += 1
        if len(self.chng_arms) != 0:
            arm = self.chng_arms.pop(0)
            if (self.time % self.chng_time == 0):
                self.mean[arm] = max(self.mean)+2

    def get_reward(self, arms):
        if type(arms) == list:
            new_mean = np.mean([self.mean[i] for i in arms])
            new_std = np.sqrt(np.mean([np.square(self.std[i]) for i in arms]))
        else:
            new_mean = self.mean[arms]
            new_std = self.std[arms]
        return np.clip(np.random.normal(new_mean, new_std), 0, self.max_reward)

    def onehot(self, i, d):
        one_hot = np.zeros(d)
        one_hot[i] = 1
        one_hot = ''.join(str(int(i)) for i in one_hot)
        return one_hot

    def super_arms(self):
        d = int(np.sqrt(self.n_arms))
        # zero = np.zeros(d)
        # zero = ''.join(str(int(i)) for i in zero)
        super_arms = [[] for i in range(int(d))]
        for i in range(self.n_arms):
            for j in range(d):
                one_hot = self.onehot(j, d)
                common = int(np.binary_repr(i),2) & int(one_hot,2)
                if (common != 0):
                    super_arms[d-j-1].append(i)
        # print(super_arms)
        return super_arms

    def ThompsonSampling(self, change):
        for j in range(self.time_ts):
            theta = [np.random.beta(self.alpha[k], self.beta[k]) for k in range(self.n_arms)]
            arm = np.argmax(theta)
            reward = self.get_reward(arm)
            self.arm_rewards[arm].append(reward)
            reward_norm = reward/self.max_reward
            reward2 = np.random.binomial(1, reward_norm)
            self.alpha[arm] += reward2
            self.beta[arm] += 1-reward2
            if change:self.update_arm()
            self.regret.append(max(self.mean)-reward)
        rewards = [self.get_reward(i) for i in range(self.n_arms)]
        return np.mean(rewards)

    def BroadcastProbing(self):
        rewards = 0
        arms = [i for i in range(self.n_arms)]
        for i in range(self.time_bp):
            reward = self.get_reward(arms)
            rewards+=reward
            # self.update_arm()
        return rewards/self.time_bp

    def GrpExploration(self):
        super_arms = self.super_arms()

        rewards = [self.get_reward(super_arms[i]) for i in range(len(super_arms))]

        rewards2 = [[] for i in range(len(super_arms))]
        for i in range(self.time_ge):
            for j in range(len(super_arms)):
                for k in range(len(super_arms[j])):
                    rewards2[j].append(self.get_reward(super_arms[j][k]))
            # self.update_arm()

        zeros = []
        for i in range(len(rewards2)):
            if abs(np.mean(rewards2[i])-rewards[i]) <= self.delta_bp:
                zeros.append(0)
            else:
                zeros.append(1)
        zeros = ''.join(str(int(i)) for i in zeros)
        #convert to decimal
        return int(zeros, 2)

    def act(self, change = False):
        rewards = [0 for i in range(self.n_arms)]
        for i in range(self.n_etc):
            for k in range(self.n_arms):
                rewards[k]+=(self.get_reward(k))
        for eps in range(self.episodes):
            rewards_ts = self.ThompsonSampling(change)
            rewards_bp = self.BroadcastProbing()
            if abs(rewards_ts-rewards_bp) > 4*self.delta_bp:
                arm = self.GrpExploration()
        result = [np.mean(self.arm_rewards[i]) for i in range(self.n_arms)]
        return {'rewards': result, 'chng_arms': self.chng_arms, 'regret': self.regret}
    
    def plot_regret(self):
        # find cummulitive regret
        cum_regret = np.cumsum(self.regret)
        cum_regret = [cum_regret[i]/(i+1) for i in range(len(cum_regret))]
        # plot cummulitive regret
        plt.plot(cum_regret)
        plt.xlabel('Time')
        plt.ylabel('Cummulitive Regret')
        plt.title('Cummulitive Regret vs Time')
        plt.show()