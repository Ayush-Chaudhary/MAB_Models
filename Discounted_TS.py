import numpy as np
import matplotlib.pyplot as plt

# class env(object):
#     def __init__(self, arms, config):
#         self.n = config.n_klucb
#         self.probs = [(i+1)/arms for i in range(arms)]
#         self.k_arms = arms
    
#     def pull(self, arm):
#         if arm not in range(self.k_arms):
#             raise ValueError(f"arm {arm} must be in range of k_arms")
#         return np.random.binomial(self.n, self.probs[arm])

class DiscTS:
    def __init__(self, config):
        self.config = config
        self.probs = config.probs_dts
        self.n_arms = config.n_arms_dts
        self.gama = config.gama_dts
        self.T = config.time_dts
        self.time = 0
        self.chng_time = config.chng_time_dts
        self.chng_arms = config.chng_arms
        self.alpha = np.ones(self.n_arms)
        self.beta = np.ones(self.n_arms)
        self.S = np.zeros(self.n_arms)
        self.F = np.zeros(self.n_arms)
        self.rewards = [[] for i in range(self.n_arms)]
        self.mean = config.mean
        self.std = config.std
        self.max_reward = config.max_reward
        self.regret = []

    def update_arm(self):
        self.time += 1
        if len(self.chng_arms) != 0:
            arm = self.chng_arms.pop(0)
            if (self.time % self.chng_time == 0):
                self.mean[arm] = max(self.mean)+2

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

    def act(self, change = False):
        for i in range(self.T):
            self.S*=self.gama
            self.F*=self.gama
            theta = np.random.beta(self.alpha+self.S, self.beta+self.F)
            arm = np.argmax(theta)
            reward = np.random.binomial(1, self.probs[arm])
            reward1 = np.clip(np.random.normal(self.mean[arm], self.std[arm]), 0, self.max_reward)
            self.rewards[arm].append(reward1)
            self.regret.append(max(self.mean)-reward1)
            if reward == 1:
                self.S[arm] += 1
            else:
                self.F[arm] += 1  
            if change: self.update_arm()

        # get mean rewards
        self.rewards = [np.mean(i) for i in self.rewards]       
        return {'rewards': self.rewards, 'chng_arms': self.chng_arms, 'regret': self.regret} 