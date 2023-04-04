import numpy as np
from scipy import optimize
from config import config
from scipy.optimize import fsolve
from matplotlib import pyplot as plt

# create an environment for the agent to interact with
class env(object):
    def __init__(self, arms, config):
        self.n = config.n_klucb
        self.probs = [(i+1)/arms for i in range(arms)]
        self.k_arms = arms
    
    def pull(self, arm):
        if arm not in range(self.k_arms):
            raise ValueError(f"arm {arm} must be in range of k_arms")
        return np.random.normal(self.mean[arm], self.std[arm])

class KL_UCB:
    def __init__(self, config):
        self.config = config
        self.n_arms = config.n_arms_klucb
        self.c = config.c_klucb
        self.n = config.n_klucb # horizon
        self.rewards_per_arm = [[] for _ in range(self.n_arms)]
        self.binom_rewards = [[] for _ in range(self.n_arms)]
        self.probs = [(self.n_arms-i)/(self.n_arms*2) for i in range(self.n_arms)]
        self.n_pulls = [0 for _ in range(self.n_arms)]
        self.mean = config.mean
        self.std = config.std
        self.chng_arms = config.chng_arms
        self.regret = []
        self.time = 0
        self.eps = 1e-15
        self.max_reward = config.max_reward
        self.chng_time = config.chng_time_klucb

    def function(self, q, rewards_per_arm, t, n_pulls):
        rhs = np.log(t) + self.c * np.log(np.log(t))
        reward = np.sum(rewards_per_arm) / n_pulls
        # if q>1 or q<0: np.clip(q, 0, 1)
        kl = q * np.log(q / reward) + (1 - q) * np.log((1 - q) / (1 - reward))
        lhs = n_pulls * kl
        return lhs - rhs
    
    def get_argmax(self, t):
        q = [-1 for i in range(self.n_arms)]
        for i in range(self.n_arms):
            f = lambda q: self.function(q, self.binom_rewards[i], t, self.n_pulls[i])
            q_opt = fsolve(f, 0.75)
            q[i] = q_opt
        return np.argmax(q)

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

    def act(self, change=False):
        for i in range(0, min(self.n, self.n_arms)):
            reward = np.clip(np.random.normal(self.mean[i], self.std[i]), 0, self.max_reward)
            self.rewards_per_arm[i].append(reward)
            self.binom_rewards[i].append(np.random.binomial(1, self.probs[i]))
            self.time += 1
            self.n_pulls[i] += 1
        
        if self.n_arms < self.n:
            for t in range(self.n_arms, self.n):
                arm = self.get_argmax(t)
                reward = np.clip(np.random.normal(self.mean[arm], self.std[arm]), 0, self.max_reward)
                binomial = np.random.binomial(1, self.probs[arm])
                self.rewards_per_arm[arm].append(reward)
                self.binom_rewards[arm].append(binomial)
                self.n_pulls[arm] += 1
                if change: self.update_arm()
                self.regret.append(max(self.mean)-self.mean[arm])

        for i in range(self.n_arms):
            self.rewards_per_arm[i] = np.mean(self.rewards_per_arm[i])
        return {'rewards': self.rewards_per_arm, 'chng_arms': self.chng_arms, 'regret': self.regret}
