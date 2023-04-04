import numpy as np
from matplotlib import pyplot as plt

# implement the ADSwitch algorithm to identify the best arm

class ADSwitch:
    def __init__(self, config):
        self.config = config
        self.n_arms = config.n_arms_adswitch
        self.T = config.time_adswitch
        self.episodes = config.episodes_adswitch
        self.reward = [[] for i in range(self.n_arms)]
        self.selected = [i for i in range(self.n_arms)]
        self.time = 0
        self.chng_arms = config.chng_arms
        self.chng_time = config.chng_time_adswitch
        self.max_reward = config.max_reward
        self.mean = config.mean
        self.std = config.std
        self.C = config.C1
        self.set_arms = {i:[] for i in range(self.n_arms)}
        self.regret = []

    def update_arm(self):
        if len(self.chng_arms) != 0:
            arm = self.chng_arms.pop(0)
            if (self.time % self.chng_time == 0):
                self.mean[arm] = max(self.mean)+8

    def get_reward(self, arm):
        # get reward for the arm
        reward = np.clip(np.random.normal(self.mean[arm], self.std[arm]), 0, self.max_reward)
        self.reward[arm].append([self.time, reward])
        return reward
    
    def check_bad(self, bad, eps_delta, l):
        for a in bad:
            i=1
            while (2**-i>(eps_delta[a]/16)):
                n1 = np.sqrt(l/(self.n_arms*self.T*np.log(self.T)))*(2**-i)
                n2 = np.random.random()
                if n1 > n2:
                    self.set_arms[a].append(((2**-i),np.ceil((2**(2*i+1))*np.log(self.T)),self.time))
                i+=1

    def calc_delta(self, good, bad, reward):
        # calculate mean of reward
        mean = [0 for i in range(self.n_arms)]
        for i in range(self.n_arms):
            if len(reward[i])>0:
                mean[i] = np.mean([r[1] for r in reward[i]])
        ret = 0
        for i in good:
            # print(mean[i])
            # print(mean[bad])
            # print(ret)
            ret = max(ret, mean[i]-mean[bad])
        return ret

    def select_arm(self, good):
        arm = np.argmin(self.selected)
        if arm==0:
            if arm in good or len(self.set_arms[arm])>0:
                self.selected[arm] = self.time
                return self.get_reward(arm), arm
        temp = 0
        for i in range(1, self.n_arms):
            if (i in good or len(self.set_arms[i])>0):
                if self.selected[i] < self.selected[temp]:
                    temp = i
        self.selected[temp] = self.time
        return self.get_reward(temp), temp

    def condition1(self, arm, start, reward, good):
        arm2 = min(good)
        for s in range(start, self.time+1):
            # mean of arm between time start and s
            u1, n1 = 0, 0
            for r in reward[arm]:
                if r[0]>=start and r[0]<=s:
                    u1 += r[1]
                    n1 += 1
            # mean of arm2 between time start and s
            u2, n2 = 0, 0
            for r in reward[arm2]:
                if r[0]>=start and r[0]<=s:
                    u2 += r[1]
                    n2 += 1
            if n1>0 and n2>0:
                u1 = u1/n1    
                u2 = u2/n2
                if n1>1:
                    if abs(u1-u2)>(np.sqrt(self.C*np.log(self.T)/(n1-1))):
                        return True
        return False

    def condition3(self, s1, s2, s, reward):
        # mean of reward between time s1 and s2
        u1, n1 = 0, 0
        for r in reward:
            if r[0]>=s1 and r[0]<=s2:
                u1 += r[1]
                n1 += 1
        # mean of reward between time s and self.time
        u2, n2 = 0, 0
        for r in reward:
            if r[0]>=s and r[0]<=self.time:
                u2 += r[1]
                n2 += 1
        
        if n1>0 and n2>0:
            u1 = u1/n1
            u2 = u2/n2
            if abs(u1-u2)>(np.sqrt(2*np.log(self.T)/n1) + np.sqrt(2*np.log(self.T)/n2)):
                return True
        return False
    
    def condition4(self, s, reward, arm, eps_mean, eps_delta):
        # mean of reward between time s and self.time
        u, n = 0, 0
        for r in reward:
            if r[0]>=s and r[0]<=self.time:
                u += r[1]
                n += 1
        if n>0:
            u = u/n
            if abs(u-eps_mean[arm])>(np.sqrt(2*np.log(self.T)/n))+eps_delta[arm]/4:
                return True
        return False

    
    def change_in_good(self, good, start_time, reward):
        # check for changes in good arms
        check = False
        for arm in good:
            for s1 in range(start_time, self.time):
                for s2 in range(s1, self.time):
                    for s in range(start_time, self.time):
                        if self.condition3(s1, s2, s, reward[arm]):
                            check = True
                            break
        return check

    def change_in_bad(self, bad, start_time, reward, esp_mean, eps_delta):
        # check for changes in bad arms
        check = False
        for arm in bad:
            for s in range(start_time, self.time):
                if self.condition4(s, reward[arm], arm, esp_mean, eps_delta):
                    check = True
                    break
        return check

    def update_set(self, bad):
        for a in bad:
            ret = []
            for elem in self.set_arms[a]:
                s, n = elem[2], 0
                for rew in self.reward[a]:
                    if rew[0]>=s and rew[0]<=self.time:n+=1
                if n<elem[1]:
                    ret.append(elem)
            self.set_arms[a] = ret

    def act(self):
        # initialize the variables
        l = 0
        # start new episode
        while l < self.episodes and self.time < self.T:
            print('episode:', l)
            l+=1
            tl = self.time+1

            esp_mean = [0 for i in range(self.n_arms)]
            eps_delta = [0 for i in range(self.n_arms)]
            esp_reward = [[] for i in range(self.n_arms)]
            good = [i for i in range(self.n_arms)]
            bad = []

            # next time step
            episode = True
            while episode and self.time < self.T:    
                print(good)
                self.time += 1
                self.update_arm()
                print('time:', self.time)
                # add checks for bad arms
                self.check_bad(bad, eps_delta, l)

                # select arm
                reward, arm = self.select_arm(good)
                # print(arm)
                self.regret.append(max(self.mean)-self.mean[arm])
                esp_reward[arm].append([self.time, reward])
                # print(esp_reward)

                # check for changes of good arm
                if not self.change_in_good(good, tl, esp_reward):
                    # print('good arm')
                    # check for changes of bad arm
                    if not self.change_in_bad(bad, tl, esp_reward, esp_mean, eps_delta):
                        # print('bad arm')
                        #update set
                        self.update_set(bad)
                        # evict arms from good
                        for a in good:
                            if self.condition1(a, tl, esp_reward, good):
                                good.remove(a)
                                bad.append(a)
                                eps_delta[a] = self.calc_delta(good, a, esp_reward)
                                esp_mean[a] = np.mean([r[1] for r in esp_reward[a]])
                                self.set_arms[a] = []
                                # print(self.set_arms)
                                # break after every 100 time steps
                        # if self.time%100==0:
                            # self.plot_regret()
                        #     break
                    else: episode = False
                else: episode = False
        self.plot_regret()
        return {'regret': self.regret}

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
