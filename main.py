import numpy as np
from KL_UCB import KL_UCB
from Discounted_TS import DiscTS
from config import config
from TS_GE import TS_GE
from ADSwitch import ADSwitch
import matplotlib.pyplot as plt

def plot_regret(regret_klucb, regret_disc_ts, regret_ts_ge):
        # find cummulitive regret
        klucb_no_change, klucb_change = regret_klucb['no change'], regret_klucb['change']
        disc_ts_no_change, disc_ts_change = regret_disc_ts['no change'], regret_disc_ts['change']
        ts_ge_no_change, ts_ge_change = regret_ts_ge['no change'], regret_ts_ge['change']
        
        cum_regret_klucb1 = np.cumsum(klucb_no_change)
        cum_regret_klucb1 = [cum_regret_klucb1[i]/(i+1) for i in range(len(cum_regret_klucb1))]

        cum_regret_klucb2 = np.cumsum(klucb_change)
        cum_regret_klucb2 = [cum_regret_klucb2[i]/(i+1) for i in range(len(cum_regret_klucb2))]

        cum_regret_disc_ts1 = np.cumsum(disc_ts_no_change)
        cum_regret_disc_ts1 = [cum_regret_disc_ts1[i]/(i+1) for i in range(len(cum_regret_disc_ts1))]

        cum_regret_disc_ts2 = np.cumsum(disc_ts_change)
        cum_regret_disc_ts2 = [cum_regret_disc_ts2[i]/(i+1) for i in range(len(cum_regret_disc_ts2))]

        cum_regret_ts_ge1 = np.cumsum(ts_ge_no_change)
        cum_regret_ts_ge1 = [cum_regret_ts_ge1[i]/(i+1) for i in range(len(cum_regret_ts_ge1))]

        cum_regret_ts_ge2 = np.cumsum(ts_ge_change)
        cum_regret_ts_ge2 = [cum_regret_ts_ge2[i]/(i+1) for i in range(len(cum_regret_ts_ge2))]

        # plot cummulitive regret
        plt.plot(cum_regret_klucb1, label='KL-UCB no change')
        plt.plot(cum_regret_klucb2, label='KL-UCB change')
        plt.plot(cum_regret_disc_ts1, label='Discounted TS no change')
        plt.plot(cum_regret_disc_ts2, label='Discounted TS change')
        plt.plot(cum_regret_ts_ge1, label='TS-GE no change')
        plt.plot(cum_regret_ts_ge2, label='TS-GE change')
        plt.xlabel('Time')
        plt.ylabel('Cummulitive Regret')
        plt.title('Cummulitive Regret vs Time')
        plt.legend()
        plt.show()

if __name__ == "__main__":
    config1 = config()

    # klucb1 = KL_UCB(config1)
    # rewards1_klucb = klucb1.act()
    # klucb2 = KL_UCB(config1)
    # rewards2_klucb = klucb2.act(True)
    # print('kl_ucb rewards no change:',rewards1_klucb['rewards'])
    # print('kl_ucb rewards change:',rewards2_klucb['rewards'])
    # regret_klucb = {'no change': rewards1_klucb['regret'], 'change': rewards2_klucb['regret']}
    # # klucb.plot_regret()

    # disc_ts1 = DiscTS(config1)
    # rewards1_disc_ts = disc_ts1.act()
    # disc_ts2 = DiscTS(config1)
    # rewards2_disc_ts = disc_ts2.act(True)
    # print('disc_ts rewards no change:',rewards1_disc_ts['rewards'])
    # print('disc_ts rewards change:',rewards2_disc_ts['rewards'])
    # regret_disc_ts = {'no change': rewards1_disc_ts['regret'], 'change': rewards2_disc_ts['regret']}
    # # disc_ts.plot_regret()

    # ts_ge1 = TS_GE(config1)
    # rewards1_ts_ge = ts_ge1.act()
    # ts_ge2 = TS_GE(config1)
    # rewards2_ts_ge = ts_ge2.act(True)
    # print('ts_ge rewards no change:',rewards1_ts_ge['rewards'])
    # print('ts_ge rewards change:',rewards2_ts_ge['rewards'])
    # regret_ts_ge = {'no change': rewards1_ts_ge['regret'], 'change': rewards2_ts_ge['regret']}
    # ts_ge.plot_regret()

    adswitch = ADSwitch(config1)
    rewards = adswitch.act()
    # ADSwitch.plot_regret()

    # plot_regret(regret_klucb, regret_disc_ts, regret_ts_ge)