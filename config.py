import numpy as np

class config:
    # config for KL-UCB
    n_klucb = 10000
    c_klucb = 0
    n_arms_klucb = 16
    chng_time_klucb = 1500

    # config for Discounted TS
    n_arms_dts = 16
    gama_dts = np.random.uniform(0.9, 0.99)
    time_dts = 10000
    chng_time_dts = 1500
    probs_dts = [(i) for i in range(n_arms_dts)]
    probs_dts = (n_arms_dts -np.array(probs_dts))/(n_arms_dts*3)

    # config for AD-Switch
    n_arms_adswitch = 16
    time_adswitch = 200
    episodes_adswitch = 50
    chng_time_adswitch = 15
    C1 = 0.1

    # config for TS-GE
    episodes = 20
    tsge_time_horizon = 10000
    n_arms_tsge = 16
    tsge_chng_time = 1500
    ts_episode_time = 500
    bp_episode_time = 150
    ge_episode_time = 100
    n_etc_tsge = 20
    delta_bp = 0.1
    
    max_reward = 100
    chng_arms = [15, 3, 9, 6, 12]
    mean = max_reward/2 - n_arms_dts/2 + np.array([(i+1) for i in range(n_arms_dts, 0, -1)])
    std = [1 for i in range(n_arms_dts)]