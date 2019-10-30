from ppo import PPO
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from gym_unity.envs import UnityEnv


MAXEP = 1000
if __name__ == "__main__":
    env = UnityEnv('test.app', 0,use_visual=True)
    ppo = PPO(env,load=True)
    for ep in range(MAXEP):
        s = env.reset()
        ep_r = 0
        done = False
        while not done:
            env.render()
            a,v = ppo.choose_action(s)
            s_, r, done, _ = env.step(a)
            ep_r += r
        print("episode = {}, ep_r = {}".format(ep,ep_r))
