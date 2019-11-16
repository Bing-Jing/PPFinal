from ppo import PPO
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
multi_thread = True
if multi_thread:
    modelPath = "multi_thread_Model/"
else:
    modelPath = "single_thread_Model/"

MAXEP = 1000
if __name__ == "__main__":
    ppo = PPO(None,load=True,testing=True,ModelPath=modelPath)
    for ep in range(MAXEP):
        s = 1# first image
        ep_r = 0
        done = False
        while not done:
            a,v = ppo.choose_action(s) # a == 1 press D, a == 2 press F, a == 0 do nothing 
            s_ = 2# push the buttom and return the image
            s = s_
