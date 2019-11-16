import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import os
from gym_unity.envs import UnityEnv
import scipy.signal
import threading
from collections import deque

EP_MAX = 5000
GAMMA = 0.99
LR = 0.0001
N_WORKER = 5
BATCH = 10240
MINIBATCH = 64
EPOCHS = 10
Epsilon=0.2
LAMBDA = 0.95

GameDir = 'test.app'
modelPath = "multi_thread_Model/"

TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")

class PPO(object):

    def __init__(self,environment=None,load=False,testing=False,gpu=True,ModelPath="model/"):
        self.testing = testing
        self.s_dim, self.a_dim = (84, 84, 3), 3
        self.model_path = ModelPath
        # self.buffer_s, self.buffer_a, self.buffer_r, self.buffer_v = [], [], [] , []
        # self.buffer_done = []
        config = tf.ConfigProto(log_device_placement=False, device_count={'GPU': gpu})
        config.gpu_options.per_process_gpu_memory_fraction = 0.1
        self.sess = tf.Session(config=config)
        self.tfs = tf.placeholder(tf.float32, [None] +  list(self.s_dim), 'state')
        self.tfdc_r = tf.placeholder(tf.float32, [None, 1], 'discounted_r')
        self.tfa = tf.placeholder(tf.int32, [None, 1], 'action')
        self.tfadv = tf.placeholder(tf.float32, [None, 1], 'advantage')
        

        self.dataset = tf.data.Dataset.from_tensor_slices({"state": self.tfs, "actions": self.tfa,
                                                           "rewards": self.tfdc_r, "advantage": self.tfadv})
        self.dataset = self.dataset.shuffle(buffer_size=10000)
        self.dataset = self.dataset.batch(MINIBATCH)
        self.dataset = self.dataset.cache()
        self.dataset = self.dataset.repeat(EPOCHS)
        self.iterator = self.dataset.make_initializable_iterator()
        self.batch = self.iterator.get_next()

        # actor
        pi, pi_params = self._build_anet(self.batch["state"],'pi', trainable=True)
        oldpi, oldpi_params = self._build_anet(self.batch["state"],'oldpi', trainable=False)
        pi_eval, _ = self._build_anet(self.tfs, 'pi', trainable=False,reuse=True)

        
        # critic
        self.v, vf_params = self._build_cnet(self.batch["state"], "vf",trainable=True)
        vf_old, vf_old_params = self._build_cnet(self.batch["state"], "oldvf",trainable=False)
        self.vf_eval, _ = self._build_cnet(self.tfs, 'vf', trainable=False,reuse=True)
        
        with tf.variable_scope('sample_action'):
            self.sample_op = tf.squeeze(pi_eval.sample(1), axis=0)
        self.eval_action = pi_eval.mode()

        # self.advantage = self.tfdc_r - self.v
        # self.closs = tf.reduce_mean(tf.square(self.advantage))
        # self.ctrain_op = tf.train.AdamOptimizer(LR).minimize(self.closs)

        self.global_step = tf.train.get_or_create_global_step()
        
        

        with tf.variable_scope('loss'):
            epsilon_decay = tf.train.polynomial_decay(Epsilon, self.global_step, 1e5, 0.01, power=0.0)
            with tf.variable_scope('surrogate'):
                ratio = tf.maximum(pi.prob(self.batch["actions"]),1e-6) / tf.maximum(oldpi.prob(self.batch["actions"]),1e-6)
                ratio = tf.clip_by_value(ratio, 0, 10)
                surr1 = ratio * self.batch["advantage"]
                surr2 = tf.clip_by_value(ratio, 1.-epsilon_decay, 1.+epsilon_decay)* self.batch["advantage"]
                self.aloss = -tf.reduce_mean(tf.minimum(surr1,surr2))
                tf.summary.scalar("loss", self.aloss)
            with tf.variable_scope('value_function'):
                clipped_value_estimate = vf_old + tf.clip_by_value(self.v - vf_old, -epsilon_decay, epsilon_decay)
                loss_vf1 = tf.squared_difference(clipped_value_estimate, self.batch["rewards"])
                loss_vf2 = tf.squared_difference(self.v, self.batch["rewards"])
                self.loss_vf = tf.reduce_mean(tf.maximum(loss_vf1, loss_vf2)) * 0.5
                tf.summary.scalar("loss", self.loss_vf)
            with tf.variable_scope('entropy'):
                entropy = pi.entropy()
                entropen = -0.01 * tf.reduce_mean(entropy)
            loss = self.aloss + self.loss_vf + entropen
            tf.summary.scalar("total", loss)
            
        tf.summary.scalar("value", tf.reduce_mean(self.v))
        tf.summary.scalar("policy_entropy", tf.reduce_mean(entropy))

        with tf.variable_scope('atrain'):
            self.atrain_op = tf.train.AdamOptimizer(LR).minimize(loss, 
                                global_step=self.global_step, var_list=pi_params + vf_params)

        with tf.variable_scope('update_old'):
            self.update_oldpi_op = [oldp.assign(p) for p, oldp in zip(pi_params, oldpi_params)]
            self.update_vf_old_op = [oldp.assign(p) for p, oldp in zip(vf_params, vf_old_params)]

        tf.summary.FileWriter("log/", self.sess.graph)
        
        self.saver = tf.train.Saver()
        if load:
            self.load_model()
        else:
            self.initialize_model()
        self.summarise = tf.summary.merge(tf.get_collection(tf.GraphKeys.SUMMARIES))

    def initialize_model(self):
        init = tf.global_variables_initializer()
        self.sess.run(init)

    def load_model(self):
        ckpt = tf.train.get_checkpoint_state(self.model_path)
        print("Loading {}".format(ckpt))
        if ckpt is None:
            print("ckpt not found")
        self.saver.restore(self.sess, ckpt.model_checkpoint_path)
    def save_model(self, steps=None):
        last_checkpoint = self.model_path + "model_"+ TIMESTAMP + "_" + str(steps) + ".cptk"
        
        self.saver.save(self.sess, last_checkpoint)
    
    def update(self):
        global GLOBAL_COUNTER, GLOBAL_DATA
        while not COORD.should_stop():
            if GLOBAL_EP < EP_MAX:
                UPDATE_EVENT.wait()
                print("updating....")
                
                s = GLOBAL_DATA["state"]
                a = GLOBAL_DATA["action"]
                r = GLOBAL_DATA["reward"]
                adv = GLOBAL_DATA["advantage"]
            
                self.sess.run([self.update_oldpi_op, self.update_vf_old_op, self.iterator.initializer],
                            feed_dict={self.tfs: s, self.tfa: a \
                                , self.tfdc_r: r, self.tfadv: adv})
                while True:
                    try:
                        self.sess.run([self.summarise, self.global_step, self.atrain_op])
                    except:
                        break
                GLOBAL_DATA = {"state":[],"action":[],"reward":[],"advantage":[]}
                UPDATE_EVENT.clear()        
                GLOBAL_COUNTER = 0 
                COLLECT_EVENT.set()
        
    def _build_anet(self,inputs, name, trainable,reuse=False):
        w_reg = tf.contrib.layers.l2_regularizer(0.001)
        with tf.variable_scope(name,reuse=reuse):
            conv1 = tf.layers.conv2d(inputs=inputs, filters=32, kernel_size=8, strides=4, activation=tf.nn.relu)
            conv2 = tf.layers.conv2d(inputs=conv1, filters=64, kernel_size=4, strides=2, activation=tf.nn.relu)
            conv3 = tf.layers.conv2d(inputs=conv2, filters=64, kernel_size=3, strides=1, activation=tf.nn.relu)
            state_in = tf.layers.flatten(conv3)
            l1 = tf.layers.dense(state_in, 32, tf.nn.relu, kernel_regularizer=w_reg)
            l1 = tf.layers.dense(l1, 64, tf.nn.relu, kernel_regularizer=w_reg)
            l1 = tf.layers.dense(l1, 64, tf.nn.relu, kernel_regularizer=w_reg)

            a_logits = tf.layers.dense(l1, self.a_dim, kernel_regularizer=w_reg)
            dist = tf.distributions.Categorical(logits=a_logits)
        params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=name)
        return dist, params
    def _build_cnet(self,inputs, name, trainable,reuse=False):
        w_reg = tf.contrib.layers.l2_regularizer(0.001)
        with tf.variable_scope(name,reuse=reuse):
            conv1 = tf.layers.conv2d(inputs=inputs, filters=32, kernel_size=8, strides=4, activation=tf.nn.relu)
            conv2 = tf.layers.conv2d(inputs=conv1, filters=64, kernel_size=4, strides=2, activation=tf.nn.relu)
            conv3 = tf.layers.conv2d(inputs=conv2, filters=64, kernel_size=3, strides=1, activation=tf.nn.relu)
            state_in = tf.layers.flatten(conv3)

            l1 = tf.layers.dense(state_in, 32, tf.nn.relu, kernel_regularizer=w_reg)
            l1 = tf.layers.dense(l1, 64, tf.nn.relu, kernel_regularizer=w_reg)
            vf = tf.layers.dense(l1, 1, tf.nn.relu, kernel_regularizer=w_reg)

        params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=name)
        return vf, params
    def choose_action(self, s):
        s = s[np.newaxis, :]
        if not self.testing:
            a, v = self.sess.run([self.sample_op, self.vf_eval], {self.tfs: s})
        else:
            a, v = self.sess.run([self.eval_action, self.vf_eval], {self.tfs: s})

        return a[0], np.squeeze(v)
def discount(x, gamma, terminal_array=None):
    if terminal_array is None:
        return scipy.signal.lfilter([1], [1, -gamma], x[::-1], axis=0)[::-1]
    else:
        y, adv = 0, []
        terminals_reversed = terminal_array[1:][::-1]
        for step, dt in enumerate(reversed(x)):
            y = dt + gamma * y * (1 - terminals_reversed[step])
            adv.append(y)
        return np.array(adv)[::-1]


class Worker(object):
    def __init__(self, wid):
        self.wid = wid
        self.lock = threading.Lock()
        self.env = UnityEnv(GameDir, wid,use_visual=True)

        self.ppo = GLOBAL_PPO

    def work(self):
        global GLOBAL_EP, GLOBAL_COUNTER
        t = 0
        while not COORD.should_stop():
            s = self.env.reset()
            ep_r = 0
            buffer_s, buffer_a, buffer_r, buffer_v ,buffer_done = [], [], [], [], []
            done = False
            
            while not done:
                if not COLLECT_EVENT.is_set():                  
                    COLLECT_EVENT.wait()                        
                    buffer_s, buffer_a, buffer_r, buffer_v ,buffer_done = [], [], [], [], []
                a,v = self.ppo.choose_action(s)
                s_, r, done, _ = self.env.step(a)
                buffer_s.append(s)
                buffer_a.append(a)
                buffer_r.append(r)
                buffer_v.append(v)
                buffer_done.append(done)
                s = s_
                ep_r += r
                t+=1
                GLOBAL_COUNTER += 1
                # update ppo
                if (done or GLOBAL_COUNTER >= BATCH):
                    
                    t = 0
                    rewards = np.array(buffer_r)
                    v_final = [v * (1 - done)] 
                    terminals = np.array(buffer_done + [done])
                    values = np.array(buffer_v + v_final)
                    delta = rewards + GAMMA * values[1:] * (1 - terminals[1:]) - values[:-1]
                    advantage = discount(delta, GAMMA * LAMBDA, terminals)
                    returns = advantage + np.array(buffer_v)
                    advantage = (advantage - advantage.mean()) / np.maximum(advantage.std(), 1e-6)


                    bs, ba, br,badv = np.reshape(buffer_s, (-1,) + self.ppo.s_dim), np.vstack(buffer_a), \
                                    np.vstack(returns), np.vstack(advantage)
                    buffer_s, buffer_a, buffer_r = [], [], []
                    buffer_v, buffer_done = [], []
                    COLLECT_EVENT.wait()
                    self.lock.acquire()
                    for i in range(len(bs)):
                        GLOBAL_DATA["state"].append(bs[i])
                        GLOBAL_DATA["reward"].append(br[i])
                        GLOBAL_DATA["action"].append(ba[i])
                        GLOBAL_DATA["advantage"].append(badv[i])
                    self.lock.release()
                    if GLOBAL_COUNTER >= BATCH and len(GLOBAL_DATA["state"])>= BATCH:
                        COLLECT_EVENT.clear()
                        UPDATE_EVENT.set() 
                    # self.ppo.update(bs, ba, br,badv)

                if GLOBAL_EP >= EP_MAX:
                    self.env.close()
                    COORD.request_stop()
                    break
            print("episode = {}, ep_r = {}, wid = {}".format(GLOBAL_EP,ep_r,self.wid))
            GLOBAL_EP += 1
            if GLOBAL_EP != 0 and GLOBAL_EP % 500 == 0:
                self.ppo.save_model(steps=GLOBAL_EP)

if __name__ == '__main__':
    tmpenv = UnityEnv(GameDir, 0,use_visual=True).unwrapped
    GLOBAL_PPO = PPO(tmpenv,ModelPath=modelPath)
    tmpenv.close()
    GLOBAL_DATA = {"state":[],"action":[],"reward":[],"advantage":[]}
    UPDATE_EVENT, COLLECT_EVENT = threading.Event(), threading.Event()
    UPDATE_EVENT.clear()
    COLLECT_EVENT.set()
    workers = [Worker(wid=i) for i in range(1,N_WORKER+1)]
    
    GLOBAL_COUNTER, GLOBAL_EP = 0, 0
    COORD = tf.train.Coordinator()
    
    threads = []
    for worker in workers:
        t = threading.Thread(target=worker.work, args=())
        t.start()
        threads.append(t)
    threads.append(threading.Thread(target=GLOBAL_PPO.update,))
    threads[-1].start()
    COORD.join(threads)
    
