import tensorflow as tf
import tensorflow.contrib.slim as slim
import tensorflow_probability as tfp
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import os
from gym_unity.envs import UnityEnv
import scipy.signal
EP_MAX = 1000
GAMMA = 0.99
LR = 0.0001
BATCH = 1024
MINIBATCH = 64
EPOCHS = 10
UPDATE_STEPS = 10
Epsilon=0.2
ENTROPY_BETA = 0.01
LAMBDA = 0.95

TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")

class PPO(object):

    def __init__(self,environment,load=False,testing=False,gpu=False):
        self.testing = testing
        self.s_dim, self.a_dim = environment.observation_space.shape, environment.action_space.n
        self.model_path = "model/"
        self.buffer_s, self.buffer_a, self.buffer_r, self.buffer_v = [], [], [] , []
        self.buffer_done = []
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
                pol_entpen = -ENTROPY_BETA * tf.reduce_mean(entropy)
            loss = self.aloss + self.loss_vf + pol_entpen
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
    
    def update(self, s, a, r, adv):
        
        self.sess.run([self.update_oldpi_op, self.update_vf_old_op, self.iterator.initializer],
                      feed_dict={self.tfs: s, self.tfa: a, self.tfdc_r: r, self.tfadv: adv})

        [self.sess.run([self.summarise, self.global_step, self.atrain_op]) for _ in range(UPDATE_STEPS)]
        
    def _build_anet(self,inputs, name, trainable,reuse=False):
        with tf.variable_scope(name,reuse=reuse):
            conv1 = slim.conv2d(inputs, 32, [3,3], activation_fn=tf.nn.relu,trainable=trainable)
            conv2 = slim.conv2d(conv1, 64, [3,3], activation_fn=tf.nn.relu,trainable=trainable)
            conv3 = slim.conv2d(conv2, 64, [3,3], activation_fn=tf.nn.relu,trainable=trainable)
            state_in = tf.layers.flatten(conv3)
            l1 = slim.fully_connected(state_in, 32,activation_fn=tf.nn.relu,trainable=trainable)
            l1 = slim.fully_connected(l1, 64,activation_fn=tf.nn.relu,trainable=trainable)
            l1 = slim.fully_connected(l1, 64,activation_fn=tf.nn.relu,trainable=trainable)

            a_logits = slim.fully_connected(l1, self.a_dim)
            dist = tfp.distributions.Categorical(logits=a_logits)
        params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=name)
        return dist, params
    def _build_cnet(self,inputs, name, trainable,reuse=False):

        with tf.variable_scope(name,reuse=reuse):
            conv1 = slim.conv2d(inputs, 32, [3,3], activation_fn=tf.nn.relu,trainable=trainable)
            conv2 = slim.conv2d(conv1, 64, [3,3], activation_fn=tf.nn.relu,trainable=trainable)
            conv3 = slim.conv2d(conv2, 64, [3,3], activation_fn=tf.nn.relu,trainable=trainable)
            state_in = slim.flatten(conv3)

            l1 = slim.fully_connected(state_in, 32,activation_fn=tf.nn.relu,trainable=trainable)
            l1 = slim.fully_connected(l1, 64,activation_fn=tf.nn.relu,trainable=trainable)
            vf = slim.fully_connected(l1, 1,activation_fn=tf.nn.relu,trainable=trainable)

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

if __name__ == "__main__":
    env = UnityEnv('test.app', 0,use_visual=True)
    ppo = PPO(env)
    all_ep_r = []
    t = 0
    for ep in range(EP_MAX):
        s = env.reset()
        
        ep_r = 0
        
        done = False
        while not done: 
            t+=1
            env.render()
            a,v = ppo.choose_action(s)
            s_, r, done, _ = env.step(a)
            ppo.buffer_s.append(s)
            ppo.buffer_a.append(a)
            ppo.buffer_r.append(r)
            ppo.buffer_v.append(v)
            ppo.buffer_done.append(done)
            s = s_
            ep_r += r

            # update ppo
            if (t+1) % BATCH == 0:
                print("updating...")
                t = 0
                v_s_ = v
                discounted_r = []
                rewards = np.array(ppo.buffer_r)
                v_final = [v * (1 - done)] 
                terminals = np.array(ppo.buffer_done + [done])
                values = np.array(ppo.buffer_v + v_final)
                delta = rewards + GAMMA * values[1:] * (1 - terminals[1:]) - values[:-1]
                advantage = discount(delta, GAMMA * LAMBDA, terminals)
                returns = advantage + np.array(ppo.buffer_v)
                advantage = (advantage - advantage.mean()) / np.maximum(advantage.std(), 1e-6)

                for r in ppo.buffer_r[::-1]:
                    v_s_ = r + GAMMA * v_s_
                    discounted_r.append(v_s_)
                discounted_r.reverse()

                bs, ba, br,badv = np.reshape(ppo.buffer_s, (-1,) + ppo.s_dim), np.vstack(ppo.buffer_a), \
                                np.vstack(discounted_r), np.vstack(advantage)
                ppo.buffer_s, ppo.buffer_a, ppo.buffer_r = [], [], []
                ppo.buffer_v, ppo.buffer_done = [], []
                ppo.update(bs, ba, br,badv)
        print("episode = {}, ep_r = {}".format(ep,ep_r))
        if ep != 0 and ep % 100 == 0:
            ppo.save_model(steps=ep)

        all_ep_r.append(ep_r)


    
