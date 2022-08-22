import numpy as np
import tensorflow as tf
import gym
import tensorflow_probability as tfp
from multiprocessing import Process, Queue, Barrier, Lock
import tensorflow.keras.losses as kls

!pip3 install box2d-py

env= gym.make("CartPole-v0")
low = env.observation_space.low
high = env.observation_space.high

class Actor(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.d1 = tf.keras.layers.Dense(128,activation='relu')
        #self.d2 = tf.keras.layers.Dense(32,activation='relu')
        self.a = tf.keras.layers.Dense(2,activation='softmax')

    def call(self, input_data):
        x = self.d1(input_data)
        #x = self.d2(x)
        a = self.a(x)
        return a
    
class Critic(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.d1 = tf.keras.layers.Dense(128,activation='relu')
        #self.d2 = tf.keras.layers.Dense(32,activation='relu')
        self.v = tf.keras.layers.Dense(1, activation = None)

    def call(self, input_data):
        x = self.d1(input_data)
        #x = self.d2(x)
        v = self.v(x)
        return v

class Agent():
    def __init__(self, gamma = 0.99):
        self.actor = Actor()
        self.critic = Critic()
        self.gamma = gamma
        # self.a_opt = tf.keras.optimizers.RMSprop(learning_rate=1e-5)
        # self.c_opt = tf.keras.optimizers.RMSprop(learning_rate=1e-5)
        self.a_opt = tf.keras.optimizers.Adam(learning_rate=7e-3)
        self.c_opt = tf.keras.optimizers.Adam(learning_rate=7e-3)
    
    def get_action(self,state):
        prob = self.actor(np.array([state]))
        prob = prob.numpy()
        dist = tfp.distributions.Categorical(probs=prob, dtype=tf.float32)
        action = dist.sample()
        return int(action.numpy()[0])

    def actor_loss(self, probs, actions, td):
        
        probability = []
        log_probability= []
        for pb,a in zip(probs,actions): 
            dist = tfp.distributions.Categorical(probs=pb, dtype=tf.float32)
            log_prob = dist.log_prob(a)
            prob = dist.prob(a)
            probability.append(prob)
            log_probability.append(log_prob)

        # print(probability)
        # print(log_probability)

        p_loss= []
        e_loss = []
        td = td.numpy()
        #print(td)
        for pb, t, lpb in zip(probability, td, log_probability):
            t =  tf.constant(t)
            policy_loss = tf.math.multiply(lpb,t)
            entropy_loss = tf.math.negative(tf.math.multiply(pb,lpb))
            p_loss.append(policy_loss)
            e_loss.append(entropy_loss)
        p_loss = tf.stack(p_loss)
        e_loss = tf.stack(e_loss)
        p_loss = tf.reduce_mean(p_loss)
        e_loss = tf.reduce_mean(e_loss)
        # print(p_loss)
        # print(e_loss)
        loss = -p_loss - 0.0001 * e_loss
        #print(loss)
        return loss

    def train_step(self, states, actions, discnt_rewards):
        discnt_rewards = tf.reshape(discnt_rewards, (len(discnt_rewards),))
        
        with tf.GradientTape() as tape1, tf.GradientTape() as tape2:
            p = self.actor(states, training=True)
            v =  self.critic(states,training=True)
            v = tf.reshape(v, (len(v),))
            td = tf.math.subtract(discnt_rewards, v)
            # print(discnt_rewards)
            # print(v)
            # print(td.numpy())
            critic_loss = 0.5 * kls.mean_squared_error(discnt_rewards, v)
            actor_loss = self.actor_loss(p, actions, td)
            
        grads1 = tape1.gradient(actor_loss,  self.actor.trainable_variables)
        grads2 = tape2.gradient(critic_loss, self.critic.trainable_variables)
        self.a_opt.apply_gradients(zip(grads1, self.actor.trainable_variables))
        self.c_opt.apply_gradients(zip(grads2, self.critic.trainable_variables))
        return actor_loss, critic_loss

def n_step_td_target(states, actions, rewards, gamma, s_queue, a_queue, r_queue, lock):
    R = 0
    discnt_rewards = []
    rewards.reverse()
    for r in rewards:
        R = r + gamma*R
        discnt_rewards.append(R)
    discnt_rewards.reverse()
    states         = np.array(states, dtype=np.float32)
    actions        = np.array(actions, dtype=np.int32)
    discnt_rewards = np.array(discnt_rewards, dtype=np.float32)
    #exp = np.array([states, actions,discnt_rewards])
    lock.acquire()
    s_queue.put(states)
    a_queue.put(actions)
    r_queue.put(discnt_rewards)
    lock.release()

def preprocess2(s_queue, a_queue, r_queue):
    states = []
    while not s_queue.empty():
        states.append(s_queue.get())

    actions = []
    while not a_queue.empty():
        actions.append(a_queue.get())
    dis_rewards = []
    while not r_queue.empty():
        dis_rewards.append(r_queue.get())

    state_batch = np.concatenate(*(states,), axis=0)  
    action_batch = np.concatenate(*(actions,), axis=None)  
    reward_batch = np.concatenate(*(dis_rewards,), axis=None)  
    # exp = np.transpose(exp)  

    return state_batch, action_batch, reward_batch

def runner(barrier, lock, s_queue, a_queue, r_queue):
    tf.random.set_seed(336699)
    agent = Agent()
    max_episodes = 500
    ep_reward = []
    total_avgr = []
    
    for episode in range(max_episodes):

        state = env.reset()
        done = False
        total_reward = 0
        all_aloss = []
        all_closs = []
        rewards = []
        states  = []
        actions = []
        while not done:
            #env.render()
            action = agent.get_action(state)
            next_state, reward, done, _ = env.step(action)
            rewards.append(reward)
            states.append(state)
            #actions.append(tf.one_hot(action, 2, dtype=tf.int32).numpy().tolist())
            actions.append(action)
            state = next_state
            total_reward += reward

            if done:
                ep_reward.append(total_reward)
                avg_reward = np.mean(ep_reward[-100:])
                total_avgr.append(avg_reward)
                print("total reward after {} steps is {} and avg reward is {}".format(episode+1, total_reward, avg_reward))
                n_step_td_target(states, actions, rewards, 1, s_queue, a_queue, r_queue, lock)
                b = barrier.wait()
                if b == 0:
                    if (s_queue.qsize() == n_workers) & (a_queue.qsize() == n_workers) & (r_queue.qsize() == n_workers):
                        print(s_queue.qsize())
                        print(a_queue.qsize())
                        print(r_queue.qsize())
                        state_batch, action_batch, reward_batch = preprocess2(s_queue, a_queue, r_queue) 
                        # print(state_batch)
                        # print(action_batch)
                        # print(reward_batch)  
                        actor_loss, critic_loss = agent.train_step(state_batch, action_batch, reward_batch) 
                        all_aloss.append(actor_loss)
                        all_closs.append(critic_loss)
                        print(f"actor_loss  : {actor_loss}") 
                        print(f"critic_loss : {critic_loss}") 

                barrier.wait()     

n_workers = 8
barrier = Barrier(n_workers)
s_queue = Queue()
a_queue = Queue()
r_queue = Queue()
lock = Lock()

processes = []
for i in range(n_workers):
    worker = Process(target=runner, args=(barrier, lock, s_queue, a_queue, r_queue))
    processes.append(worker)
    worker.start()

for process in processes:
    process.join()    

