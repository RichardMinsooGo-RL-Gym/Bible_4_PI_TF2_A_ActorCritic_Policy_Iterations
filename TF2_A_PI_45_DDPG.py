import sys
IN_COLAB = "google.colab" in sys.modules

import numpy as np
import tensorflow as tf
import gym
from tensorflow.keras.models import load_model

!pip3 install box2d-py

env= gym.make("LunarLanderContinuous-v2")

state_low   = env.observation_space.low
state_high  = env.observation_space.high
action_low  = env.action_space.low 
action_high = env.action_space.high
print(state_low)
print(state_high)
print(action_low)
print(action_high)

action_size = len(env.action_space.high)
hidden_size = 512

class ReplayBuffer():
    def __init__(self, maxsize, statedim, naction):
        self.cnt = 0
        self.maxsize = maxsize
        self.state_memory = np.zeros((maxsize, *statedim), dtype=np.float32)
        self.action_memory = np.zeros((maxsize, naction), dtype=np.float32)
        self.reward_memory = np.zeros((maxsize,), dtype=np.float32)
        self.next_state_memory = np.zeros((maxsize, *statedim), dtype=np.float32)
        self.done_memory = np.zeros((maxsize,), dtype= np.bool)

    def storexp(self, state, action, reward, next_state, done):
        index = self.cnt % self.maxsize
        self.state_memory[index] = state
        self.action_memory[index] = action
        self.reward_memory[index] = reward
        self.next_state_memory[index] = next_state
        self.done_memory[index] = 1- int(done)
        self.cnt += 1

    def sample(self, batch_size):
        max_mem = min(self.cnt, self.maxsize)
        batch = np.random.choice(max_mem, batch_size, replace= False)  
        states = self.state_memory[batch]
        next_states = self.next_state_memory[batch]
        rewards = self.reward_memory[batch]
        actions = self.action_memory[batch]
        dones = self.done_memory[batch]
        return states, next_states, rewards, actions, dones

class Actor(tf.keras.Model):
    def __init__(self):
        super(Actor, self).__init__()    
        self.fc1 = tf.keras.layers.Dense(hidden_size,activation='relu')
        self.fc2 = tf.keras.layers.Dense(hidden_size,activation='relu')
        self.mu =  tf.keras.layers.Dense(action_size, activation='tanh')

    def call(self, state):
        x = self.fc1(state)
        x = self.fc2(x)
        mu = self.mu(x)
        return mu
    
class CriticQ(tf.keras.Model):
    def __init__(self):
        super(CriticQ, self).__init__()
        self.fc1 = tf.keras.layers.Dense(hidden_size,activation='relu')
        self.fc2 = tf.keras.layers.Dense(hidden_size,activation='relu')
        self.v = tf.keras.layers.Dense(1, activation = None)

    def call(self, state, action):
        x = self.fc1(tf.concat([state, action], axis=1))
        x = self.fc2(x)
        v = self.v(x)
        return v

class Agent():
    def __init__(self):
        self.actor = Actor()
        self.actor_target = Actor()
        self.critic = CriticQ()
        self.critic_target = CriticQ()
        self.gamma = 0.99
        self.batch_size = 64
        self.action_size = len(env.action_space.high)
        # self.a_opt = tf.keras.optimizers.RMSprop(learning_rate=1e-5)
        # self.c_opt = tf.keras.optimizers.RMSprop(learning_rate=1e-5)
        self.a_opt = tf.keras.optimizers.Adam(learning_rate=7e-3)
        self.c_opt = tf.keras.optimizers.Adam(learning_rate=7e-3)
        self.memory = ReplayBuffer(1_00_000, env.observation_space.shape, len(env.action_space.high))
        self.trainstep = 0
        self.replace = 5
        self.min_action = env.action_space.low[0]
        self.max_action = env.action_space.high[0]

    def get_action(self, state, evaluate=False):
        state = tf.convert_to_tensor([state], dtype=tf.float32)
        actions = self.actor(state)
        if not evaluate:
            actions += tf.random.normal(shape=[self.action_size], mean=0.0, stddev=0.1)

        actions = self.max_action * (tf.clip_by_value(actions, self.min_action, self.max_action))
        #print(actions)
        return actions[0]
    
    def savexp(self,state, next_state, action, done, reward):
        self.memory.storexp(state, next_state, action, done, reward)

    def update_target(self):
        self.actor_target.set_weights(self.actor.get_weights())
        self.critic_target.set_weights(self.critic.get_weights())

    def train_step(self):
        if self.memory.cnt < self.batch_size:
            return 

        states, next_states, rewards, actions, dones = self.memory.sample(self.batch_size)

        states      = tf.convert_to_tensor(states, dtype= tf.float32)
        next_states = tf.convert_to_tensor(next_states, dtype= tf.float32)
        rewards     = tf.convert_to_tensor(rewards, dtype= tf.float32)
        actions     = tf.convert_to_tensor(actions, dtype= tf.float32)
        # dones       = tf.convert_to_tensor(dones, dtype= tf.bool)
        
        with tf.GradientTape() as tape1, tf.GradientTape() as tape2:
            
            curr_Ps = self.actor(states)
            curr_Qs = tf.squeeze(self.critic(states, actions), 1)
            next_P_targs = self.actor_target(next_states)
            next_Q_targs = tf.squeeze(self.critic_target(next_states, next_P_targs), 1)
            expected_Qs  = rewards + self.gamma * next_Q_targs * dones
            
            # critic loss
            critic_loss = tf.keras.losses.MSE(expected_Qs, curr_Qs)
            
            actor_loss = -self.critic(states, curr_Ps)
            actor_loss = tf.math.reduce_mean(actor_loss)
            
        actorGrads = tape1.gradient(actor_loss,  self.actor.trainable_variables)
        criticGrads = tape2.gradient(critic_loss, self.critic.trainable_variables)
        self.a_opt.apply_gradients(zip(actorGrads, self.actor.trainable_variables))
        self.c_opt.apply_gradients(zip(criticGrads, self.critic.trainable_variables))
        
        self.trainstep +=1
        if self.trainstep % self.replace == 0:
            self.update_target()

if __name__ == "__main__":
    tf.random.set_seed(336699)
    agent = Agent()
    
    max_episodes = 400
    ep_reward = []
    total_avgr = []
    target = False

    for episode in range(max_episodes):

        state = env.reset()
        done = False
        total_reward = 0

        while not done:
            #env.render()
            action = agent.get_action(state)
            next_state, reward, done, _ = env.step(action)
            agent.savexp(state, action, reward, next_state, done)
            agent.train_step()
            state = next_state
            total_reward += reward

            if done:
                ep_reward.append(total_reward)
                avg_reward = np.mean(ep_reward[-100:])
                total_avgr.append(avg_reward)
                print("total reward after {} steps is {} and avg reward is {}".format(episode+1, total_reward, avg_reward))
                if avg_reward == 200:
                    target = True

total_reward = 0
state = env.reset()
while not done:
    action = agent.act(state, True)
    next_state, reward, done, _ = env.step(action)
    state = next_state
    total_reward += reward
    if done:
        print(total_reward)

