import sys
IN_COLAB = "google.colab" in sys.modules

import numpy as np
import tensorflow as tf
import gym
import tensorflow_probability as tfp
import tensorflow.keras.losses as kls

seed = 1234
env_name = "CartPole-v0"
# set environment
env = gym.make(env_name)
env.seed(seed)     # reproducible, general Policy gradient has high variance

state_size = env.observation_space.shape[0]
action_size = env.action_space.n
hidden_size = 64

class Actor(tf.keras.Model):
    def __init__(self):
        super(Actor, self).__init__()    
        self.fc1 = tf.keras.layers.Dense(hidden_size,activation='relu')
        self.fc2 = tf.keras.layers.Dense(hidden_size,activation='relu')
        self.out = tf.keras.layers.Dense(action_size,activation='softmax')

    def call(self, state):
        x = self.fc1(state)
        x = self.fc2(x)
        mu = self.out(x)
        return mu
    
class CriticV(tf.keras.Model):
    def __init__(self):
        super(CriticV, self).__init__()
        self.fc1 = tf.keras.layers.Dense(hidden_size,activation='relu')
        self.fc2 = tf.keras.layers.Dense(hidden_size,activation='relu')
        self.v = tf.keras.layers.Dense(1, activation = None)

    def call(self, state):
        x = self.fc1(state)
        x = self.fc2(x)
        v = self.v(x)
        return v

class Agent():
    def __init__(self):
        self.actor = Actor()
        self.critic = CriticV()
        self.gamma = 0.99
        # self.a_opt = tf.keras.optimizers.RMSprop(learning_rate=1e-5)
        # self.c_opt = tf.keras.optimizers.RMSprop(learning_rate=1e-5)
        self.a_opt = tf.keras.optimizers.Adam(learning_rate=7e-3)
        self.c_opt = tf.keras.optimizers.Adam(learning_rate=7e-3)
        self.log_prob = None
    
    def get_action(self, state):
        prob = self.actor(np.array([state]))
        prob = prob.numpy()
        dist = tfp.distributions.Categorical(probs=prob, dtype=tf.float32)
        action = dist.sample()
        return int(action.numpy()[0])
    
    def actor_loss(self, prob, action, TD):
        
        dist = tfp.distributions.Categorical(probs=prob, dtype=tf.float32)
        log_prob = dist.log_prob(action)
        loss = -log_prob*TD
        return loss
    
    def train_step(self, state, action, reward, next_state, done):
        state = np.array([state])
        next_state = np.array([next_state])
        
        with tf.GradientTape() as tape1, tf.GradientTape() as tape2:
            
            curr_P = self.actor(state, training=True)
            curr_Q = self.critic(state,training=True)
            next_Q = self.critic(next_state, training=True)
            
            expected_Q = reward + self.gamma*next_Q*(1-int(done))
            TD = expected_Q - curr_Q
            
            # critic loss
            critic_loss = tf.keras.losses.MSE(expected_Q, curr_Q)
            
            actor_loss = self.actor_loss(curr_P, action, TD)
            
        actorGrads = tape1.gradient(actor_loss,  self.actor.trainable_variables)
        criticGrads = tape2.gradient(critic_loss, self.critic.trainable_variables)
        self.a_opt.apply_gradients(zip(actorGrads, self.actor.trainable_variables))
        self.c_opt.apply_gradients(zip(criticGrads, self.critic.trainable_variables))
        
        return actor_loss, critic_loss

if __name__ == "__main__":
    tf.random.set_seed(336699)
    agent = Agent()
    
    max_episodes = 400

    for episode in range(max_episodes):

        state = env.reset()
        done = False
        total_reward = 0

        while not done:
            #env.render()
            action = agent.get_action(state)
            next_state, reward, done, _ = env.step(action)
            aloss, closs = agent.train_step(state, action, reward, next_state, done)
            
            state = next_state
            total_reward += reward

            if done:
                #print("total step for this episord are {}".format(t))
                print("total reward after {} steps is {}".format(episode+1, total_reward))

