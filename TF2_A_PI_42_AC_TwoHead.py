import numpy as np
import tensorflow as tf
import gym
import tensorflow_probability as tfp

!pip3 install box2d-py

env= gym.make("LunarLander-v2")
low = env.observation_space.low
high = env.observation_space.high

class model(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.d1 = tf.keras.layers.Dense(2048,activation='relu')
        self.d2 = tf.keras.layers.Dense(1536,activation='relu')
        self.v = tf.keras.layers.Dense(1, activation = None)
        self.a = tf.keras.layers.Dense(4,activation=None)

    def call(self, input_data):
        x = self.d1(input_data)
        x1 = self.d2(x)
        a = self.a(x1)
        v = self.v(x1)
        return v, a
    
class agent():
    def __init__(self, gamma = 0.99):
        self.gamma = gamma
        self.opt = tf.keras.optimizers.Adam(learning_rate=5e-6)
        self.ac = model()
    
    def act(self,state):
        v, prob = self.ac(np.array([state]))
        prob = tf.nn.softmax(prob)
        #print(prob)
        prob = prob.numpy()
        dist = tfp.distributions.Categorical(probs=prob, dtype=tf.float32)
        action = dist.sample()
        return int(action.numpy()[0])
    
    def actor_loss(self, prob, action, td):
        prob = tf.nn.softmax(prob)
        dist = tfp.distributions.Categorical(probs=prob, dtype=tf.float32)
        log_prob = dist.log_prob(action)
        loss = -log_prob*td
        return loss    

    def learn(self, state, action, reward, next_state, done):
        state = np.array([state])
        next_state = np.array([next_state])

        with tf.GradientTape() as tape:
            v, a =  self.ac(state,training=True)
            vn, an = self.ac(next_state, training=True)
            td = reward + self.gamma*vn*(1-int(done)) - v
            a_loss = self.actor_loss(a,action,td)
            c_loss = td**2
            total_loss = a_loss + c_loss
        grads = tape.gradient(total_loss, self.ac.trainable_variables)
        self.opt.apply_gradients(zip(grads, self.ac.trainable_variables))
        return total_loss

agentoo7 = agent()
steps = 500
for s in range(steps):
  
    done = False
    state = env.reset()
    total_reward = 0
    all_loss = []

    while not done:
        #env.render()
        action = agentoo7.act(state)
        #print(action)
        next_state, reward, done, _ = env.step(action)
        loss = agentoo7.learn(state, action, reward, next_state, done)
        all_loss.append(loss)

        state = next_state
        total_reward += reward

        if done:
            #print("total step for this episord are {}".format(t))
            print("total reward after {} steps is {}".format(s, total_reward))


