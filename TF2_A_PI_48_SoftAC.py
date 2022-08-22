import numpy as np
import tensorflow as tf
import gym
from tensorflow.keras.models import load_model
import tensorflow_probability as tfp

!pip3 install box2d-py

print(tf.config.list_physical_devices('GPU'))

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
    def __init__(self, action_size):
        super(Actor, self).__init__()    
        self.fc1 = tf.keras.layers.Dense(hidden_size,activation='relu')
        self.fc2 = tf.keras.layers.Dense(hidden_size,activation='relu')
        self.mu =  tf.keras.layers.Dense(action_size, activation=None)
        self.sigma =  tf.keras.layers.Dense(action_size, activation=None)
        self.min_action = -1
        self.max_action = 1
        self.repram = 1e-6

    def call(self, state):
        x = self.fc1(state)
        x = self.fc2(x)
        mu = self.mu(x)
        s = self.sigma(x)
        s = tf.clip_by_value(s, self.repram, 1)
        return mu, s
    
    def sample_normal(self, state, reparameterize=True):
        mu, sigma = self(state)
        # print(mu)
        # print(sigma)
        #mu = tf.squeeze(mu)
        #sigma =tf.squeeze(sigma)
        #print(mu)
        probabilities = tfp.distributions.Normal(mu, sigma)
        if reparameterize:
            actions = probabilities.sample()
            #actions += tf.random.normal(shape=tf.shape(actions), mean=0.0, stddev=0.1)

        else:
            actions = probabilities.sample()

        action = tf.math.scalar_mul(tf.constant(self.max_action, dtype=tf.float32),tf.math.tanh(actions))
        action = tf.squeeze(action)
        log_prob = probabilities.log_prob(actions)
        log_prob -= tf.math.log(1 - tf.math.pow(action, 2) + self.repram)
        log_prob = tf.reduce_sum(log_prob, axis=1)

        return action, log_prob
        
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

class CriticV(tf.keras.Model):
    def __init__(self):
        super(CriticV, self).__init__()
        self.fc1 = tf.keras.layers.Dense(hidden_size, activation='relu')
        self.fc2 = tf.keras.layers.Dense(hidden_size, activation='relu')
        self.v =  tf.keras.layers.Dense(1, activation=None)

    def call(self, inputstate):
        x = self.fc1(inputstate)
        x = self.fc2(x)
        x = self.v(x)
        return x

class Agent():
    def __init__(self):
        self.actor = Actor(action_size)
        self.critic1 = CriticQ()
        self.critic2 = CriticQ()
        self.value_net = CriticV()
        self.target_value_net = CriticV()
        self.gamma = 0.99
        self.batch_size = 64
        self.action_size = len(env.action_space.high)
        # self.actor_target = tf.keras.optimizers.Adam(.001)
        # self.critictarget = tf.keras.optimizers.Adam(.002)
        
        self.a_opt = tf.keras.optimizers.Adam(learning_rate=7e-3)
        self.c_opt1 = tf.keras.optimizers.Adam(learning_rate=7e-3)
        self.c_opt2 = tf.keras.optimizers.Adam(learning_rate=7e-3)
        self.v_opt  = tf.keras.optimizers.Adam(learning_rate=7e-3)
        
        self.memory = ReplayBuffer(1_00_000, env.observation_space.shape, len(env.action_space.high))
        self.trainstep = 0
        self.replace = 5
        self.min_action = env.action_space.low[0]
        self.max_action = env.action_space.high[0]
        self.scale = 2
    

    def get_action(self, state):
        state = tf.convert_to_tensor([state], dtype=tf.float32)
        action, _ = self.actor.sample_normal(state, reparameterize=False)
        # print(action)
        return action
    
    def savexp(self,state, next_state, action, done, reward):
        self.memory.storexp(state, next_state, action, done, reward)

    def update_target(self):
        self.target_value_net.set_weights(self.value_net.get_weights())

    def train_step(self):
        if self.memory.cnt < self.batch_size:
            return 

        states, next_states, rewards, actions, dones = self.memory.sample(self.batch_size)

        states      = tf.convert_to_tensor(states, dtype= tf.float32)
        next_states = tf.convert_to_tensor(next_states, dtype= tf.float32)
        rewards     = tf.convert_to_tensor(rewards, dtype= tf.float32)
        actions     = tf.convert_to_tensor(actions, dtype= tf.float32)
        # dones       = tf.convert_to_tensor(dones, dtype= tf.bool)
        
        with tf.GradientTape() as tape1, tf.GradientTape() as tape2, tf.GradientTape() as tape3, tf.GradientTape() as tape4:
            value = tf.squeeze(self.value_net(states))
            #value = self.CriticV(states)
            
            # value loss
            v_actions, v_log_probs = self.actor.sample_normal(states, reparameterize=False)
            #print(v_log_probs)
            #v_log_probs = tf.squeeze(v_log_probs, 1)
            v_q1 = self.critic1(states, v_actions)
            v_q2 = self.critic2(states, v_actions)
            v_critic_value = tf.math.minimum(tf.squeeze(v_q1), tf.squeeze(v_q2))
            target_value = v_critic_value - v_log_probs
            #print(target_value)   
            value_loss = 0.5 * tf.keras.losses.MSE(target_value, value)

            next_state_value = tf.squeeze(self.target_value_net(next_states)) 
            #next_state_value = self.target_value_net(next_states)
            #critic loss          
            expected_Qs = self.scale * rewards + self.gamma * next_state_value * dones
            c_q1 = self.critic1(states, actions)
            c_q2 = self.critic2(states, actions)
            critic_loss1 = 0.5 * tf.keras.losses.MSE(expected_Qs, tf.squeeze(c_q1))
            critic_loss2 = 0.5 * tf.keras.losses.MSE(expected_Qs, tf.squeeze(c_q2))  
      
            #actor loss
            a_actions, a_log_probs = self.actor.sample_normal(states, reparameterize=True)
            #a_log_probs = tf.squeeze(a_log_probs, 1)
            a_q1 = self.critic1(states, a_actions)
            a_q2 = self.critic2(states, a_actions)
            a_critic_value = tf.math.minimum(tf.squeeze(a_q1), tf.squeeze(a_q2))
            actor_loss =  a_log_probs - a_critic_value
            actor_loss = tf.reduce_mean(actor_loss)

        grads1 = tape1.gradient(value_loss, self.value_net.trainable_variables)
        self.v_opt.apply_gradients(zip(grads1, self.value_net.trainable_variables))
        
        grads3 = tape3.gradient(critic_loss1, self.critic1.trainable_variables)
        grads4 = tape4.gradient(critic_loss2, self.critic2.trainable_variables)
        self.c_opt1.apply_gradients(zip(grads3, self.critic1.trainable_variables))
        self.c_opt2.apply_gradients(zip(grads4, self.critic2.trainable_variables))

        grads2 = tape2.gradient(actor_loss, self.actor.trainable_variables)
        self.a_opt.apply_gradients(zip(grads2, self.actor.trainable_variables))
        
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
        if target == True:
            break
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



