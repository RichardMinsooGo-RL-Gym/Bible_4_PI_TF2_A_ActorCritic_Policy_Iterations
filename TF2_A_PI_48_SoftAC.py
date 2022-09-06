! pip install gym[box2d]

# IMPORTING LIBRARIES

import sys
IN_COLAB = "google.colab" in sys.modules

import gym
import numpy as np

import tensorflow as tf
from tensorflow.keras import Model

from IPython.display import clear_output
from tensorflow.keras.models import load_model
import tensorflow_probability as tfp

# !pip3 install box2d-py

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

class Actor(Model):
    def __init__(self, state_size: int, action_size: int, 
    ):
        """Initialization."""
        super(Actor, self).__init__()
        
        self.state_size = state_size
        self.action_size = action_size
        # set the hidden layers
        self.layer1 = tf.keras.layers.Dense(hidden_size, activation='relu')
        self.layer2 = tf.keras.layers.Dense(hidden_size, activation='relu')
        self.mu =  tf.keras.layers.Dense(action_size, activation=None)
        self.sigma =  tf.keras.layers.Dense(action_size, activation=None)
        self.min_action = -1
        self.max_action = 1
        self.repram = 1e-6

    def call(self, state):
        layer1 = self.layer1(state)
        layer2 = self.layer2(layer1)
        mu = self.mu(layer2)
        s = self.sigma(layer2)
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
        
class CriticQ(Model):
    def __init__(
        self, 
        state_size: int, 
    ):
        """Initialize."""
        super(CriticQ, self).__init__()
        self.layer1 = tf.keras.layers.Dense(hidden_size, activation='relu')
        self.layer2 = tf.keras.layers.Dense(hidden_size, activation='relu')
        self.value = tf.keras.layers.Dense(1, activation = None)

    def call(self, state, action):
        layer1 = self.layer1(tf.concat([state, action], axis=1))
        layer2 = self.layer2(layer1)
        value = self.value(layer2)
        return value

class CriticV(Model):
    def __init__(
        self, 
        state_size: int, 
    ):
        """Initialize."""
        super(CriticV, self).__init__()
        self.layer1 = tf.keras.layers.Dense(hidden_size, activation='relu')
        self.layer2 = tf.keras.layers.Dense(hidden_size, activation='relu')
        self.v =  tf.keras.layers.Dense(1, activation=None)

    def call(self, state):
        x = self.layer1(state)
        x = self.layer2(x)
        x = self.v(x)
        return x

class Agent():
    """
        
    Attributes:
        env (gym.Env): openAI Gym environment
        gamma (float): discount factor
        entropy_weight (float): rate of weighting entropy into the loss function
        actor (tf.keras.Model): target actor model to select actions
        critic (tf.keras.Model): critic model to predict state values
        actor_optimizer (optim.Optimizer) : optimizer of actor
        critic_optimizer (optim.Optimizer) : optimizer of critic
        transition (list): temporory storage for the recent transition
        is_test (bool): flag to show the current mode (train / test)
    """

    def __init__(
        self, 
        env: gym.Env,
    ):
        """Initialization.
        
        Args:
            env (gym.Env): openAI Gym environment
            gamma (float): discount factor
        """
        
        # CREATING THE Q-Network
        self.env = env
        
        self.state_size = self.env.observation_space.shape
        self.action_size = len(env.action_space.high)
        
        self.actor_lr = 7e-3
        self.critic_lr = 7e-3
        self.gamma = 0.99    # discount rate
        self.actor = Actor(self.state_size, self.action_size
                          )
        self.critic1 = CriticQ(self.state_size
                          )
        self.critic2 = CriticQ(self.state_size
                          )
        self.value_net = CriticV(self.state_size
                          )
        self.target_value_net = CriticV(self.state_size
                          )
        self.batch_size = 64
        # self.actor_target = tf.keras.optimizers.Adam(.001)
        # self.critictarget = tf.keras.optimizers.Adam(.002)
        self.a_opt = tf.keras.optimizers.Adam(learning_rate=self.actor_lr)
        self.c_opt1 = tf.keras.optimizers.Adam(learning_rate=self.critic_lr)
        self.c_opt2 = tf.keras.optimizers.Adam(learning_rate=self.critic_lr)
        self.v_opt  = tf.keras.optimizers.Adam(learning_rate=self.critic_lr)
        self.memory = ReplayBuffer(1_00_000, env.observation_space.shape, len(env.action_space.high))
        self.trainstep = 0
        self.update_freq = 5
        self.min_action = env.action_space.low[0]
        self.max_action = env.action_space.high[0]
        self.scale = 2
        self.update_target()
    
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
        actions     = tf.convert_to_tensor(actions, dtype= tf.float32)
        rewards     = tf.convert_to_tensor(rewards, dtype= tf.float32)
        next_states = tf.convert_to_tensor(next_states, dtype= tf.float32)
        # dones       = tf.convert_to_tensor(dones, dtype= tf.bool)
        
        with tf.GradientTape() as tape1:
            value = tf.squeeze(self.value_net(states))
            
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

        grads1 = tape1.gradient(value_loss, self.value_net.trainable_variables)
        self.v_opt.apply_gradients(zip(grads1, self.value_net.trainable_variables))
        
        with tf.GradientTape() as tape3, tf.GradientTape() as tape4:
            
            next_state_value = tf.squeeze(self.target_value_net(next_states)) 

            #critic loss          
            expected_Qs = self.scale * rewards + self.gamma * next_state_value * dones
            c_q1 = self.critic1(states, actions)
            c_q2 = self.critic2(states, actions)
            critic_loss1 = 0.5 * tf.keras.losses.MSE(expected_Qs, tf.squeeze(c_q1))
            critic_loss2 = 0.5 * tf.keras.losses.MSE(expected_Qs, tf.squeeze(c_q2))  
      
        grads3 = tape3.gradient(critic_loss1, self.critic1.trainable_variables)
        grads4 = tape4.gradient(critic_loss2, self.critic2.trainable_variables)
        self.c_opt1.apply_gradients(zip(grads3, self.critic1.trainable_variables))
        self.c_opt2.apply_gradients(zip(grads4, self.critic2.trainable_variables))
        
        self.trainstep +=1
        if self.trainstep % self.update_freq == 0:
            with tf.GradientTape() as tape2:

                #actor loss
                a_actions, a_log_probs = self.actor.sample_normal(states, reparameterize=True)
                #a_log_probs = tf.squeeze(a_log_probs, 1)
                a_q1 = self.critic1(states, a_actions)
                a_q2 = self.critic2(states, a_actions)
                a_critic_value = tf.math.minimum(tf.squeeze(a_q1), tf.squeeze(a_q2))
                actor_loss =  a_log_probs - a_critic_value
                actor_loss = tf.reduce_mean(actor_loss)

            grads2 = tape2.gradient(actor_loss, self.actor.trainable_variables)
            self.a_opt.apply_gradients(zip(grads2, self.actor.trainable_variables))
            self.update_target()

seed = 1234
# CREATING THE ENVIRONMENT
env_name = "LunarLanderContinuous-v2"
env = gym.make(env_name)
env.seed(seed)     # reproducible, general Policy gradient has high variance
state_low   = env.observation_space.low
state_high  = env.observation_space.high
action_low  = env.action_space.low 
action_high = env.action_space.high
print("state_low   :", state_low)
print("state_high  :", state_high)
print("action_low  :", action_low)
print("action_high :", action_high)

# INITIALIZING THE Q-PARAMETERS
hidden_size = 512
max_episodes = 300  # Set total number of episodes to train agent on.

# train
agent = Agent(
    env, 
#     memory_size, 
#     batch_size, 
#     epsilon_decay,
)

if __name__ == "__main__":
    tf.random.set_seed(336699)
    # TRAINING LOOP
    #List to contain all the rewards of all the episodes given to the agent
    scores = []
    
    # EACH EPISODE    
    for episode in range(max_episodes):
        ## Reset environment and get first new observation
        state = agent.env.reset()
        episode_reward = 0
        done = False  # has the enviroment finished?
        
            
        # EACH TIME STEP    
        while not done:
        # for step in range(max_steps):  # step index, maximum step is 200
            action = agent.get_action(state)
            
            # TAKING ACTION
            next_state, reward, done, _ = agent.env.step(action)
            
            agent.savexp(state, action, reward, next_state, done)
            agent.train_step()
            
            # Our new state is state
            state = next_state
            
            episode_reward += reward

            # if episode ends
            if done:
                scores.append(episode_reward)
                print("Episode " + str(episode+1) + ": " + str(episode_reward))
                
                break

episode_reward = 0
state = env.reset()
while not done:
    action = agent.act(state, True)
    next_state, reward, done, _ = env.step(action)
    state = next_state
    episode_reward += reward
    if done:
        print(episode_reward)

