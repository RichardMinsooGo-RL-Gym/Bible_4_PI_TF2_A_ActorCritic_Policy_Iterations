# IMPORTING LIBRARIES

import sys
IN_COLAB = "google.colab" in sys.modules

import gym
import numpy as np

import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow.keras import Model

from IPython.display import clear_output

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
        self.policy = tf.keras.layers.Dense(self.action_size,activation='softmax')

    def call(self, state):
        layer1 = self.layer1(state)
        layer2 = self.layer2(layer1)
        policy = self.policy(layer2)
        return policy
    
class CriticV(Model):
    def __init__(
        self, 
        state_size: int, 
    ):
        """Initialize."""
        super(CriticV, self).__init__()
        self.layer1 = tf.keras.layers.Dense(hidden_size, activation='relu')
        self.layer2 = tf.keras.layers.Dense(hidden_size, activation='relu')
        self.value = tf.keras.layers.Dense(1, activation = None)

    def call(self, state):
        layer1 = self.layer1(state)
        layer2 = self.layer2(layer1)
        value = self.value(layer2)
        return value

class DQNAgent:
    """A2CAgent interacting with environment.
        
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
        
        self.state_size = self.env.observation_space.shape[0]
        self.action_size = self.env.action_space.n
        
        self.actor_lr = 7e-3
        self.critic_lr = 7e-3
        self.gamma = 0.99    # discount rate
        self.actor = Actor(self.state_size, self.action_size
                          )
        self.critic = CriticV(self.state_size
                          )
        # self.a_opt = tf.keras.optimizers.RMSprop(learning_rate=1e-5)
        # self.c_opt = tf.keras.optimizers.RMSprop(learning_rate=1e-5)
        self.a_opt = tf.keras.optimizers.Adam(learning_rate=self.actor_lr)
        self.c_opt = tf.keras.optimizers.Adam(learning_rate=self.critic_lr)
        self.log_prob = None
    
    def get_action(self, state):
        prob = self.actor(np.array([state]))
        prob = prob.numpy()
        dist = tfp.distributions.Categorical(probs=prob, dtype=tf.float32)
        action = dist.sample()
        return int(action.numpy()[0])
    
    def actor_loss(self, prob, action, TDs):
        
        dist = tfp.distributions.Categorical(probs=prob, dtype=tf.float32)
        log_prob = dist.log_prob(action)
        loss = -log_prob*TDs
        return loss
    
    def n_step_td_target(self, states, actions, rewards, gamma):
        R_to_Go = 0
        discnt_rewards = []
        rewards.reverse()
        for r in rewards:
            R_to_Go = r + self.gamma*R_to_Go
            discnt_rewards.append(R_to_Go)
        discnt_rewards.reverse()
        states  = np.array(states, dtype=np.float32)
        actions = np.array(actions, dtype=np.int32)
        discnt_rewards = np.array(discnt_rewards, dtype=np.float32)
        
        return states, actions, discnt_rewards
        
    def train_step(self, states, actions, discnt_rewards):
        discnt_rewards = tf.reshape(discnt_rewards, (len(discnt_rewards),))
        
        with tf.GradientTape() as tape1, tf.GradientTape() as tape2:
            
            curr_Ps = self.actor(states, training=True)
            curr_Qs = self.critic(states,training=True)
            curr_Qs = tf.reshape(curr_Qs, (len(curr_Qs),))
            
            TDs = tf.math.subtract(discnt_rewards, curr_Qs)
            
            critic_loss = 0.5 * kls.mean_squared_error(discnt_rewards, curr_Qs)
            actor_loss = self.actor_loss(curr_Ps, actions, TDs)
            
        actorGrads = tape1.gradient(actor_loss,  self.actor.trainable_variables)
        criticGrads = tape2.gradient(critic_loss, self.critic.trainable_variables)
        self.a_opt.apply_gradients(zip(actorGrads, self.actor.trainable_variables))
        self.c_opt.apply_gradients(zip(criticGrads, self.critic.trainable_variables))
        
        return actor_loss, critic_loss

seed = 1234
# CREATING THE ENVIRONMENT
env_name = "CartPole-v0"
env = gym.make(env_name)
env.seed(seed)     # reproducible, general Policy gradient has high variance

# INITIALIZING THE Q-PARAMETERS
hidden_size = 64
max_episodes = 300  # Set total number of episodes to train agent on.

# train
agent = DQNAgent(
    env, 
#     memory_size, 
#     batch_size, 
#     epsilon_decay,
)

if __name__ == "__main__":
    tf.random.set_seed(336699)
    # 2.5 TRAINING LOOP
    #List to contain all the rewards of all the episodes given to the agent
    scores = []
    
    # EACH EPISODE    
    for episode in range(max_episodes):
        ## Reset environment and get first new observation
        state = agent.env.reset()
        episode_reward = 0
        done = False  # has the enviroment finished?
        
        states  = []
        actions = []
        rewards = []
            
        # EACH TIME STEP    
        while not done:
        # for step in range(max_steps):  # step index, maximum step is 200
            action = agent.get_action(state)
            
            # TAKING ACTION
            next_state, reward, done, _ = agent.env.step(action)
            
            states.append(state)
            actions.append(action)
            rewards.append(reward)
            
            # Our new state is state
            state = next_state
            
            episode_reward += reward

            # if episode ends
            if done:
                scores.append(episode_reward)
                print("Episode " + str(episode+1) + ": " + str(episode_reward))
                
                states, actions, discnt_rewards = agent.n_step_td_target(states, actions, rewards, 1)
                actor_loss, critic_loss = agent.train_step(states, actions, discnt_rewards) 
                break

