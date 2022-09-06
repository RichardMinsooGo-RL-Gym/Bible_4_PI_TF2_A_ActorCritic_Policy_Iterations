# IMPORTING LIBRARIES

import sys
IN_COLAB = "google.colab" in sys.modules

import gym
import numpy as np

import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow.keras import Model
import tensorflow.keras.losses as kls

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
        self.clip_pram = 0.2
    
    def get_action(self, state):
        prob = self.actor(np.array([state]))
        prob = prob.numpy()
        dist = tfp.distributions.Categorical(probs=prob, dtype=tf.float32)
        action = dist.sample()
        return int(action.numpy()[0])
    
    def actor_loss(self, probs, actions, adv, old_probs, closs):
        
        probability = probs      
        entropy = tf.reduce_mean(tf.math.negative(tf.math.multiply(probability,tf.math.log(probability))))
        #print(probability)
        #print(entropy)
        sur1 = []
        sur2 = []
        
        for pb, t, op in zip(probability, adv, old_probs):
            t =  tf.constant(t)
            op =  tf.constant(op)
            # print(f"t{t}")
            # ratio = tf.math.exp(tf.math.log(pb + 1e-10) - tf.math.log(op + 1e-10))
            ratio = tf.math.divide(pb,op)
            # print(f"ratio{ratio}")
            s1 = tf.math.multiply(ratio,t)
            # print(f"s1{s1}")
            s2 =  tf.math.multiply(tf.clip_by_value(ratio, 1.0 - self.clip_pram, 1.0 + self.clip_pram),t)
            # print(f"s2{s2}")
            sur1.append(s1)
            sur2.append(s2)

        sr1 = tf.stack(sur1)
        sr2 = tf.stack(sur2)
        
        #closs = tf.reduce_mean(tf.math.square(td))
        loss = tf.math.negative(tf.reduce_mean(tf.math.minimum(sr1, sr2)) - closs + 0.001 * entropy)
        #print(loss)
        return loss
    
    def gae_target(self, states, actions, rewards, done, values, gamma):
        gae = 0
        lmbda = 0.95
        returns = []
        for i in reversed(range(len(rewards))):
            delta = rewards[i] + gamma * values[i + 1] * done[i] - values[i]
            gae = delta + gamma * lmbda * dones[i] * gae
            returns.append(gae + values[i])

        returns.reverse()
        adv     = np.array(returns, dtype=np.float32) - values[:-1]
        adv     = (adv - np.mean(adv)) / (np.std(adv) + 1e-10)
        states  = np.array(states, dtype=np.float32)
        actions = np.array(actions, dtype=np.int32)
        returns = np.array(returns, dtype=np.float32)
        
        return states, actions, returns, adv    
        
    def train_step(self, states, actions,  adv , old_probs, discnt_rewards):
        discnt_rewards = tf.reshape(discnt_rewards, (len(discnt_rewards),))
        adv = tf.reshape(adv, (len(adv),))

        old_p = old_probs

        old_p = tf.reshape(old_p, (len(old_p),2))
        with tf.GradientTape() as tape1, tf.GradientTape() as tape2:
            
            curr_Ps = self.actor(states, training=True)
            curr_Qs = self.critic(states,training=True)
            curr_Qs = tf.reshape(curr_Qs, (len(curr_Qs),))
            
            # TDs = tf.math.subtract(discnt_rewards, curr_Qs)
            
            critic_loss = 0.5 * kls.mean_squared_error(discnt_rewards, curr_Qs)
            actor_loss = self.actor_loss(curr_Ps, actions, adv, old_probs, critic_loss)
            
        actorGrads = tape1.gradient(actor_loss,  self.actor.trainable_variables)
        criticGrads = tape2.gradient(critic_loss, self.critic.trainable_variables)
        self.a_opt.apply_gradients(zip(actorGrads, self.actor.trainable_variables))
        self.c_opt.apply_gradients(zip(criticGrads, self.critic.trainable_variables))
        
        return actor_loss, critic_loss


def test_reward(env):
    total_reward = 0
    state = env.reset()
    done = False
    while not done:
        action = np.argmax(agent.actor(np.array([state])).numpy())
        next_state, reward, done, _ = env.step(action)
        state = next_state
        total_reward += reward

    return total_reward

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
    # TRAINING LOOP
    #List to contain all the rewards of all the episodes given to the agent
    scores = []
    total_avgr = []
    target = False 
    best_reward = 0
    avg_rewards_list = []
    
    # EACH EPISODE    
    for episode in range(max_episodes):
        ## Reset environment and get first new observation
        state = agent.env.reset()
        episode_reward = 0
        done = False  # has the enviroment finished?
        all_aloss = []
        all_closs = []
        states  = []
        actions = []
        rewards = []
        probs   = []
        dones   = []
        values  = []

        while not done:
        # for step in range(max_steps):  # step index, maximum step is 200
            action = agent.get_action(state)
            value  = agent.critic(np.array([state])).numpy()
            # TAKING ACTION
            next_state, reward, done, _ = agent.env.step(action)
            
            states.append(state)
            actions.append(action)
            rewards.append(reward)
            dones.append(1-done)
            
            prob = agent.actor(np.array([state]))
            probs.append(prob[0])
            values.append(value[0][0])
            
            state = next_state
            
            episode_reward += reward

            # if episode ends
            if done:
                scores.append(episode_reward)
                print("Episode " + str(episode+1) + ": " + str(episode_reward))
                value = agent.critic(np.array([state])).numpy()
                values.append(value[0][0])
                np.reshape(probs, (len(probs),2))
                probs = np.stack(probs, axis=0)

                states, actions, returns, adv  = agent.gae_target(states, actions, rewards, dones, values, 1)

                for epocs in range(7):
                    al,cl = agent.train_step(states, actions, adv, probs, returns)
                    # print(f"al{al}") 
                    # print(f"cl{cl}")   

                avg_reward = np.mean([test_reward(env) for _ in range(5)])
                print(f"total test reward is {avg_reward}")
                avg_rewards_list.append(avg_reward)
                if avg_reward > best_reward:
                    print('best reward=' + str(avg_reward))
                    best_reward = avg_reward
                break

