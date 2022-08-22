import numpy as np
import tensorflow as tf
import gym
import tensorflow_probability as tfp
import tensorflow.keras.losses as kls

seed = 1234
env_name = "CartPole-v0"
# set environment
env = gym.make(env_name)

state_low   = env.observation_space.low
state_high  = env.observation_space.high

print(state_low)
print(state_high)

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
    
class Critic(tf.keras.Model):
    def __init__(self):
        super(Critic, self).__init__()
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
        self.critic = Critic()
        self.gamma = 0.99
        # self.a_opt = tf.keras.optimizers.RMSprop(learning_rate=1e-5)
        # self.c_opt = tf.keras.optimizers.RMSprop(learning_rate=1e-5)
        self.a_opt = tf.keras.optimizers.Adam(learning_rate=7e-3)
        self.c_opt = tf.keras.optimizers.Adam(learning_rate=7e-3)
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
            #print(f"t{t}")
            #ratio = tf.math.exp(tf.math.log(pb + 1e-10) - tf.math.log(op + 1e-10))
            ratio = tf.math.divide(pb,op)
            #print(f"ratio{ratio}")
            s1 = tf.math.multiply(ratio,t)
            #print(f"s1{s1}")
            s2 =  tf.math.multiply(tf.clip_by_value(ratio, 1.0 - self.clip_pram, 1.0 + self.clip_pram),t)
            #print(f"s2{s2}")
            sur1.append(s1)
            sur2.append(s2)

        sr1 = tf.stack(sur1)
        sr2 = tf.stack(sur2)
        
        #closs = tf.reduce_mean(tf.math.square(td))
        loss = tf.math.negative(tf.reduce_mean(tf.math.minimum(sr1, sr2)) - closs + 0.001 * entropy)
        #print(loss)
        return loss

    def n_step_td_target(self, states, actions, rewards, done, values, gamma):
        g = 0
        lmbda = 0.95
        returns = []
        for i in reversed(range(len(rewards))):
            delta = rewards[i] + gamma * values[i + 1] * done[i] - values[i]
            g = delta + gamma * lmbda * dones[i] * g
            returns.append(g + values[i])

        returns.reverse()
        adv = np.array(returns, dtype=np.float32) - values[:-1]
        adv = (adv - np.mean(adv)) / (np.std(adv) + 1e-10)
        states         = np.array(states, dtype=np.float32)
        actions        = np.array(actions, dtype=np.int32)
        returns        = np.array(returns, dtype=np.float32)

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
            
            td = tf.math.subtract(discnt_rewards, curr_Qs)
            
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


if __name__ == "__main__":
    tf.random.set_seed(336699)
    agent = Agent()
    
    max_episodes = 400
    ep_reward = []
    total_avgr = []
    target = False 
    best_reward = 0
    avg_rewards_list = []
    
    for s in range(max_episodes):
        if target == True:
            break

        state = env.reset()
        done = False
        all_aloss = []
        all_closs = []
        states  = []
        actions = []
        rewards = []
        probs = []
        dones = []
        values = []
        print("new episod")

        for e in range(128):

            action = agent.get_action(state)
            value = agent.critic(np.array([state])).numpy()
            next_state, reward, done, _ = env.step(action)
            dones.append(1-done)
            rewards.append(reward)
            states.append(state)
            #actions.append(tf.one_hot(action, 2, dtype=tf.int32).numpy().tolist())
            actions.append(action)
            prob = agent.actor(np.array([state]))
            probs.append(prob[0])
            values.append(value[0][0])
            state = next_state
            if done:
                env.reset()

        value = agent.critic(np.array([state])).numpy()
        values.append(value[0][0])
        np.reshape(probs, (len(probs),2))
        probs = np.stack(probs, axis=0)

        states, actions, returns, adv  = agent.n_step_td_target(states, actions, rewards, dones, values, 1)

        for epocs in range(10):
            al,cl = agent.train_step(states, actions, adv, probs, returns)
            # print(f"al{al}") 
            # print(f"cl{cl}")   

        avg_reward = np.mean([test_reward(env) for _ in range(5)])
        print(f"total test reward is {avg_reward}")
        avg_rewards_list.append(avg_reward)
        if avg_reward > best_reward:
            print('best reward=' + str(avg_reward))
            agent.actor.save('model_actor_{}_{}'.format(s, avg_reward), save_format="tf")
            agent.critic.save('model_critic_{}_{}'.format(s, avg_reward), save_format="tf")
            best_reward = avg_reward
        if best_reward == 200:
            target = True
        env.reset()

    env.close()
    

import matplotlib.pyplot as plt

ep = [i  for i in range(len(avg_rewards_list))]
plt.plot( range(len(avg_rewards_list)),avg_rewards_list,'b')
plt.title("Avg Test Aeward Vs Test Episods")
plt.xlabel("Test Episods")
plt.ylabel("Average Test Reward")
plt.grid(True)
plt.show()

