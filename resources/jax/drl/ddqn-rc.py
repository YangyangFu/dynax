# %% [markdown]
# Double deep q learning
# 
# The difference between DDQN and DQN:
# - In DQN, the same network is used to select the best action and to estimate the value of that action. This can lead to an overoptimistic estimation of Q-values, which may result in suboptimal policies.
# - DDQN decouples the action selection and action value estimation by using two separate networks: the online Q-network (with weights θ) and the target Q-network (with weights θ').
#   - the online Q-network is used to select the best action, the target Q-network is used to estimate the value of that action
#   - the target value computation in DDQN is as follows:
#     - use the online Q to select the best action
#     - use the target Q to estimate the value of taking this action
#     - compute the target value using Bellman optimality with the target Q-network
#   - the online Q network is updated as DQN, and the target Q-network is updated periodically by copying the weights from the online Q-network.
#  

# %%
import gymnasium as gym
import jax
import jax.numpy as jnp
import numpy as np
from collections import deque
import random
from flax import linen as nn
import optax
import matplotlib 
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
import pandas as pd
import json

# %%
# need a virtual display for rendering in docker
from pyvirtualdisplay import Display
display = Display(visible=0, size=(1400, 900))
display.start()
from IPython import display as ipythondisplay
# %%
class QNetwork(nn.Module):
    action_size: int
    
    def setup(self):
        self.dense1 = nn.Dense(features=256)
        self.dense2 = nn.Dense(features=256)
        self.dense3 = nn.Dense(features=256)
        self.dense4 = nn.Dense(features=self.action_size)

    def __call__(self, x):
        x = nn.relu(self.dense1(x))
        x = nn.relu(self.dense2(x))
        x = nn.relu(self.dense3(x))
        x = self.dense4(x)
        return x

# %%
class DDQNAgent:
    def __init__(self, state_size, action_size, rng_key, buffer_size=10000, batch_size=64, gamma=0.99, lr=1e-3, epsilon_start=1.0, epsilon_end=0.01, epsilon_decay=0.99):
        self.state_size = state_size
        self.action_size = action_size
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.gamma = gamma
        self.lr = lr
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.memory = deque(maxlen=buffer_size)

        rng_key, rng_key_init = jax.random.split(rng_key)
        self.network, self.params, self.optimizer, self.opt_state = self.initialize_network_and_optimizer(rng_key_init)
        self.target_params = self.params
        self.tau = 0.001
        self.steps = 0

        # debug
        self.losses = []
        self.grads = []
    
    def initialize_network_and_optimizer(self, rng):
        network = QNetwork(self.action_size)
        params = network.init(rng, jnp.ones((self.state_size,)))
        optimizer = optax.adam(self.lr)
        opt_state = optimizer.init(params)
        return network, params, optimizer, opt_state

    def sync_target(self):
        #self.target_params = self.params
        # this is soft update for each step
        self.target_params = jax.tree_map(lambda x, y: self.tau * x + (1 - self.tau) * y, self.params, self.target_params)

    def act(self, state):
        if random.random() <= self.epsilon:
            return random.randint(0, self.action_size - 1)
        else:
            state = jnp.expand_dims(jnp.array(state, dtype=jnp.float32), axis=0)
            q_values = self.network.apply(self.params, state)
            return int(jnp.argmax(q_values))

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def replay(self):
        if len(self.memory) < self.batch_size:
            return

        batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = map(np.array, zip(*batch))

        states = jnp.array(states, dtype=jnp.float32)
        actions = jnp.array(actions, dtype=jnp.int32)
        rewards = jnp.array(rewards, dtype=jnp.float32)
        next_states = jnp.array(next_states, dtype=jnp.float32)
        dones = jnp.array(dones, dtype=jnp.float32)

        def loss(params):
            # (s,a,r,s')
            # Q(s)
            q_values = self.network.apply(params, states)
            # Q(s,a)
            q_values = jnp.take_along_axis(q_values, actions[:, None], axis=-1).squeeze()#jax.vmap(lambda s: s[actions])(q_values)

            # Q(s')
            online_q_next = self.network.apply(params, next_states)
            # Q'(s')
            target_q_next = self.network.apply(self.target_params, next_states)
            
            # a' = argmax_a(Q(s'))
            next_action = jnp.argmax(online_q_next, axis=-1)
            # Q'(s', a')
            q_target_next = jnp.take_along_axis(target_q_next, next_action[:, None], axis=-1).squeeze()
            # target: r + gamma*Q'(s', a')
            targets = rewards + self.gamma * (1 - dones) * q_target_next
            
            return jnp.mean((targets - q_values) ** 2)

        grad_fn = jax.value_and_grad(loss)
        loss_value, gradients = grad_fn(self.params)
        updates, self.opt_state = self.optimizer.update(gradients, self.opt_state)
        self.params = optax.apply_updates(self.params, updates)
        self.losses.append(loss_value)
        self.grads.append(gradients)
        #self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)
        #print(self.epsilon)
        self.steps += 1


# %%
# RC model parameters
rc_params = [6.9789902e+03, 2.1591113e+04, 1.8807944e+05, 3.4490612e+00, 4.9556872e-01, 9.8289281e-02, 4.6257420e+00]
x0 = np.array([20, 35.8, 26.])
x_high = np.array([40., 80., 40.])
x_low = np.array([10., 10., 10.])
n_actions = 101
u_high = [0]
u_low = [-10.0] # -12

# load disturbances
file_path = os.path.abspath('')
parent_path = os.path.dirname(file_path)
data_path = os.path.join(file_path, 'disturbance_1min.csv')
data = pd.read_csv(data_path, index_col=[0])
# assign time index
t_base = 181*24*3600 # 7/1
n = len(data)
index = range(t_base, t_base + n*60, 60)
data.index = index

# sample
dt = 900
data = data.groupby([data.index // dt]).mean()
index_dt = range(t_base, t_base + len(data)*dt, dt)
data.index = index_dt 

# get disturbances for lssm
t_d = index_dt
disturbance_names = ['out_temp', 'qint_lump', 'qwin_lump', 'qradin_lump']
disturbance = data[disturbance_names].values


# %%
# random seed
seed = 0
np.random.seed(seed)

# Train the agent
import env
ts = 195*24*3600
ndays = 1
te = ndays*24*3600 + ts
weights = [100., 1., 0.] # for energy cost, dT, du
cop = 1.0

env = gym.make("R4C3Discrete-v0",
            rc_params = rc_params,
            x0 = x0,
            x_high = x_high,
            x_low = x_low,
            n_actions = n_actions,
            u_high = u_high,
            u_low = u_low,
            disturbances = (t_d, disturbance),
            cop = cop,
            ts = ts,
            te = te,
            dt = dt,
            weights = weights).env

state_size = env.observation_space.shape[0]
action_size = env.action_space.n
rng_key = jax.random.PRNGKey(42) 
lr = 1e-05
epsilon_decay = 0.98

agent = DDQNAgent(state_size, action_size, rng_key, lr=lr, epsilon_decay=epsilon_decay)

n_episodes = 500
reward_history = []
max_episode_steps=200 # env.spec.max_episode_steps
reward_threshold= -10 # env.spec.reward_threshold
solved_window = 20
 
# sync target q-network
steps_since_target_update = 0
target_update_freq = 5

# main loop
for episode in range(n_episodes):
    state, _ = env.reset(seed=0)
    state = jnp.array(state, dtype=jnp.float32)

    total_reward = 0
    done = False
    step_in_episode = 0
    
    while not done:
        action = agent.act(state)
        next_state, reward, done, _, _ = env.step(action)
        next_state = jnp.array(next_state, dtype=jnp.float32)

        agent.remember(state, action, reward, next_state, done)
        agent.replay()

        state = next_state
        total_reward += reward
        step_in_episode += 1

        # update  target q-network 
        steps_since_target_update += 1
        if steps_since_target_update >= target_update_freq:
            agent.sync_target()
            steps_since_target_update = 0
  
        # udpate epsiolon after episode
        if done:
            if agent.epsilon > agent.epsilon_end:
                agent.epsilon *= agent.epsilon_decay
    
    reward_history.append(total_reward)
    print(f"Episode {episode}, Total Reward: {total_reward}")

    # stop training if average reward reaches requirement
    # Calculate the average reward over the last 'solved_window' episodes
    if episode >= solved_window:
        avg_reward = np.mean(reward_history[-solved_window:])
        print(f'Episode: {episode}, Average Reward: {avg_reward}')

        if avg_reward >= reward_threshold:
            print(f"RC solved in {episode} episodes!")
            break

# %%
# save rewards
with open('./ddqn-rc-reward.json', 'w') as f:
    json.dump(reward_history, f)

# Plot the historical rewards
plt.figure()
plt.plot(reward_history)
plt.xlabel("Episode")
plt.ylabel("Total Reward")
plt.title("Historical Rewards for RC-v1")
plt.savefig('ddqn-rc-reward.png')


# %%
plt.figure()
plt.subplot(2,1,1)
plt.plot(agent.losses)
plt.grid()
plt.savefig('ddqn-rc-losses.png')

# %%
# plot training 
def plot_moving_average_reward(episode_rewards, window_size=20):
    cumsum_rewards = np.cumsum(episode_rewards)
    moving_avg_rewards = (cumsum_rewards[window_size:] - cumsum_rewards[:-window_size]) / window_size
    plt.figure()
    plt.plot(moving_avg_rewards)
    plt.xlabel('Episode')
    plt.ylabel('Moving Average Reward')
    plt.title('Moving Average Reward over Episodes')
    plt.savefig('ddqn-rc-ma-reward.png')

plot_moving_average_reward(reward_history)


# %%
# need a virtual display for rendering in docker
from pyvirtualdisplay import Display
display = Display(visible=0, size=(1400, 900))
display.start()
from IPython import display as ipythondisplay

# Test the trained agent

print("\nTesting the trained agent...")
state, _ = env.reset(seed=0)
state = jnp.array(state, dtype=jnp.float32)

total_reward = 0
done = False
pre_screen = env.render()
step_in_episode = 0
actions = []
Tz = [state[1]]
P = [state[6]]
To = [state[4]]

while not done:
    agent.epsilon = 0
    action = agent.act(state)
    next_state, reward, done, _, _ = env.step(action)
    next_state = jnp.array(next_state, dtype=jnp.float32)
    state = next_state
    total_reward += reward
    step_in_episode += 1

    # save for future use
    actions.append(action)
    Tz.append(next_state[1])
    P.append(next_state[6])
    To.append(next_state[4])
print(f"Total Reward: {total_reward}")

env.close()

# %%

n_days = step_in_episode*dt // (3600*24)
n_steps_per_hour = 3600 // dt
prices = env.energy_price
Tub = env.Tz_high
Tlb = env.Tz_low


# need make sure the length of Tub and Tlb is the same as Tz
prices = np.tile(prices.reshape(-1,1), (n_days, n_steps_per_hour))
Tub = np.tile(np.array(Tub).reshape(-1, 1), (n_days, n_steps_per_hour)) 
Tlb = np.tile(np.array(Tlb).reshape(-1, 1), (n_days, n_steps_per_hour)) 
# actions to actual cooling rate 
actions = [action/(n_actions-1)*(np.array(u_high) - np.array(u_low)) + np.array(u_low) for action in actions]

print(n_days)
plt.figure(figsize=(12,8))
plt.subplot(4,1,1)
plt.plot(prices.flatten(), label='Energy Price [$/kWh]')
plt.xticks([])
plt.legend() 

plt.subplot(4,1,2)
plt.plot(Tz, label='Zone Temperature [C]')
plt.plot(To, label="Outdoor")
plt.plot(Tub.flatten(), 'k--', label='Bounds')
plt.plot(Tlb.flatten(), 'k--')
plt.legend()

plt.subplot(4,1,3)
plt.plot(P, label="Power [kW]")
plt.xticks([])
plt.legend()

plt.subplot(4,1,4)
plt.plot(actions, label="HVAC Cooling Rate [kW]")
plt.legend()
plt.savefig('ddqn-rc.png')

# save actions for future use
with open('./ddqn-rc-actions.json', 'w') as f:
    json.dump([float(action) for action in actions], f)
