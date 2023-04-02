#%% docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/dqn/#dqn_jaxpy
import argparse
import os
import random
import time
from distutils.util import strtobool

import flax
import flax.linen as nn
import gymnasium as gym
import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax.training.train_state import TrainState
from stable_baselines3.common.buffers import ReplayBuffer
from torch.utils.tensorboard import SummaryWriter
import pandas as pd
from typing import Callable, List, Tuple, Union

#%%
def parse_args():
    # fmt: off
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp-name", type=str, default=os.path.basename(__file__).rstrip(".py"),
        help="the name of this experiment")
    parser.add_argument("--seed", type=int, default=1,
        help="seed of the experiment")
    parser.add_argument("--track", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
        help="if toggled, this experiment will be tracked with Weights and Biases")
    parser.add_argument("--wandb-project-name", type=str, default="cleanRL",
        help="the wandb's project name")
    parser.add_argument("--wandb-entity", type=str, default=None,
        help="the entity (team) of wandb's project")
    parser.add_argument("--capture-video", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
        help="whether to capture videos of the agent performances (check out `videos` folder)")
    parser.add_argument("--save-model", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
        help="whether to save model into the `runs/{run_name}` folder")
    parser.add_argument("--upload-model", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
        help="whether to upload the saved model to huggingface")
    parser.add_argument("--hf-entity", type=str, default="",
        help="the user or org name of the model repository from the Hugging Face Hub")

    # Algorithm specific arguments
    parser.add_argument("--env-id", type=str, default="CartPole-v1",
        help="the id of the environment")
    parser.add_argument("--total-timesteps", type=int, default=200000,
        help="total timesteps of the experiments")
    parser.add_argument("--learning-rate", type=float, default=2.5e-4,
        help="the learning rate of the optimizer")
    parser.add_argument("--buffer-size", type=int, default=20000,
        help="the replay memory buffer size")
    parser.add_argument("--gamma", type=float, default=0.99,
        help="the discount factor gamma")
    parser.add_argument("--tau", type=float, default=1.,
        help="the target network update rate")
    parser.add_argument("--target-network-frequency", type=int, default=500,
        help="the timesteps it takes to update the target network")
    parser.add_argument("--batch-size", type=int, default=128,
        help="the batch size of sample from the reply memory")
    parser.add_argument("--start-e", type=float, default=1,
        help="the starting epsilon for exploration")
    parser.add_argument("--end-e", type=float, default=0.05,
        help="the ending epsilon for exploration")
    parser.add_argument("--exploration-fraction", type=float, default=0.5,
        help="the fraction of `total-timesteps` it takes from start-e to go end-e")
    parser.add_argument("--learning-starts", type=int, default=10000,
        help="timestep to start learning")
    parser.add_argument("--train-frequency", type=int, default=10,
        help="the frequency of training")

    # Dyna-Q specific arguments
    parser.add_argument("--planning-steps", type=int, default=10,
        help="the number of planning steps")
    parser.add_argument("--planning-frequency", type=int, default=10,
        help="the frequency of planning")

    args = parser.parse_args()
    # fmt: on
    return args

#%% make rc envrionemnt 
import env

# Initialize env model
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
data_path = os.path.join(parent_path, 'data/disturbance_1min.csv')
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

ts = 195*24*3600
ndays = 1
te = ndays*24*3600 + ts
weights = [100., 1., 0.] # for energy cost, dT, du
cop = 1.0

env_rc = gym.make("R4C3Discrete-v0",
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
            weights = weights)

#%%
def make_env(env_rc, seed, idx, capture_video, run_name):
    def thunk():
        # this wrapper keeps recording cumulative reward and episode length
        # and at the end of episode, it will log them to "info"
        env = gym.wrappers.RecordEpisodeStatistics(env_rc)
        if capture_video:
            if idx == 0:
                env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
        # TODO: check how to make seed to env
        #env.seed(seed)
        env.action_space.seed(seed)
        env.observation_space.seed(seed)
        return env

    return thunk

#%%
# ALGO LOGIC: initialize agent here:
class QNetwork(nn.Module):
    action_dim: int

    @nn.compact
    def __call__(self, x: jnp.ndarray):
        x = nn.Dense(120)(x)
        x = nn.relu(x)
        x = nn.Dense(84)(x)
        x = nn.relu(x)
        x = nn.Dense(self.action_dim)(x)
        return x

# ALGO LOGIC: initialize env model here:
class LSSM(nn.Module):
    state_dim: int 
    action_dim: int
    disturbancce_dim: int
    output_dim: int
    dt: float 

    initializer: Callable = nn.initializers.lecun_normal()
    @nn.compact
    def __call__(self, x, u, d):
      """
      x: (b, state_dim)
      u: (b, action_dim)
      d: (b, disturbance_dim)
      """
      A = self.param('A',
                      self.initializer, # initialization
                      (self.state_dim, self.state_dim)) # shape
      Bu = self.param('Bu',
                      self.initializer, # initialization
                      (self.state_dim, self.action_dim))
      Bd = self.param('Bd',
                      self.initializer,
                      (self.state_dim, self.disturbancce_dim))

      C = self.param('C',
                      self.initializer,
                      (self.output_dim, self.state_dim))
      D = self.param("D",
                      self.initializer,
                      (self.output_dim, self.action_dim))
      xdot = jnp.dot(x, A.T) + jnp.dot(u, Bu.T) + jnp.dot(d, Bd.T)

      x_next = xdot * self.dt + x
      y_next = jnp.dot(x_next, C.T) + jnp.dot(u, D.T)
      
      return xdot, x_next, y_next

#%%
class EnvModelTrainState(TrainState):
    params: flax.core.FrozenDict

class TrainState(TrainState):
    target_params: flax.core.FrozenDict

#%%
#%%
def linear_schedule(start_e: float, end_e: float, duration: int, t: int):
    slope = (end_e - start_e) / duration
    return max(slope * t + start_e, end_e)

#%%
if __name__ == "__main__":
    args = parse_args()
    run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}"
    if args.track:
        # use wandb to track the progress
        pass
    
    writer = SummaryWriter(f"runs/{run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )

    # TRY NOT TO MODIFY: seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    key = jax.random.PRNGKey(args.seed)
    key, q_key = jax.random.split(key, 2)
    q_key, m_key = jax.random.split(q_key, 2)

    # env setup
    envs = gym.vector.SyncVectorEnv([make_env(env_rc, args.seed, 0, args.capture_video, run_name)])
    assert isinstance(envs.single_action_space, gym.spaces.Discrete), "only discrete action space is supported"
    assert isinstance(envs.single_observation_space, gym.spaces.Box), "only box space is supported"
    obs, _ = envs.reset()

    q_network = QNetwork(action_dim=envs.single_action_space.n)

    q_state = TrainState.create(
        apply_fn=q_network.apply,
        params=q_network.init(q_key, obs),
        target_params=q_network.init(q_key, obs),
        tx=optax.adam(learning_rate=args.learning_rate),
    )

    q_network.apply = jax.jit(q_network.apply)
    # This step is not necessary as init called on same observation and key will always lead to same initializations
    #q_state = q_state.replace(target_params=optax.incremental_update(q_state.params, q_state.target_params, 1))

    # model: assumed perfect model here
    # say we have an extact model of the system
    A = np.array([[-1.97402551e-04,  0.00000000e+00,  1.92610415e-04],
                [0.00000000e+00, -3.30784418e-06,  2.00281620e-06],
                [1.51360733e-06,  7.56564853e-07, -2.27017218e-06]])
    Bu = np.array([[9.63095932e-05], [0.], [0.]])
    Bd = np.array([[4.79213586e-06, 9.63095932e-05, 0.00000000e+00, 0.00000000e+00],
                [1.30502798e-06, 0.00000000e+00, 2.00353963e-06, 0.00000000e+00],
                [0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 7.56838127e-07]])
    C = np.array([[1., 0., 0.],
                [0., 0., 0.]])
    D = np.array([[0.],
                [1.]])

    ABCD = {'A': A, 'Bu': Bu, 'Bd': Bd, 'C': C, 'D': D}
    exact_model_params = {"params": ABCD}

    # TODO: update to learn env model over time
    env_model = LSSM(state_dim=3, action_dim=1, disturbancce_dim=4, output_dim=2, dt=dt)
    env_state = EnvModelTrainState.create(
        apply_fn=env_model.apply,
        params=exact_model_params, #env_model.init(m_key, jnp.ones((1, 3)), jnp.ones((1,1)), jnp.ones((1,4))), # state + action
        tx=optax.adam(learning_rate=args.learning_rate),
    )
    env_model.apply = jax.jit(env_model.apply)

    # timeout_termination seems related to infinite horizon tasks
    # we ignore this for now
    rb = ReplayBuffer(
        args.buffer_size,
        envs.single_observation_space,
        envs.single_action_space,
        "cpu",
        handle_timeout_termination=False, # change this may have effects 
    )

    @jax.jit
    def update(q_state, observations, actions, next_observations, rewards, dones):
        q_next_target = q_network.apply(q_state.target_params, next_observations)  # (batch_size, num_actions)
        q_next_target = jnp.max(q_next_target, axis=-1)  # (batch_size,)
        next_q_value = rewards + (1 - dones) * args.gamma * q_next_target

        def mse_loss(params):
            q_pred = q_network.apply(params, observations)  # (batch_size, num_actions)
            q_pred = q_pred[np.arange(q_pred.shape[0]), actions.squeeze()]  # (batch_size,)
            return ((q_pred - next_q_value) ** 2).mean(), q_pred

        (loss_value, q_pred), grads = jax.value_and_grad(mse_loss, has_aux=True)(q_state.params)
        q_state = q_state.apply_gradients(grads=grads)
        return loss_value, q_pred, q_state

    @jax.jit
    def update_model(env_state, observations, actions, next_observations, rewards, dones):
        #obs_actions = jnp.concatenate([observations, actions], axis = 1)

        def mse_loss(params):
            pred_next_obs = env_model.apply(params, observations)
            #pred_next_obs = pred[:, :state_dim]
            #pred_reward = pred[:, state_dim]
            loss_next_obs = ((pred_next_obs - next_observations) ** 2).mean()
            #loss_reward = ((pred_reward - rewards) ** 2).mean()
            loss_reward = 0
            return loss_next_obs + loss_reward, pred_next_obs
        
        (loss_value, next_obs_pred), grads = jax.value_and_grad(mse_loss, has_aux = True)(env_state.params)
        env_state = env_state.apply_gradients(grads=grads)
        return loss_value, next_obs_pred, env_state
    
    # this is now not jittable becasue 
    def get_objective_batch(h, dt, Tz, P, Tz_ub, Tz_lb, price, weights):
        h = h.astype(np.int32)
        
        price_batch = np.array([price[i] for i in h])
        cost_batch = price_batch*P*dt/3600.
        Tz_high_batch = np.array([Tz_ub[i] for i in h])
        Tz_low_batch = np.array([Tz_lb[i] for i in h])

        dTz = np.fmax(Tz - Tz_high_batch, Tz_low_batch - Tz)
        dTz = np.fmax(dTz, 0)

        du = 0

        return -(weights[0]*cost_batch + weights[1]*dTz + weights[2]*du)

    start_time = time.time()

    # TRY NOT TO MODIFY: start the game
    obs, _ = envs.reset(seed = args.seed)
    for global_step in range(args.total_timesteps):
    #for episode in range(num_episodes):
        # ALGO LOGIC: put action logic here
        epsilon = linear_schedule(args.start_e, args.end_e, args.exploration_fraction * args.total_timesteps, global_step)
        if random.random() < epsilon:
            actions = np.array([envs.single_action_space.sample() for _ in range(envs.num_envs)])
        else:
            q_values = q_network.apply(q_state.params, obs)
            actions = q_values.argmax(axis=-1)
            actions = jax.device_get(actions)

        # TRY NOT TO MODIFY: execute the game and log data.
        next_obs, rewards, dones, truncated, infos= envs.step(actions)

        # TRY NOT TO MODIFY: record rewards for plotting purposes
        if "final_info" in infos.keys():
            info = infos["final_info"][0]
            print(f"global_step={global_step}, episodic_return={info['episode']['r']}")
            writer.add_scalar("charts/episodic_return", info["episode"]["r"], global_step)
            writer.add_scalar("charts/episodic_length", info["episode"]["l"], global_step)
            writer.add_scalar("charts/epsilon", epsilon, global_step)

        # TRY NOT TO MODIFY: save data to reply buffer; handle `terminal_observation`
        # INTERESTING: why the next_obs is not the terminal observation???
        # This question then is how the infos["final_observation"] is obtained?
        real_next_obs = next_obs.copy()
        #
        if dones[0]:
            # why these two is not the same at the begining?
            real_next_obs[0] = infos["final_observation"][0]
        rb.add(obs, real_next_obs, actions, rewards, dones, infos)

        # TRY NOT TO MODIFY: CRUCIAL step easy to overlook
        obs = next_obs

        # ALGO LOGIC: update model
        #if global_step >= args.batch_size:
        #    data = rb.sample(args.batch_size)

            # perform a gradient-descent step
        #    loss_env, next_obs_pred, env_state = update_model(
        #        env_state,
        #        data.observations.numpy(),
        #        data.actions.numpy(),
        #        data.next_observations.numpy(),
        #        data.rewards.flatten().numpy(),
        #        data.dones.flatten().numpy(),
        #    )

        #    if global_step % 100 == 0:
        #        writer.add_scalar("losses/model_loss", jax.device_get(loss_env), global_step)

        # ALGO LOGIC: training control agent
        if global_step > args.learning_starts:
            if global_step % args.train_frequency == 0:
                data = rb.sample(args.batch_size)

                # perform a gradient-descent step
                loss, old_val, q_state = update(
                    q_state,
                    data.observations.numpy(),
                    data.actions.numpy(),
                    data.next_observations.numpy(),
                    data.rewards.flatten().numpy(),
                    data.dones.flatten().numpy(),
                )

                if global_step % 100 == 0:
                    writer.add_scalar("losses/td_loss", jax.device_get(loss), global_step)
                    writer.add_scalar("losses/q_values", jax.device_get(old_val).mean(), global_step)
                    print("SPS:", int(global_step / (time.time() - start_time)))
                    writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)


                # perform planning
                if global_step % args.planning_frequency == 0:
                    for _ in range(args.planning_steps):
                        # sample from replay buffer
                        # TODO: how is this seed-controlled?
                        data = rb.sample(args.batch_size)

                        # sample actions from action space
                        actions = np.array([[envs.single_action_space.sample() for _ in range(envs.num_envs)] for _ in range(args.batch_size)])
                        controls = np.array([action/(envs.single_action_space.n-1)*(envs.envs[0].u_high - envs.envs[0].u_low) + envs.envs[0].u_low for action in actions])

                        # observation predictions
                        observations = data.observations.numpy()
                        model_states = observations[:, 1:4]
                        disturbances = observations[:, 4:8]
                        _, x_pred, y_pred = env_model.apply(env_state.params, model_states, controls, disturbances)
                        Tz_pred = y_pred[:, 0]
                        P_pred = -y_pred[:, 1]
                        next_observations = data.next_observations.numpy()
                        next_observations_pred = next_observations.copy()
                        next_observations_pred[:,1:4] = x_pred
                        next_observations_pred[:, 6] = P_pred
                        # reward predictions
                        hb = observations[:, 0]
                        reward_pred = get_objective_batch(hb, envs.envs[0].dt, Tz_pred, P_pred, envs.envs[0].Tz_high, envs.envs[0].Tz_low, envs.envs[0].energy_price, envs.envs[0].weights)
                        # assume done is false
                        dones = jnp.full(reward_pred.shape, False, dtype=jnp.float32)

                        # update q network
                        loss, old_val, q_state = update(
                            q_state,
                            observations,
                            actions,
                            next_observations_pred,
                            reward_pred,
                            dones,
                        )

            # update target network
            if global_step % args.target_network_frequency == 0:
                q_state = q_state.replace(
                    target_params=optax.incremental_update(q_state.params, q_state.target_params, args.tau)
                )

    # save model if needed
    if args.save_model:
        pass 

    envs.close()
    writer.close()
