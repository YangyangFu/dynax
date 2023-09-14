import jax
import jax.numpy as jnp 
import flax.linen as nn
from dynax.wrapper.envs.rc import RC 

class SimpleDNN(nn.Module):
    hidden_size: int = 32
    output_size: int = 2
    def setup(self):
        self.ann1 = nn.Dense(features=self.hidden_size)
        self.ann2 = nn.Dense(features=self.output_size)
    
    def __call__(self, x):
        x = self.ann1(x)
        x = nn.relu(x)
        x = self.ann2(x)
        return x
    
    def step(self, x, params):
        return self.apply(params, x)

ann = nn.jit(SimpleDNN)(hidden_size=16)
x = jax.random.normal(jax.random.PRNGKey(0), (10, 4))

params = ann.init(jax.random.PRNGKey(0), x)
y = ann.apply(params, x)
print(y)


class MyModel(nn.Module):
    num_neurons: int

    @nn.compact
    def __call__(self, x, training: bool):
        x = nn.Dense(self.num_neurons)(x)
        x = nn.Dropout(rate=0.5, deterministic=not training)(x)
        return x

def main():
    root_key = jax.random.PRNGKey(seed=0)
    main_key, params_key, dropout_key = jax.random.split(key=root_key, num=3)
    my_model = nn.jit(MyModel, static_argnums=(2,))(num_neurons=3)
    x = jnp.empty((3, 4, 4))
    variables = my_model.init(params_key, x, False)


main()


# jit on random modules
class Model2(nn.Module):
    num_neurons: int

    @nn.compact
    def __call__(self, x, training: bool):
        x = nn.Dense(self.num_neurons)(x)
        x = nn.Dropout(rate=0.5, deterministic=not training)(x)
        return x
    
root_key = jax.random.PRNGKey(seed=0)
main_key, params_key, dropout_key = jax.random.split(key=root_key, num=3)
model2 = nn.jit(Model2, static_argnums=(2,))(num_neurons=3)
x = jnp.empty((3, 4, 4))
# the following will not work
#variables = model2.init(params_key, x, training=False)
variables = model2.init(params_key, x, False)
print(variables)

# test jit on bool returns
def contains(x,  X):
    return bool((x >= X[0]) and (x <= X[1]))

x = 1
X = (0, 2)
print(contains(x, X))
#print(jax.jit(contains)(x, X))

#print(ssss)
ts = 0
te = 24*60*60
dt = 60

#env = nn.jit(RC)(start_time=ts, end_time=te, dt=dt, num_actions=11, name='RC-V1')
# TODO: not jittable yet
env = RC(start_time=ts, end_time=te, dt=dt, num_actions=11, name='RC-V1')

key = jax.random.PRNGKey(0)
t = ts 

obs, states, params = env.reset(key)
init_params = env.init(key, 0, states)
print(init_params)
print(params)

while t < te:
    key, key_reset = jax.random.split(key)
    action = env.action_space.sample(key)
    obs_next, reward, terminated, truncated, info, states = env.step(action, states=states, params=params)
    print(action, info)
    t += dt
