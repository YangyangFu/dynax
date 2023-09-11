from .registeration import (
    make,
    register
)

register(
    id="CartPole-v0",
    entry_point="dynax.wrapper.envs.cartpole:CartPole",
    max_episode_steps=200,
    reward_threshold=195.0,
)