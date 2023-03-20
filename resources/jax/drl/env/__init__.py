from gymnasium.envs.registration import register

register(
    id='DiscreteLinearStateSpace-v0',
    entry_point='env.r4c3_discrete:DiscreteLinearStateSpaceEnv',
    max_episode_steps=196
)

register(
    id='R4C3Discrete-v0',
    entry_point='env.r4c3_discrete:R4C3DiscreteEnv',
    max_episode_steps=196
)