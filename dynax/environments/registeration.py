from . import BuildingRC

def make(env_id, **env_kwargs):
    if env_id not in registered_envs:
        raise ValueError(f"{env_id} is not in registered gymnax environments in Dynax. If you want to use Gymnax environments, you should use gymnax.make({env_id})")

    if env_id == "BuildingRC-v1":
        env = BuildingRC(**env_kwargs)

    return env, env.default_params

registered_envs=[
    "BuildingRC-v1",
]