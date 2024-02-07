import random
from typing import Annotated, Optional

import gymnasium as gym
from tqdm import tqdm

import freshman.algorithms as algs
from freshman.algorithms import Algorithm, Parameters
import freshman.log
from freshman.env import Env, Policy, EnvMaker
import freshman.graphics as graphics
import arguably

# TODO add tests
# TODO Plot Sum of rewards


envs = {
    "lake": {"id": "FrozenLake-v1", "is_slippery": False},
    "lake_easy": {"id": "FrozenLake-v1", "desc": ["FFFSFFFG"], "is_slippery": False},
    "lake_slippery": {"id": "FrozenLake-v1", "is_slippery": True},
    "cliff": {"id": "CliffWalking-v0"},
}


# render_mode - “human”, “rgb_array”, “ansi”
def get_env(env: str, *, seed: Optional[int] = None) -> EnvMaker:
    return lambda render_mode: Env(
        gym.make(**envs[env], render_mode=render_mode), seed=seed
    )


def train(env_maker: EnvMaker, params: Parameters, alg: Algorithm) -> Policy:
    with env_maker("ansi") as env:
        p = alg(env, params)
        print(p)
        return p


def show_policy(env_maker: EnvMaker, policy: Policy):
    with env_maker("human") as env:
        for _ in policy.trajectory(env, limit=30):
            pass


def generate_video(env_maker: EnvMaker, policy: Policy, name: str):
    graphics.video(env_maker, policy, f"{name}.gif")


@arguably.command
def run(
    env: Annotated[str, arguably.arg.choices(*envs.keys())],
    alg: Annotated[str, arguably.arg.choices(*algs.algorithms.keys())],
    *,
    show: bool = False,
    video: bool = False,
):
    env_maker = get_env(env, seed=12346)  # 1234
    print("Training: ")
    policy: Policy = train(
        env_maker,
        algs.Parameters(num_episodes=1000, eps=0.2, gamma=0.9, progress=tqdm),
        algs.algorithms[alg],
    )
    if show:
        show_policy(env_maker, policy)
    if video:
        generate_video(env_maker, policy, f"{env}-{alg}")


def main():
    freshman.log.start()
    random.seed(13)
    arguably.run(name="freshman")


main()
