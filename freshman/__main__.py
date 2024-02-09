import random
from typing import Annotated, Iterable, Optional

import arguably
import gymnasium as gym
from matplotlib.animation import PillowWriter
from tqdm import tqdm

import freshman.algorithms as algs
import freshman.graphics as graphics
import freshman.log
from freshman.algorithms import Algorithm, Parameters
from freshman.env import Action, Env, EnvMaker, Policy, QValue, Reward, State

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
    random.seed(seed)
    return lambda render_mode: Env(
        gym.make(**envs[env], render_mode=render_mode), seed=seed
    )


def train(env_maker: EnvMaker, params: Parameters, alg: Algorithm) -> Policy:
    with env_maker("ansi") as env:
        policy, _ = alg(env, params)
        return policy


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
        algs.Parameters(num_episodes=1000, eps=0.2, gamma=0.9, episodic_progress=tqdm),
        algs.algorithms[alg],
    )
    if show:
        show_policy(env_maker, policy)
    if video:
        generate_video(env_maker, policy, f"{env}-{alg}")


def main():
    freshman.log.start()
    arguably.run(name="freshman")


# main()


def collect_rewards():
    rws: list[Reward] = []

    def f(xs: Iterable[tuple[State, Action, Reward, State, Action]]):
        for x in xs:
            _, _, r, _, _ = x
            rws.append(r)
            # print(f"{r=}")
            yield x

    return f, rws


def visual_training(env: Env, q: QValue):
    rws: list[Reward] = []

    def f(xs: Iterable[tuple[State, Action, Reward, State, Action]]):
        for x in xs:
            _, _, r, _, _ = x
            rws.append(r)
            graphics.plot(env, q, show=True, rewards=rws)
            # print(f"{r=}")
            yield x

    return f


env_maker = get_env("lake", seed=1234)  # 1234
print("Training: ")
policy: Policy
q: QValue
with env_maker("rgb_array") as env:
    alg = algs.algorithms["q_learning"]
    q, policy = QValue(), Policy(eps=0.2)
    # t_progress, rws = collect_rewards()
    policy, q = alg(
        env,
        algs.Parameters(
            num_episodes=1000,
            eps=0.2,
            gamma=0.9,
            episodic_progress=tqdm,
            q=q,
            # progress=t_progress,
        ),
    )
    with graphics.TrainingVideo(name="training", format="mp4") as vid:
        policy, q = alg(
            env,
            algs.Parameters(
                num_episodes=100,
                eps=0.2,
                gamma=0.9,
                episodic_progress=tqdm,
                q=q,
                progress=vid.step(env, q),
            ),
        )
