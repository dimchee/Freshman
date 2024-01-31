import random
from typing import Callable, Optional, Iterable
import sys

# from gymnasium.wrappers.record_video import RecordVideo
import gymnasium as gym
import matplotlib.animation as ani
import matplotlib.pyplot as plt
from tqdm import tqdm

import freshman.algorithms as algs
import freshman.log
from freshman.env import Env, Policy

# TODO add tests
# TODO Plot Sum of rewards


envs = {
    "lake": {"id": "FrozenLake-v1", "is_slippery": False},
    "lake_easy": {"id": "FrozenLake-v1", "desc": ["FFFSFFFG"], "is_slippery": False},
    "lake_slippery": {"id": "FrozenLake-v1", "is_slippery": True},
    "cliff": {"id": "CliffWalking-v0"},
}


EnvMaker = Callable[[str], Env]


# render_mode - “human”, “rgb_array”, “ansi”
def get_env(env: str, *, seed: Optional[int] = None) -> EnvMaker:
    return lambda render_mode: Env(
        gym.make(**envs[env], render_mode=render_mode), seed=seed
    )


# Needs ffmpeg
def video(
    em: EnvMaker,
    policy: Policy,
    file_name: str,
    *,
    fps: int = 1,
    title: str = "",
    limit: int = 30,
    dpi: int = 100,
):
    with em("rgb_array") as env:
        _, first = env.reset(), env.env.render()
        height, width, _ = first.shape  # type: ignore
        fig, ax = plt.subplots(figsize=(width / dpi, height / dpi))
        # no extra white space
        fig.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=None, hspace=None)
        writer = ani.writers["ffmpeg"](fps=fps, metadata={"title": title})
        with writer.saving(fig, "docs/video/" + file_name, dpi=dpi):
            im = ax.imshow(first, interpolation="nearest")
            writer.grab_frame()
            for img in (env.env.render() for _ in policy.trajectory(env, limit=limit)):
                im.set_data(img)
                writer.grab_frame()


def run(env_name: str, alg_name: str):
    env_maker = get_env(env_name, seed=1234)
    print("Training: ")
    with env_maker("ansi") as env:
        policy: Policy = algs.algorithms[alg_name](
            env,
            algs.Parameters(
                num_episodes=1000,
                eps=0.3,
                gamma=0.8,
                progress=tqdm,  # video_progress(env),
            ),
        )
    video(env_maker, policy, f"{env_name}-{alg_name}.mp4")
    # print("Policy: ", freshman.log.pretty(policy))
    # with Env(env_maker("human"), seed=4321) as env:
    #     for _ in policy.trajectory(env, limit=30):
    #         pass
    print("Finished")


def padded(strs: Iterable[str], padding: int):
    return ("\n" + " " * padding).join("`" + s + "`" for s in strs)


usage = f"""Requires exactly 2 arguments: <env> <alg>
    <env> is one of {padded(envs.keys(), 20)}
    <alg> is one of {padded(algs.algorithms.keys(), 20)}"""


def main(args: list[str]):
    freshman.log.start()
    random.seed(13)
    match args:
        case [env, alg]:
            run(env, alg)
        case _:
            print(usage)


main(sys.argv[1:])
