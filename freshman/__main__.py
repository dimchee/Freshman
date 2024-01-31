import random
import sys
from typing import Any, Callable

# from gymnasium.wrappers.record_video import RecordVideo
import gymnasium as gym
from tqdm import tqdm

import freshman.algorithms as algs
import freshman.log
from freshman.env import Env, Policy

# TODO add tests
# TODO Plot Sum of rewards


envs = {
    "lake": {"id": "FrozenLake-v1", "is_slippery": False},
    "lake_easy": {"id": "FrozenLake-v1", "desc": ["FFFSFFFG"], "is_slippery": False},
    "slippery_lake": {"id": "FrozenLake-v1", "is_slippery": True},
    "cliff": {"id": "CliffWalking-v0"},
}


# render_mode - “human”, “rgb_array”, “ansi”
def get_env(env: str) -> Callable[[str], gym.Env[Any, Any]]:
    return lambda render_mode: gym.make(**envs[env], render_mode=render_mode)


# def video(env):
#     return RecordVideo(
#         env=env,
#         video_folder="./docs/gifs",
#         name_prefix="video",
#         episode_trigger=lambda x: x % 100 == 1,
#         disable_logger=True,
#     )


def main(args: list[str]):
    freshman.log.start()
    random.seed(13)
    match args:
        case [env, alg]:
            gym_env = get_env(env)
            print("Training: ")
            with Env(gym_env("ansi"), seed=1234) as env:
                policy: Policy = algs.algorithms[alg](
                    env,
                    algs.Parameters(
                        num_episodes=1000,
                        eps=0.3,
                        gamma=0.8,
                        progress=tqdm,  # video_progress(env),
                    ),
                )
            # print("Policy: ", freshman.log.pretty(policy))
            with Env(gym_env("human"), seed=4321) as env:
                for _ in policy.trajectory(env, limit=30):
                    pass
            print("Finished")
        case _:
            print("Requires exactly 2 arguments: <env> <alg>")
            print(f"    <env> is one of {' '.join('`' + e + '`' for e in envs)}")
            print(
                "    <alg> is one of",
                ("\n" + " " * 20).join("`" + a + "`" for a in algs.algorithms),
            )


main(sys.argv[1:])
