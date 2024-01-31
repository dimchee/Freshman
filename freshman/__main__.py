from typing import Any, Callable

import gymnasium as gym
import sys

import freshman.algorithms as algs
import freshman.log
from freshman.env import Env, Policy

# TODO add tests
# TODO Plot Sum of rewards


# render_mode = “human”, “rgb_array”, “ansi”


def get_env(env: str) -> Callable[[str], gym.Env[Any, Any]]:
    match env:
        case "lake":
            return lambda render_mode: gym.make(
                "FrozenLake-v1", render_mode=render_mode, is_slippery=False
            )
        case "cliff":
            return lambda render_mode: gym.make(
                "CliffWalking-v0", render_mode=render_mode
            )
        case _:
            raise ValueError("env is not recognised")


def get_alg(alg: str):
    match alg:
        case "q_learning":
            return algs.q_learning
        case "double_q_learning":
            return algs.double_q_learning
        case "sarsa":
            return algs.sarsa
        case "expected_sarsa":
            return algs.expected_sarsa
        case _:
            raise ValueError("alg is not recognised")


def main(args: list[str]):
    freshman.log.start()
    print(args)
    match args:
        case [env, alg]:
            gym_env = get_env(env)
            with Env(gym_env("ansi")) as env:
                policy: Policy = get_alg(alg)(
                    env, algs.Parameters(num_episodes=1000, eps=0.3, gamma=0.8)
                )
            print("Trained")
            print(freshman.log.pretty(policy))
            with Env(gym_env("human")) as env:
                for _ in policy.trajectory(env, limit=20):
                    pass
            print("Finished")
        case _:
            print("Requires exactly 2 arguments: <env> <alg>")
            print("    <env> is one of `lake`, `cliff`")
            print(
                "    <alg> is one of",
                "`q_learning`,",
                "`double_q_learning`,",
                "`sarsa`,",
                "`expected_sarsa`",
            )
            return


main(sys.argv[1:])
