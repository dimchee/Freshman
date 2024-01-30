from typing import Any

import gymnasium as gym

import rl.algorithms
import rl.log
from rl.env import Env, Policy

# TODO add tests
# TODO Plot Sum of rewards


# render_mode = “human”, “rgb_array”, “ansi”
def frozenLake(render_mode="ansi") -> gym.Env[Any, Any]:
    # return gym.make("FrozenLake-v1", render_mode=render_mode)
    return gym.make(
        "FrozenLake-v1", desc=["FFFSFFG"], render_mode=render_mode, is_slippery=False
    )


def cliffWalking(render_mode="ansi") -> gym.Env[Any, Any]:
    return gym.make("CliffWalking-v0", render_mode=render_mode)


rl.log.start()

# with Env(frozenLake()) as env:
#     policy = Policy(eps=0.2, dict={0: {3: 1}})
#     for _ in range(10):
#         print(policy.sample(env, 0))

policy: Policy
with Env(cliffWalking()) as env:
    # with Env(frozenLake()) as env:
    policy = rl.algorithms.expected_sarsa(env, num_episodes=1000)
print("Trained")
rl.log.print_table("policy: ", policy)
with Env(cliffWalking("human")) as env:
    # with Env(frozenLake("human")) as env:
    last_s = env.reset()
    for _ in range(30):
        a = policy.sample(env, last_s)
        s, r, terminated, truncated, info = env.step(a)
        last_s = s
        if terminated or truncated:
            break
print("Finished")
