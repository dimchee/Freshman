import pprint
import gymnasium as gym
import rl


def arrowify(i: int) -> str | None:
    match i:
        case 0:
            return "← "
        case 1:
            return "↓ "
        case 2:
            return "→ "
        case 3:
            return "↑ "


def gridify(i: int) -> tuple[int, int]:
    return (i // 4, i % 4)


def readable(sa: dict[int, dict[int, float]]):
    return {
        s: {arrowify(a): f"{p:^6.3f}" for a, p in ap.items()} for s, ap in sa.items()
    }


log_file = open("log", "w")


# render_mode = “human”, “rgb_array”, “ansi”
def frozenLake(render_mode="ansi"):
    return gym.make("FrozenLake-v1", render_mode=render_mode)
    # return gym.make("FrozenLake-v1", desc=["FFFSFFG"], render_mode=render_mode)


def log(*args):
    pprint.pprint(args, stream=log_file)


env = frozenLake()

policy = rl.policy_iteration(
    env,
    num_episodes=10000,
    gamma=0.92,
    eps=0.01,
    # log_policy=lambda policy: log("policy", readable(policy)),
    # log_q=lambda q: log("q: ", readable(q)),
)

env = frozenLake("human")
last_s, info = env.reset()
for _ in range(35):
    a = rl.sample_policy(policy, last_s)
    s, r, terminated, truncated, info = env.step(a)
    last_s = s
    if terminated or truncated:
        break
print("finished")
log_file.close()
env.close()
