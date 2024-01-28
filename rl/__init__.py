import random
from typing import Any, Callable

import gymnasium as gym

# TODO add test
# TODO use Pipe https://pypi.org/project/pipe/
State = int
Action = int
Reward = float
Probability = float
Policy = dict[State, dict[Action, Probability]]
Trajectory = list[tuple[State, Action, Reward]]
Q = dict[State, dict[Action, Reward]]
Returns = dict[State, dict[Action, list[Reward]]]
Evaluation = Callable[[Policy], Q]
Improvement = Callable[[Q], Policy]


def numGymStates(env: gym.Env[Any, Any]):
    return gym.spaces.flatdim(env.observation_space)


def fromGymState(env: gym.Env[Any, Any], s) -> State:
    return gym.spaces.flatten(env.observation_space, s)  # type: ignore


def toGymState(env: gym.Env[Any, Any], s: State):
    return gym.spaces.unflatten(env.action_space, s)


def numGymActions(env: gym.Env[Any, Any]):
    return gym.spaces.flatdim(env.action_space)


def fromGymAction(env: gym.Env[Any, Any], a) -> Action:
    return gym.spaces.flatten(env.action_space, a)  # type: ignore


def toGymAction(env: gym.Env[Any, Any], a: Action):
    return gym.spaces.unflatten(env.action_space, a)


def sample_policy(policy: Policy, state: State) -> Action:
    return random.choices(list(policy[state].keys()), list(policy[state].values()))[0]


def generate_trajectory(
    env: gym.Env[Any, Any], policy: Policy, last_s: State
) -> Trajectory:
    trajectory: Trajectory = []
    for _ in range(100):
        a = sample_policy(policy, last_s)
        s, rw, terminated, truncated, _ = env.step(a)
        trajectory.append((last_s, a, rw.__float__()))
        last_s = s
        if terminated or truncated:
            break
    return trajectory


def tag_first_sa(trajectory: Trajectory) -> list[tuple[State, Action, Reward, bool]]:
    visited = set()
    return [
        (s, a, r, (s, a) not in visited and not visited.add((s, a)))
        for s, a, r in trajectory
    ]


def average(xs: list[float]) -> float:
    return sum(xs) / len(xs)


def deterministic(
    dist: dict[Action, Probability], action: Action | None, eps: float
) -> dict[Action, Probability]:
    nA = len(dist)
    if action is None:
        return {a: 1.0 / nA for a in dist.keys()}
    else:
        return {a: 1.0 - eps if action == a else eps / (nA - 1) for a in dist.keys()}


def argmax(expected: dict[Action, Reward]) -> Action:
    return max(expected, key=expected.get)  # type: ignore - pyright


def one_iteration(
    trajectory: Trajectory,
    policy: Policy,
    q: Q,
    returns: Returns,
    gamma: float,
    eps: float,
    log_policy: Callable[[Policy], None] = lambda _: None,
    log_q: Callable[[Q], None] = lambda _: None,
) -> None:
    G = 0.0
    for s, a, r, first in reversed(tag_first_sa(trajectory)):
        G = gamma * G + r
        if first:
            returns[s][a].append(G)
            q[s][a] = average(returns[s][a])
            log_q(q)
            policy[s] = deterministic(policy[s], argmax(q[s]) if q[s] else None, eps)
            log_policy(policy)


def policy_iteration(
    env: gym.Env[Any, Any],
    num_episodes=1000,
    gamma: float = 0.95,
    eps: float = 0.05,
    log_policy: Callable[[Policy], None] = lambda _: None,
    log_q: Callable[[Q], None] = lambda _: None,
) -> Policy:
    nA = numGymActions(env)
    nS = numGymStates(env)
    q: Q = {s: {} for s in range(nS)}
    log_q(q)
    policy: Policy = {s: {a: 1.0 / nA for a in range(nA)} for s in range(nS)}
    log_policy(policy)
    returns: Returns = {s: {a: [] for a in range(nA)} for s in range(nS)}
    for _ in range(num_episodes):
        last_s, _ = env.reset()
        trajectory = generate_trajectory(env, policy, last_s)
        # if any([r > 0 for _, _, r in trajectory]):
        #     print("YES")
        one_iteration(trajectory, policy, q, returns, gamma, eps)
        log_q(q)
        log_policy(policy)
    return policy
