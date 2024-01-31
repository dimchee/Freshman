import itertools
import random
from typing import Callable, Iterable

import gymnasium as gym
from dataclasses import dataclass

import freshman.log
from freshman.env import (
    Env,
    Policy,
    QValue,
    Trajectory,
    greedy,
    tabular,
)


Progress = Callable[[Iterable[int]], Iterable[int]]


@dataclass(kw_only=True)
class Parameters:
    num_episodes: int = 200
    gamma: float = 0.9
    eps: float = 0.1
    alpha: float = 0.9
    progress: Progress = lambda x: x  # noqa: E731


algorithms = {}


def algorithm(alg: Callable[[Env, Parameters], Policy]):
    algorithms[alg.__name__] = alg
    return alg


def tag_first_sa(trajectory: Trajectory):
    then = lambda _: True  # noqa: E731
    visited = set()
    return [
        (s, a, r, (s, a) not in visited and then(visited.add((s, a))))
        for s, a, r, _, _ in trajectory
    ]


# On-policy first-visit Monte Carlo control (for eps-soft policies)
@algorithm
def on_policy_mc_slow(env: Env, opts: Parameters) -> Policy:
    q, policy = QValue(), Policy(eps=opts.eps)
    returns = tabular([])
    for _ in opts.progress(range(opts.num_episodes)):
        ts = policy.trajectory(env)
        G = 0.0
        for s, a, r, first in reversed(tag_first_sa(ts)):
            G = opts.gamma * G + r
            if first:
                returns[s][a].append(G)
                q[s][a] = sum(returns[s][a]) / len(returns[s][a])
                policy[s] = greedy(q[s])
    return policy


# On-policy first-visit Monte Carlo control (for eps-soft policies)
@algorithm
def on_policy_mc(env: Env, opts: Parameters) -> Policy:
    q, policy = QValue(), Policy(eps=opts.eps)
    returns = tabular((0.0, int(0)))
    for _ in opts.progress(range(opts.num_episodes)):
        ts = policy.trajectory(env)
        G = 0.0
        for s, a, r, first in reversed(tag_first_sa(ts)):
            G = opts.gamma * G + r
            if first:
                avg, nr = returns[s][a]
                avg = G / (nr + 1) + avg * nr / (nr + 1)
                returns[s][a] = (avg, nr + 1)
                q[s][a] = sum(returns[s][a]) / len(returns[s][a])
                policy[s] = greedy(q[s])
    return policy


# Off-policy Monte Carlo control
@algorithm
def off_policy_mc(env: Env, opts: Parameters) -> Policy:
    q, policy = QValue(), Policy()
    c = tabular(0.0)
    for _ in opts.progress(range(opts.num_episodes)):
        b = Policy(eps=opts.eps)
        ts = b.trajectory(env)
        G = 0.0
        W = 1.0
        for s, a, r, _, _ in reversed(list(ts)):
            G = opts.gamma * G + r
            c[s][a] = c[s][a] + W
            q[s][a] = q[s][a] + W / c[s][a] * (G - q[s][a])
            policy[s] = greedy(q[s])
            if a != policy[s]:
                break
            W = W / b[s][a]
    return policy


# On-policy every-visit Monte Carlo control with constant alpha
@algorithm
def constant_alpha_mc(env: Env, opts: Parameters) -> Policy:
    q, policy = QValue(), Policy(eps=opts.eps)
    for _ in opts.progress(range(opts.num_episodes)):
        ts = policy.trajectory(env)
        G = 0.0
        for s, a, r, ss, _ in reversed(list(ts)):
            G = opts.gamma * G + r
            q[s][a] += opts.alpha * (G - sum(q[ss].values()))
            policy[s] = greedy(q[s])
    return policy.deterministic


# Sarsa - On-policy Temporal Difference control
@algorithm
def sarsa(env: Env, opts: Parameters) -> Policy:
    q, policy = QValue(), Policy(eps=opts.eps)
    for _ in opts.progress(range(opts.num_episodes)):
        for s, a, r, ss, aa in policy.trajectory(env):
            q[s][a] += opts.alpha * (r + opts.gamma * q[ss][aa] - q[s][a])
            policy[s] = greedy(q[s])
    return policy.deterministic


# Q-learning - Off-policy Temporal Difference control
@algorithm
def q_learning(env: Env, opts: Parameters) -> Policy:
    q, policy = QValue(), Policy(eps=opts.eps)
    for _ in opts.progress(range(opts.num_episodes)):
        for s, a, r, ss, _ in policy.trajectory(env):
            # print(f"{s=} {a=} {r=} {ss=}")
            update_step = r + opts.gamma * max(q[ss].values(), default=0) - q[s][a]
            q[s][a] += opts.alpha * update_step
            policy[s] = greedy(q[s])
        # freshman.log.print_table("q: ", q)
        # freshman.log.print_table("policy: ", policy)
    return policy.deterministic


# Expected Sarsa - On-policy Temporal Difference control
@algorithm
def expected_sarsa(env: Env, opts: Parameters) -> Policy:
    q, policy = QValue(), Policy(eps=opts.eps)
    nA = gym.spaces.flatdim(env.env.action_space)
    for _ in opts.progress(range(opts.num_episodes)):
        for s, a, r, _, _ in policy.trajectory(env):
            EG = sum((policy[s][a] + policy.eps / nA) * q[s][a] for a in range(nA))
            q[s][a] += opts.alpha * (r + opts.gamma * EG - q[s][a])
            policy[s] = greedy(q[s])
    return policy.deterministic


# Double Q-learning - Off-policy Temporal Difference control
@algorithm
def double_q_learning(env: Env, opts: Parameters) -> Policy:
    def update_step(q, p, s, a, r, ss):
        return r + opts.gamma * max(p[ss].values(), default=0) - q[s][a]

    q1, q2, policy = QValue(), QValue(), Policy(eps=opts.eps)
    for _ in opts.progress(range(opts.num_episodes)):
        for s, a, r, ss, _ in policy.trajectory(env):
            if random.random() < 0.5:
                q1[s][a] += opts.alpha * update_step(q1, q2, s, a, r, ss)
            else:
                q2[s][a] += opts.alpha * update_step(q2, q1, s, a, r, ss)
            policy[s] = greedy(
                {a: (q1[s][a] + q2[s][a]) / 2 for a in itertools.chain(q1[s], q2[s])}
            )
    return policy.deterministic
