import itertools
import random

import gymnasium as gym

# import rl.log
from rl.env import Env, Policy, QValue, Trajectory, greedy, tabular


def tag_first_sa(trajectory: Trajectory):
    then = lambda _: True  # noqa: E731
    visited = set()
    return [
        (s, a, r, (s, a) not in visited and then(visited.add((s, a))))
        for s, a, r, _, _ in trajectory
    ]


# On-policy first-visit Monte Carlo control (for eps-soft policies)
def on_policy_mc_slow(env: Env, num_episodes=1000, gamma=0.95, eps=0.05) -> Policy:
    q, policy = QValue(), Policy(eps=eps)
    returns = tabular([])
    for _ in range(num_episodes):
        ts = policy.trajectory(env, env.reset())
        G = 0.0
        for s, a, r, first in reversed(tag_first_sa(ts)):
            G = gamma * G + r
            if first:
                returns[s][a].append(G)
                q[s][a] = sum(returns[s][a]) / len(returns[s][a])
                policy[s] = greedy(q[s])
        # rl.log.print("q: ", q)
    return policy


# On-policy first-visit Monte Carlo control (for eps-soft policies)
def on_policy_mc(env: Env, num_episodes=1000, gamma=0.95, eps=0.05) -> Policy:
    q, policy = QValue(), Policy(eps=eps)
    returns = tabular((0.0, int(0)))
    for _ in range(num_episodes):
        ts = policy.trajectory(env, env.reset())
        G = 0.0
        for s, a, r, first in reversed(tag_first_sa(ts)):
            G = gamma * G + r
            if first:
                avg, nr = returns[s][a]
                avg = G / (nr + 1) + avg * nr / (nr + 1)
                returns[s][a] = (avg, nr + 1)
                q[s][a] = sum(returns[s][a]) / len(returns[s][a])
                policy[s] = greedy(q[s])
    return policy


# Off-policy Monte Carlo control
def off_policy_mc(env: Env, num_episodes=1000, gamma=0.95, eps=0.05):
    q, policy = QValue(), Policy()
    c = tabular(0.0)
    for _ in range(num_episodes):
        b = Policy(eps=eps)
        ts = b.trajectory(env, env.reset())
        G = 0.0
        W = 1.0
        for s, a, r, _, _ in reversed(list(ts)):
            G = gamma * G + r
            c[s][a] = c[s][a] + W
            q[s][a] = q[s][a] + W / c[s][a] * (G - q[s][a])
            policy[s] = greedy(q[s])
            if a != policy[s]:
                break
            W = W / b[s][a]
    return policy


# On-policy every-visit Monte Carlo control with constant alpha
def constant_alpha_mc(env: Env, num_episodes=1000, gamma=0.95, eps=0.05, alpha=0.9):
    q, policy = QValue(), Policy(eps=eps)
    for _ in range(num_episodes):
        ts = policy.trajectory(env, env.reset())
        G = 0.0
        for s, a, r, ss, _ in reversed(list(ts)):
            G = gamma * G + r
            q[s][a] += alpha * (G - sum(q[ss].values()))
            policy[s] = greedy(q[s])
    return policy.deterministic


# Sarsa - On-policy Temporal Difference control
def sarsa(env: Env, num_episodes=1000, gamma=0.9, eps=0.1, alpha=0.9):
    q, policy = QValue(), Policy(eps=eps)
    for _ in range(num_episodes):
        ts = list(policy.trajectory(env, env.reset()))
        for s, a, r, ss, aa in reversed(ts):
            q[s][a] += alpha * (r + gamma * q[ss][aa] - q[s][a])
            policy[s] = greedy(q[s])
    return policy.deterministic


# Q-learning - Off-policy Temporal Difference control
def q_learning(env: Env, num_episodes=1000, gamma=0.9, eps=0.1, alpha=0.9):
    q, policy = QValue(), Policy(eps=eps)
    for _ in range(num_episodes):
        ts = list(policy.trajectory(env, env.reset()))
        for s, a, r, ss, _ in reversed(ts):
            q[s][a] += alpha * (r + gamma * max(q[ss].values(), default=0) - q[s][a])
            policy[s] = greedy(q[s])
        # rl.log.print_traj(ts)
        # rl.log.print_table("q: ", q)
        # rl.log.print_table("policy: ", policy)
    return policy.deterministic


# Expected Sarsa - On-policy Temporal Difference control
def expected_sarsa(env: Env, num_episodes=1000, gamma=0.9, eps=0.1, alpha=0.9):
    q, policy = QValue(), Policy(eps=eps)
    nA = gym.spaces.flatdim(env.env.action_space)
    for _ in range(num_episodes):
        ts = policy.trajectory(env, env.reset())
        for s, a, r, _, _ in reversed(list(ts)):
            q[s][a] += alpha * (
                r
                + gamma
                * sum(
                    (policy[s].get(a, 0) + policy.eps / nA) * q[s][a] for a in range(nA)
                )
                - q[s][a]
            )
            policy[s] = greedy(q[s])
    return policy.deterministic


# Double Q-learning - Off-policy Temporal Difference control
def double_q_learning(env: Env, num_episodes=1000, gamma=0.9, eps=0.1, alpha=0.9):
    q1, q2, policy = QValue(), QValue(), Policy(eps=eps)
    for _ in range(num_episodes):
        ts = policy.trajectory(env, env.reset())
        for s, a, r, ss, _ in reversed(list(ts)):
            if random.random() < 0.5:
                q1[s][a] += alpha * (
                    r + gamma * max(q2[ss].values(), default=0) - q1[s][a]
                )
            else:
                q2[s][a] += alpha * (
                    r + gamma * max(q1[ss].values(), default=0) - q2[s][a]
                )
            policy[s] = greedy(
                {
                    a: 0.5 * q1[s][a] + 0.5 * q2[s][a]
                    for a in itertools.chain(q1[s], q2[s])
                }
            )
    return policy.deterministic
