import random
from typing import Any, Callable, DefaultDict, Iterator, TypeVar
from collections import defaultdict
from collections.abc import Mapping
from dataclasses import dataclass

import gymnasium as gym
from gymnasium.wrappers.flatten_observation import FlattenObservation
import freshman.log


State = int
Action = int
Reward = float
Probability = float
Trajectory = Iterator[tuple[State, Action, Reward, State, Action]]
T = TypeVar("T")
Tabular = Mapping[State, Mapping[Action, T]]


def tabular(default: T) -> DefaultDict[State, DefaultDict[Action, T]]:
    return defaultdict(lambda: defaultdict(lambda: default))


@dataclass
class Env:
    env: gym.Env[Any, Any]
    seed: int | None

    def __init__(self, env: gym.Env[Any, Any], *, seed: int | None = None):
        self.env = env  # FlattenObservation(env)
        self.seed = seed
        self.env.action_space.seed(seed)

    # gym.spaces.flatten(self.env.observation_space, s)  # type: ignore
    # gym.spaces.flatten(self.env.action_space, a)  # type: ignore
    # gym.spaces.unflatten(self.env.action_space, s)
    # gym.spaces.unflatten(self.env.action_space, a)
    def reset(self) -> State:
        state, _ = self.env.reset(seed=self.seed)
        return state

    def step(self, a: Action) -> tuple[State, Reward, bool, bool]:
        s, r, terminated, truncated, _ = self.env.step(a)
        return s, r.__float__(), terminated, truncated

    def __enter__(self, *_):
        return self

    def __exit__(self, *_):
        self.env.close()


@dataclass
class QValue(DefaultDict[State, DefaultDict[Action, Reward]]):
    vals: Tabular[Reward]

    def __init__(self, default: float = 0, dict: Tabular[Reward] = {}):
        super().__init__(
            lambda: defaultdict(lambda: default),
            ((s, defaultdict(lambda: default, ap)) for s, ap in dict.items()),
        )


@dataclass
class Policy(DefaultDict[State, DefaultDict[Action, Probability]]):
    eps: float

    def __init__(self, eps: float = 0, dict: Tabular[Probability] = {}):
        super().__init__(
            lambda: defaultdict(lambda: 0),
            ((s, defaultdict(lambda: 0, ap)) for s, ap in dict.items()),
        )
        self.eps = eps

    def __setitem__(self, s: State, it: Mapping[Action, Probability]):
        super().__setitem__(s, defaultdict(lambda: 0, it))

    @property
    def deterministic(self):
        return Policy(eps=0, dict=self)

    def sample(self, env: Env, s: State) -> Action:
        if not self[s] or random.random() < self.eps:
            return env.env.action_space.sample()
        else:
            return random.choices(list(self[s].keys()), list(self[s].values()))[0]

    def trajectory(self, env: Env, *, limit: int = 1000) -> Trajectory:
        s0 = env.reset()
        a0 = self.sample(env, s0)  # s0 -> a0
        for _ in range(limit):
            s1, r, terminated, truncated = env.step(a0)  # a0 -> s1
            a1 = self.sample(env, s1)  # s1 -> a1
            yield s0, a0, r, s1, a1
            s0, a0 = s1, a1  # restart
            if terminated or truncated:
                return


def greedy(qs: dict[Action, Reward]) -> dict[Action, Probability]:
    return {max(qs, key=qs.get): 1.0} if qs else {}  # type: ignore - pyright


Evaluation = Callable[[Policy], QValue]
Improvement = Callable[[QValue], Policy]
