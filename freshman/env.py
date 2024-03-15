import gymnasium as gym
import random

from typing import Any, Optional, DefaultDict, TypeVar, Iterator
from collections.abc import Mapping
from collections import defaultdict
from enum import StrEnum
from dataclasses import dataclass

envs = {
    "lake": {"id": "FrozenLake-v1", "is_slippery": False},
    "lake_easy": {"id": "FrozenLake-v1", "desc": ["FFFSFFFG"], "is_slippery": False},
    "lake_slippery": {"id": "FrozenLake-v1", "is_slippery": True},
    "cliff": {"id": "CliffWalking-v0"},
}


State = int
Action = int
Reward = float
Probability = float
Sarsa = tuple[State, Action, Reward, State, Action]
Trajectory = Iterator[Sarsa]


@dataclass
class Env:
    env: gym.Env[Any, Any]
    seed: int | None

    def __post_init__(self):
        self.env.action_space.seed(self.seed)
        self.reset()

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


class RenderMode(StrEnum):
    HUMAN = "human"
    RGB_ARRAY = "rgb_array"
    ANSI = "ansi"


@dataclass
class EnvMaker:
    env_name: str
    seed: Optional[int] = None

    def get(self, render_mode: RenderMode):
        random.seed(self.seed)
        return Env(
            gym.make(**envs[self.env_name], render_mode=render_mode), seed=self.seed
        )


U = TypeVar("U")
T = TypeVar("T")
Tabular = Mapping[State, Mapping[Action, T]]


def tabular(default: T, dict: Mapping[U, T] = {}) -> DefaultDict[U, T]:
    return defaultdict(lambda: default, dict)


@dataclass
class QValue(DefaultDict[State, DefaultDict[Action, Reward]]):
    def __init__(self, default: float = 0, dict: Tabular[Reward] = {}):
        super().__init__(
            lambda: tabular(default),
            ((s, tabular(default, ap)) for s, ap in dict.items()),
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
