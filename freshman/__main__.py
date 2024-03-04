import arguably
import gymnasium as gym
import random

from typing import Any, Optional, Annotated
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


@dataclass
class Env:
    env: gym.Env[Any, Any]
    seed: int | None

    def __post_init__(self):
        self.env.action_space.seed(self.seed)

    # # gym.spaces.flatten(self.env.observation_space, s)  # type: ignore
    # # gym.spaces.flatten(self.env.action_space, a)  # type: ignore
    # # gym.spaces.unflatten(self.env.action_space, s)
    # # gym.spaces.unflatten(self.env.action_space, a)
    def reset(self) -> State:
        state, _ = self.env.reset(seed=self.seed)
        return state

    def step(self, a: Action) -> tuple[State, Reward, bool, bool]:
        s, r, terminated, truncated, _ = self.env.step(a)
        return s, r.__float__(), terminated, truncated

    #
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


@arguably.command
def run(environment: Annotated[str, arguably.arg.choices(*envs.keys())]) -> None:
    print(f"Running {environment}")
    maker = EnvMaker(environment)
    with maker.get(RenderMode.HUMAN) as env:
        env.reset()
        env.step(1)
        env.step(1)
        env.step(2)
        env.step(1)
        env.step(2)
        env.step(2)


if __name__ == "__main__":
    arguably.run(name="freshman")
