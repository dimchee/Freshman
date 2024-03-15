import arguably
from typing import Annotated, Callable, Dict, Iterable, TypeVar
from dataclasses import dataclass, field
from tqdm import tqdm
from freshman.env import (
    Env,
    EnvMaker,
    RenderMode,
    Action,
    Reward,
    Probability,
    Policy,
    QValue,
    Trajectory,
    envs,
)


T = TypeVar("T")
Progress = Callable[[Iterable[T], Policy, QValue], Iterable[T]]


@dataclass(kw_only=True)
class Parameters:
    env: Env
    num_episodes: int = 200
    gamma: float = 0.9
    eps: float = 0.1
    alpha: float = 0.9
    episodic_progress: Progress = lambda x, *_: x  # noqa: E731
    progress: Progress = lambda x, *_: x  # noqa: E731
    q: QValue = field(default_factory=QValue)
    policy: Policy = field(default_factory=Policy)

    def __post_init__(self):
        self.policy.eps = self.eps

    def trajectory(self) -> Trajectory:
        return self.progress(self.policy.trajectory(self.env), self.policy, self.q)  # type: ignore


AlgStep = Callable[[Parameters], None]
Algorithm = Callable[[Parameters], tuple[Policy, QValue]]
algorithms: Dict[str, Algorithm] = {}


# Add way to add input parameters (like 'returns')
def algorithm(alg_step: AlgStep):
    def alg(opts: Parameters) -> tuple[Policy, QValue]:
        for _ in opts.episodic_progress(range(opts.num_episodes), opts.policy, opts.q):
            alg_step(opts)
        return opts.policy, opts.q

    algorithms[alg_step.__name__] = alg
    return alg


def greedy(qs: dict[Action, Reward]) -> dict[Action, Probability]:
    return {max(qs, key=qs.get): 1.0} if qs else {}  # type: ignore - pyright


@algorithm
def q_learning(opts: Parameters):
    for s, a, r, ss, _ in opts.trajectory():
        update = r + opts.gamma * max(opts.q[ss].values(), default=0) - opts.q[s][a]
        opts.q[s][a] += opts.alpha * update
        opts.policy[s] = greedy(opts.q[s])


@arguably.command
def run(environment: Annotated[str, arguably.arg.choices(*envs.keys())]) -> None:
    print(f"Running {environment}")
    maker = EnvMaker(environment)
    with maker.get(RenderMode.ANSI) as env:
        model = Parameters(
            env=env,
            num_episodes=10000,
            eps=0.2,
            gamma=0.9,
            episodic_progress=lambda x, *_: tqdm(x),
        )
        q_learning(model)
        policy = model.policy
    with maker.get(RenderMode.HUMAN) as env:
        for _ in policy.trajectory(env):
            pass


if __name__ == "__main__":
    arguably.run(name="freshman")
