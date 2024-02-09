from dataclasses import dataclass, field
import itertools
from typing import Iterable, Optional

from matplotlib.figure import Figure
from freshman.algorithms import Progress
from freshman.env import SARSA, Env, QValue, Policy, EnvMaker, Reward
import numpy as np
from matplotlib.tri import Triangulation
import matplotlib.animation as ani
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec


def _triangles():
    xs = [i - 0.5 for i in range(5) for _ in range(5)] + [
        i for i in range(4) for _ in range(4)
    ]
    ys = [j - 0.5 for _ in range(5) for j in range(5)] + [
        j for _ in range(4) for j in range(4)
    ]
    tri = np.asarray(
        [
            triangle
            for i in range(4)
            for j in range(4)
            for triangle in (
                [j * 5 + i + 0, j * 5 + i + 1, 4 * j + i + 25],
                [j * 5 + i + 1, j * 5 + i + 6, 4 * j + i + 25],
                [j * 5 + i + 6, j * 5 + i + 5, 4 * j + i + 25],
                [j * 5 + i + 5, j * 5 + i + 0, 4 * j + i + 25],
            )
        ]
    )
    return Triangulation(xs, ys, triangles=tri)


triangles = _triangles()


def q_to_arrows(*, ax, q: QValue):
    def args(val, mx):
        if val == mx:
            return {"width": 0.03, "color": "limegreen"}
        else:
            return {"width": 0.03, "color": "red"}

    for s, acts in q.items():
        mx = max(max(acts.values()), 1e-5)
        for a, val in acts.items():
            length = val / mx / 2 * 0.75
            match a:
                case 0:
                    ax.arrow(x=s % 4, y=s // 4, dx=-length, dy=0, **args(val, mx))
                case 1:
                    ax.arrow(x=s % 4, y=s // 4, dx=0, dy=length, **args(val, mx))
                case 2:
                    ax.arrow(x=s % 4, y=s // 4, dx=length, dy=0, **args(val, mx))
                case 3:
                    ax.arrow(x=s % 4, y=s // 4, dx=0, dy=-length, **args(val, mx))


def heatmap(q: QValue, ax):
    ax.axis("off")
    ax.invert_yaxis()
    ax.set_aspect(1)
    ax.tripcolor(
        triangles, [q[s][a] for s in range(16) for a in range(4)], cmap="Blues"
    )
    q_to_arrows(q=q, ax=ax)
    # ax.colorbar(tc, ax=ax)


def plot_rewards(rws: Iterable[Reward], ax):
    # ax.cla()
    partials = list(itertools.accumulate(rws))
    ax.plot(partials)


def plot(
    env: Env,
    q: QValue,
    *,
    fig=None,
    rewards: Optional[Iterable[Reward]] = None,
    show: bool = False,
):
    if fig is None:
        fig = plt.gcf()
    gs = GridSpec(nrows=2, ncols=2)
    if rewards is not None:
        plot_rewards(rewards, fig.add_subplot(gs[1, :]))
    heatmap(q, fig.add_subplot(gs[0, 0]))
    ax = fig.add_subplot(gs[0, 1])
    frame = env.env.render()
    ax.axis("off")
    ax.imshow(frame)  # type: ignore
    if show:
        plt.show()
    return fig


def plot_img(q: QValue, img):
    # dpi: int = 128
    fig, axs = plt.subplots(1, 2, figsize=(15, 5))
    ax = axs[0]
    # fig.subplots_adjust(left=0, bottom=0, right=1, top=1)
    ax.axis("off")
    ax.imshow(img)
    heatmap(q, axs[1])
    # ax.arrow(x=64, y=64, dx=64, dy=64, width=0.4)
    # q_to_arrows(q, axs[1])
    fig.savefig("test.png")


def video_episode(env: Env, q: QValue, name: str) -> Progress:
    def f(xs: Iterable[SARSA]) -> Iterable[SARSA]:
        rws: list[Reward] = []
        writer = ani.PillowWriter()
        fig = plt.figure()
        with writer.saving(fig, f"docs/video/{name}.gif", dpi=100):
            for x in xs:
                _, _, r, _, _ = x
                rws.append(r)
                plt.clf()
                plot(env, q, rewards=rws)
                writer.grab_frame()
                yield x

    return f


@dataclass
class TrainingVideo:
    writer: ani.FFMpegWriter = field(default_factory=ani.FFMpegWriter)
    name: str = "untitled"
    fig: Figure = field(default_factory=plt.figure)
    rws: list[Reward] = field(default_factory=list)
    format: str = "gif"

    def __enter__(self):
        self.writer.setup(self.fig, f"docs/video/{self.name}.{self.format}", dpi=100)
        return self

    def step(self, env: Env, q: QValue):
        def f(xs: Iterable[SARSA]) -> Iterable[SARSA]:
            for x in xs:
                _, _, r, _, _ = x
                self.rws.append(r)
                plt.clf()
                plot(env, q, rewards=self.rws)
                self.writer.grab_frame()
                yield x

        return f

    def __exit__(self, *_):
        self.writer.finish()


# Needs ffmpeg
def video(
    em: EnvMaker,
    policy: Policy,
    file_name: str,
    *,
    fps: int = 1,
    title: str = "",
    limit: int = 30,
    dpi: int = 100,
):
    with em("rgb_array") as env:
        _, first = env.reset(), env.env.render()
        height, width, _ = first.shape  # type: ignore
        fig, ax = plt.subplots(figsize=(width / dpi, height / dpi))
        # no extra white space
        fig.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=None, hspace=None)
        writer = ani.writers["ffmpeg"](fps=fps, metadata={"title": title})
        with writer.saving(fig, "docs/video/" + file_name, dpi=dpi):
            im = ax.imshow(first, interpolation="nearest")
            writer.grab_frame()
            for img in (env.env.render() for _ in policy.trajectory(env, limit=limit)):
                im.set_data(img)
                writer.grab_frame()
