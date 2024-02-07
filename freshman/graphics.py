from freshman.env import QValue, Policy, EnvMaker
import numpy as np
from matplotlib.tri import Triangulation
import matplotlib.animation as ani
import matplotlib.pyplot as plt


def triangles():
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


def heatmap(q: QValue, ax, fig):
    ax.axis("off")
    tri = triangles()
    plt.gca().invert_yaxis()
    ax.set_aspect(1)
    tc = ax.tripcolor(tri, [q[s][a] for s in range(16) for a in range(4)], cmap="Blues")
    q_to_arrows(q=q, ax=ax)
    fig.colorbar(tc, ax=ax)


def plot_img(q: QValue, img):
    # dpi: int = 128
    plt.axis("off")
    fig, axs = plt.subplots(1, 2, figsize=(15, 5))
    ax = axs[0]
    # fig.subplots_adjust(left=0, bottom=0, right=1, top=1)
    ax.axis("off")
    ax.imshow(img)
    heatmap(q, axs[1], fig)
    # ax.arrow(x=64, y=64, dx=64, dy=64, width=0.4)
    # q_to_arrows(q, axs[1])
    plt.axis("off")
    plt.savefig("test.png")


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


def q_to_arrows(*, ax, q: QValue):
    def args(val, mx):
        if val == mx:
            return {"width": 0.03, "color": "green", "length_includes_head": True}
        else:
            return {"width": 0.03, "color": "red", "length_includes_head": True}

    m = max((val for acts in q.values() for val in acts.values())) * 2
    for s, acts in q.items():
        mx = max(acts.values())
        for a, val in acts.items():
            match a:
                case 0:
                    ax.arrow(x=s % 4, y=s // 4, dx=-val / m, dy=0, **args(val, mx))
                case 1:
                    ax.arrow(x=s % 4, y=s // 4, dx=0, dy=val / m, **args(val, mx))
                case 2:
                    ax.arrow(x=s % 4, y=s // 4, dx=val / m, dy=0, **args(val, mx))
                case 3:
                    ax.arrow(x=s % 4, y=s // 4, dx=0, dy=-val / m, **args(val, mx))
