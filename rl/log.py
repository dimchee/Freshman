import logging
import pprint
from rl.env import QValue, Policy, Trajectory


def start():
    logging.basicConfig(
        level=logging.DEBUG,
        format=" %(asctime)s - %(levelname)s- %(message)s",
        filename="log",
        filemode="w",
    )


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


def statify(s: int):
    return s  # divmod(s, 4)


def print(reason: str, x):
    logging.log(logging.DEBUG, reason + "\n" + pprint.pformat(x, indent=4))


def print_traj(x):
    logging.log(
        logging.DEBUG,
        "trajectory: \n"
        + pprint.pformat(
            [
                (statify(s), arrowify(a), r, statify(ss), arrowify(aa))
                for s, a, r, ss, aa in x
            ],
            indent=4,
        ),
    )


def print_table(reason: str, what: QValue | Policy):
    logging.log(
        logging.DEBUG,
        reason
        + "\n"
        + pprint.pformat(
            {
                statify(s): {arrowify(a): f"{p:^6.3f}" for a, p in ap.items()}
                for s, ap in what.items()
            },
            indent=4,
        ),
    )
