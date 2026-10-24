"""Checkpointing and exact resumption.

The callback writes a checkpoint every few iterations: the iterate x, the
gradient g (so the restart can skip one gradient evaluation) and the small
ResumeState, which holds the trust-region radius, the reference gradient norm
and the iteration count. A simulated crash interrupts the run; resuming from
the last checkpoint then reproduces the uninterrupted run exactly.

Run:  python examples/checkpointing.py
"""

import dataclasses
import json
import tempfile
from pathlib import Path
from typing import Callable

import numpy as np

from nicks_optimizers import (
    Covector, IterationInfo, Point, ResumeState, RunSummary, Vector, trust_region_newton,
)
from quickstart import fun, grad, hessvec, jacobi_solve


class SimulatedCrash(Exception):
    pass


Callback = Callable[[Point, IterationInfo[Vector, Covector]], None]


def make_checkpointer(directory: Path, every: int, crash_at: int | None = None) -> Callback:
    def checkpoint(x: Point, info: IterationInfo[Vector, Covector]) -> None:
        if info.record.iteration % every == 0:
            # Write to temporary names, then rename, so a crash mid-write
            # cannot leave a half-written checkpoint behind.
            np.save(directory / "x.tmp.npy", x)
            np.save(directory / "g.tmp.npy", info.g)
            (directory / "state.tmp.json").write_text(json.dumps(dataclasses.asdict(info.resume)))
            for name in ("x", "g"):
                (directory / f"{name}.tmp.npy").replace(directory / f"{name}.npy")
            (directory / "state.tmp.json").replace(directory / "state.json")
        if crash_at is not None and info.record.iteration == crash_at:
            raise SimulatedCrash(f"crash after iteration {crash_at}")
    return checkpoint


def resume_from(directory: Path) -> tuple[Point, Covector, ResumeState]:
    state = ResumeState(**json.loads((directory / "state.json").read_text()))
    return Point(np.load(directory / "x.npy")), Covector(np.load(directory / "g.npy")), state


if __name__ == "__main__":
    x0 = Point(np.tile([-1.2, 1.0], 25))

    def run(x: Point, callback: Callback | None = None, resume: ResumeState | None = None,
            g0: Covector | None = None) -> tuple[Point, RunSummary]:
        """The same options every time, as exact resumption requires."""
        return trust_region_newton(fun, grad, hessvec, x, metric_solve=jacobi_solve,
                                   callback=callback, resume=resume, g0=g0)

    x_full, full = run(x0)
    print(f"uninterrupted run: {len(full.iterations)} iterations")

    with tempfile.TemporaryDirectory() as tmp:
        directory = Path(tmp)
        try:
            run(x0, callback=make_checkpointer(directory, every=10, crash_at=43))
        except SimulatedCrash as crash:
            print(f"first run: {crash}")
        x, g, state = resume_from(directory)
        print(f"resuming from the checkpoint taken after iteration {state.next_iteration - 1}")
        x_res, rest = run(x, resume=state, g0=g)

    same = np.array_equal(x_res, x_full)
    print(f"resumed run: iterations {rest.iterations[0].iteration} to {rest.iterations[-1].iteration}; "
          f"final x identical to the uninterrupted run: {same}")
