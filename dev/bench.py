"""Time the examples of the current checkout against another revision.

    python dev/bench.py --baseline 496b398 --repeat 3

Checks the baseline revision out into a temporary git worktree, runs each
example from each checkout with that checkout's src/ first on the path, and
prints the median wall times and their ratio. Each checkout runs its own
examples, so API changes between the revisions do not matter. Used for the
backend refactor (dev/jit_backends_plan.md): numpy runs must not slow down
by more than 5% without an explanation.
"""

import argparse
import os
import statistics
import subprocess
import sys
import tempfile
import time

COMMANDS: list[list[str]] = [
    ["quickstart.py"],
    ["checkpointing.py"],
    ["matched_vs_mismatched.py"],
    ["matrix_completion.py"],
    ["p_laplacian.py", "--rings", "25"],
    ["p_laplacian_eigen.py", "--rings", "25", "--start", "random"],
]


def time_command(checkout: str, command: list[str]) -> float:
    env = dict(os.environ, PYTHONPATH=os.path.join(checkout, "src"))
    start = time.perf_counter()
    subprocess.run([sys.executable, *command], cwd=os.path.join(checkout, "examples"), env=env,
                   check=True, stdout=subprocess.DEVNULL)
    return time.perf_counter() - start


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--baseline", required=True, help="git revision to compare against")
    parser.add_argument("--repeat", type=int, default=3)
    args = parser.parse_args()
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    with tempfile.TemporaryDirectory() as tmp:
        base = os.path.join(tmp, "baseline")
        subprocess.run(["git", "worktree", "add", "--detach", base, args.baseline], cwd=here,
                       check=True, capture_output=True)
        try:
            print(f"{'example':45s} {'baseline':>9s} {'current':>9s} {'ratio':>6s}")
            for command in COMMANDS:
                times: dict[str, list[float]] = {"base": [], "cur": []}
                for _ in range(args.repeat):          # interleaved, so drift hits both
                    times["base"].append(time_command(base, command))
                    times["cur"].append(time_command(here, command))
                b, c = statistics.median(times["base"]), statistics.median(times["cur"])
                print(f"{' '.join(command):45s} {b:8.2f}s {c:8.2f}s {c / b:6.2f}", flush=True)
        finally:
            subprocess.run(["git", "worktree", "remove", "--force", base], cwd=here, check=True)


if __name__ == "__main__":
    main()
