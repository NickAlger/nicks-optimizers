"""The switch from the Gauss-Newton to the full Hessian (hessian="switch"),
as far as the drivers of the two Newton optimizers share it (Python, around
the compiled pieces; see dev/archive/gauss_newton_switch_plan.md):

  resolve_switch      the keywords checked and resolved: the detector's
                      tau and k and the forcing powers of the two phases
  SwitchTracker       the detector's state over a run: the phase, the
                      consecutive hits, the iteration of the switch
  split_hessvec_calls the work by Hessian, from the records
  switch_header, switch_columns, switch_summary_line
                      the extra display of a switch run

A run with hessian="switch" starts with gn_hessvec as the model Hessian and
pre_switch_forcing_power. After every iteration of that phase the pieces
report the Gauss-Newton and the full model values at the step tried (one
product with the full Hessian), and criteria.switch_sample compares them
with the actual change of f. When k consecutive iterations are hits (an
invalid sample or a rejected step resets the count), the run switches for
good: from the next iteration hessvec is the model Hessian, with
forcing_power, and nothing is tracked.
"""

from dataclasses import dataclass
from typing import Any, Literal, Optional, Protocol, Sequence

import numpy as np

from .criteria import switch_sample

Hessian = Literal["full", "switch"]
Model = Literal["gauss_newton", "full"]

DEFAULT_TAU = 0.25
DEFAULT_K = 2
DEFAULT_POWER = 0.5            # the forcing power: hessian="full", and both phases of a switch


@dataclass(frozen=True, kw_only=True)
class SwitchSettings:
    """The resolved settings of a run: whether it switches, the detector's
    tau and k, and the forcing powers before and after the switch (with
    hessian="full" only post_power is used: the run's forcing power)."""
    switch: bool
    tau: float
    k: int
    pre_power: float
    post_power: float


def _check_power(name: str, power: Optional[float]) -> None:
    if power is not None and not (0.0 < power < np.inf):
        raise ValueError(f"{name} must be positive and finite, not {power}")


def resolve_switch(*, hessian: str, gn_hessvec: Optional[Any], operators_at: Optional[Any],
                   switch_tau: Optional[float], switch_k: Optional[int],
                   pre_switch_forcing_power: Optional[float],
                   forcing_power: Optional[float]) -> SwitchSettings:
    """Check the switch keywords and resolve their defaults. The forcing
    powers (before, after) of a switch run are 0.5 by default; forcing_power
    sets both, and pre_switch_forcing_power the first:

        pre_switch_forcing_power  forcing_power  ->  before  after
        None                      None               0.5     0.5
        p                         None               p       0.5
        None                      q                  q       q
        p                         q                  p       q

    (0.5 after the switch was cheaper than 1.0 in 46 of 58 runs of
    experiments/gn_newton_switch.)

    With hessian="full" the forcing power is forcing_power (None: 0.5), and
    the switch keywords raise."""
    _check_power("forcing_power", forcing_power)
    _check_power("pre_switch_forcing_power", pre_switch_forcing_power)
    if hessian not in ("full", "switch"):
        raise ValueError(f"hessian must be 'full' or 'switch', not {hessian!r}")
    if hessian == "full":
        given = [name for name, value in (("gn_hessvec", gn_hessvec), ("switch_tau", switch_tau),
                                          ("switch_k", switch_k),
                                          ("pre_switch_forcing_power", pre_switch_forcing_power))
                 if value is not None]
        if given:
            raise ValueError(f"{', '.join(given)} only with hessian='switch'")
        power = DEFAULT_POWER if forcing_power is None else forcing_power
        return SwitchSettings(switch=False, tau=DEFAULT_TAU, k=DEFAULT_K, pre_power=power,
                              post_power=power)
    if operators_at is not None and gn_hessvec is not None:
        raise ValueError("operators_at gives all the operators at a point: give gn_hessvec in "
                         "its PointOperators, not as a keyword")
    if operators_at is None and gn_hessvec is None:
        raise ValueError("hessian='switch' needs gn_hessvec (or operators_at giving it)")
    tau = DEFAULT_TAU if switch_tau is None else switch_tau
    if not 0.0 < tau < 1.0:
        raise ValueError(f"switch_tau must lie in (0, 1), not {tau}")
    k = DEFAULT_K if switch_k is None else switch_k
    if isinstance(k, bool) or not isinstance(k, (int, np.integer)) or k < 1:
        raise ValueError(f"switch_k must be a positive integer, not {k!r}")
    post = DEFAULT_POWER if forcing_power is None else forcing_power
    pre = post if pre_switch_forcing_power is None else pre_switch_forcing_power
    return SwitchSettings(switch=True, tau=float(tau), k=int(k), pre_power=pre, post_power=post)


class SwitchTracker:
    """The detector's state over a run (Python, in the driver).

      switched          whether the full Hessian is the model now
      hits              consecutive hits so far (before the switch)
      switch_iteration  the first iteration with the full Hessian, if the
                        switch happened in this run

    observe() takes one iteration's numbers and returns the switch fields
    of its record."""

    def __init__(self, settings: SwitchSettings, switched: bool = False, hits: int = 0) -> None:
        self.settings = settings
        self.switched = switched
        self.hits = hits
        self.switch_iteration: Optional[int] = None

    @property
    def tracking(self) -> bool:
        """Whether this iteration tracks the full model (before the switch)."""
        return self.settings.switch and not self.switched

    def observe(self, iteration: int, f: float, actual: float, gn_model: float,
                full_model: float, accepted: bool) -> dict[str, Any]:
        """One iteration (actual = f - f(trial), as recorded): the record's
        switch fields; switches when the hits reach k."""
        if not self.settings.switch:
            return {}
        if self.switched:
            return {"model": "full"}
        _, hit, e_gn, e_full = switch_sample(f, -actual, gn_model, full_model, self.settings.tau)
        counted = hit and accepted
        self.hits = self.hits + 1 if counted else 0
        fired = self.hits >= self.settings.k
        if fired:
            self.switched, self.hits, self.switch_iteration = True, 0, iteration + 1
        return {"model": "gauss_newton", "error_gn": e_gn, "error_full": e_full,
                "switch_hits": self.settings.k if fired else self.hits, "switch_fired": fired}


class _SwitchRecord(Protocol):
    """What the display and the work count read from either record type."""
    @property
    def model(self) -> Optional[str]: ...
    @property
    def error_gn(self) -> Optional[float]: ...
    @property
    def error_full(self) -> Optional[float]: ...
    @property
    def switch_fired(self) -> Optional[bool]: ...
    @property
    def subproblem(self) -> Any: ...


def split_hessvec_calls(records: Sequence[_SwitchRecord]) -> tuple[int, int]:
    """(products with hessvec, products with gn_hessvec) over the records:
    the solves of the Gauss-Newton phase use gn_hessvec, everything else
    hessvec, and each iteration of the Gauss-Newton phase one more product
    with hessvec (the tracking product). Without a switch every product is
    with hessvec."""
    full = sum(r.subproblem.hessvec_calls for r in records if r.model != "gauss_newton")
    gn = sum(r.subproblem.hessvec_calls for r in records if r.model == "gauss_newton")
    tracking = sum(1 for r in records if r.model == "gauss_newton")
    return full + tracking, gn


_COLUMNS = "  {h:>2}  {ratio:>7}{mark}"


def switch_header() -> str:
    """The extra column headings of a switch run: the model Hessian and
    e_full / e_gn."""
    return _COLUMNS.format(h="H", ratio="eN/eGN", mark="")


def switch_columns(record: _SwitchRecord) -> str:
    """The extra columns of a switch run's row: GN or N, the ratio of the
    model errors before the switch ("-" when it is not finite), and * on
    the iteration after which the switch happened."""
    if record.model == "full":
        return _COLUMNS.format(h="N", ratio="", mark="")
    e_gn, e_full = record.error_gn, record.error_full
    ratio = (e_full / e_gn if e_gn is not None and e_full is not None and e_gn > 0.0
             else float("nan"))
    return _COLUMNS.format(h="GN", ratio=f"{ratio:.2f}" if np.isfinite(ratio) else "-",
                           mark=" *" if record.switch_fired else "")


def switch_summary_line(hessian: str, switch_iteration: Optional[int], hessvec_calls: int,
                        gn_hessvec_calls: int) -> Optional[str]:
    """The summary's line about the switch (None without hessian='switch')."""
    if hessian != "switch":
        return None
    when = ("did not switch" if switch_iteration is None
            else f"switched to the full Hessian at iteration {switch_iteration}")
    return (f"  {when}; {gn_hessvec_calls} Gauss-Newton and {hessvec_calls} full "
            f"Hessian-vector products")
