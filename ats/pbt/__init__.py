"""Population Based Training (PBT) for ats-v2.

Runs several population members (each an independent model + hyperparameter
configuration) side by side, periodically ranks them by an eval metric,
"exploits" the best-performing members by copying their weights onto the
worst-performing ones, and "explores" by perturbing the copied
hyperparameters -- the standard PBT exploit/explore step (Jaderberg et al.,
2017, "Population Based Training of Neural Networks").

Scope, deliberately: this is a single-machine, single-process-per-member
orchestrator. It is intended for small models where running N copies on one
machine is affordable -- NOT for spawning N independently-distributed 7B/70B
training jobs. See ats/pbt/orchestrator.py's module docstring for the exact
limitation and why running the whole population as one big distributed job
is out of scope here.
"""

from __future__ import annotations

from ats.pbt.orchestrator import ATSMemberRunner, MemberRunner, PBTOrchestrator
from ats.pbt.population import (
    PopulationMember,
    exploit_and_explore,
    initialize_population,
    perturb_config,
    rank_by_fitness,
    select_cull,
)
from ats.pbt.schema import PBTConfig, PerturbSpec

__all__ = [
    "ATSMemberRunner",
    "MemberRunner",
    "PBTConfig",
    "PBTOrchestrator",
    "PerturbSpec",
    "PopulationMember",
    "exploit_and_explore",
    "initialize_population",
    "perturb_config",
    "rank_by_fitness",
    "select_cull",
]
