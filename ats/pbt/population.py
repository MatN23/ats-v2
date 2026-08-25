"""Pure population-management logic for PBT: no I/O, no training, no
DeepSpeed. Everything here operates on ATSConfig objects and plain data,
which is what makes it possible to unit-test without a GPU or even torch
actually running a model -- see tests/test_pbt.py.

Weight cloning (the actual "exploit" copy of model parameters) is
deliberately NOT done here: this module only decides WHICH member's config a
culled member should inherit and how that config should be perturbed. Moving
the resulting weights onto disk is ats.pbt.orchestrator's job, since that
requires touching checkpoint files.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Any

from ats.config.schema import ATSConfig, ConfigError
from ats.pbt.schema import PerturbSpec


@dataclass
class PopulationMember:
    """One member of the population. `config` is this member's current
    ATSConfig (post-perturbation, if any); `checkpoint_dir` is where its
    weights live on disk. `fitness` is the member's most recent eval metric
    (lower is better -- e.g. eval loss or perplexity) and is None until the
    member has trained at least one generation."""

    member_id: int
    config: ATSConfig
    checkpoint_dir: str
    fitness: float | None = None
    generation: int = 0
    parent_id: int | None = None
    history: list[dict[str, Any]] = field(default_factory=list)


def _get_nested(config: ATSConfig, dotted_name: str) -> Any:
    section_name, field_name = dotted_name.split(".")
    section = getattr(config, section_name)
    return getattr(section, field_name)


def _set_nested(config: ATSConfig, dotted_name: str, value: Any) -> ATSConfig:
    """Returns a NEW ATSConfig with one field set, re-validated through
    ATSConfig.model_validate so cross-field validators (e.g.
    training.weight_decay >= 0) still catch an invalid perturbed value
    instead of silently producing a broken config -- the same pattern
    ats.cli.train.py's apply_cli_overrides uses after a
    model_copy(update=...)."""
    section_name, field_name = dotted_name.split(".")
    section = getattr(config, section_name)
    new_section = section.model_copy(update={field_name: value})
    new_config = config.model_copy(update={section_name: new_section})
    try:
        return ATSConfig.model_validate(new_config.model_dump())
    except Exception as exc:
        raise ConfigError(
            f"Perturbing {dotted_name} to {value} produced an invalid config: {exc}"
        ) from exc


def perturb_config(
    config: ATSConfig, specs: list[PerturbSpec], rng: random.Random
) -> ATSConfig:
    """Applies every PerturbSpec in `specs` to `config` in turn, returning a
    new ATSConfig. Each field is either multiplicatively perturbed
    (current * U(factor_low, factor_high)) or, with probability
    resample_probability, resampled fresh from its bounds -- see
    PerturbSpec's docstring for exactly which."""
    for spec in specs:
        current = _get_nested(config, spec.name)
        if not isinstance(current, (int, float)) or isinstance(current, bool):
            raise ConfigError(
                f"PerturbSpec({spec.name!r}) targets a non-numeric field "
                f"(current value {current!r}); PBT perturbation only supports "
                f"numeric hyperparameters."
            )

        if rng.random() < spec.resample_probability:
            low = (
                spec.min_value
                if spec.min_value is not None
                else current * spec.factor_low
            )
            high = (
                spec.max_value
                if spec.max_value is not None
                else current * spec.factor_high
            )
            if low > high:
                low, high = high, low
            new_value: float = rng.uniform(low, high)
        else:
            factor = rng.uniform(spec.factor_low, spec.factor_high)
            new_value = current * factor

        if spec.min_value is not None:
            new_value = max(spec.min_value, new_value)
        if spec.max_value is not None:
            new_value = min(spec.max_value, new_value)
        if isinstance(current, int):
            new_value = round(new_value)

        config = _set_nested(config, spec.name, new_value)
    return config


def initialize_population(
    base_config: ATSConfig,
    population_size: int,
    steps_per_generation: int,
    perturb_specs: list[PerturbSpec],
    output_dir: str,
    seed: int,
) -> list[PopulationMember]:
    """Builds the generation-0 population. Every member's training.max_steps
    is forced to steps_per_generation (so the LR scheduler sees a complete
    warmup+decay cycle every generation instead of only the early part of a
    schedule shaped for a much longer run), and every member gets a distinct
    training.seed (base seed + member_id) so a population with no
    perturb_specs at all still starts from N different random
    initializations/data orderings instead of N identical clones."""
    members = []
    for member_id in range(population_size):
        # random.Random() only accepts None/int/float/str/bytes/bytearray --
        # a (seed, member_id) tuple raises TypeError. A distinct string per
        # member gives every member an independent, reproducible stream
        # without members' RNGs correlating trivially (e.g. seed+member_id
        # as a bare int would make member deltas predictable in lockstep).
        rng = random.Random(f"{seed}-{member_id}")
        cfg = base_config.model_copy(
            update={
                "training": base_config.training.model_copy(
                    update={
                        "max_steps": steps_per_generation,
                        "seed": base_config.training.seed + member_id,
                    }
                )
            }
        )
        if perturb_specs:
            cfg = perturb_config(cfg, perturb_specs, rng)
        checkpoint_dir = f"{output_dir.rstrip('/')}/member_{member_id}"
        cfg = cfg.model_copy(
            update={
                "checkpoint": cfg.checkpoint.model_copy(
                    update={"output_dir": checkpoint_dir}
                )
            }
        )
        members.append(
            PopulationMember(
                member_id=member_id, config=cfg, checkpoint_dir=checkpoint_dir
            )
        )
    return members


def rank_by_fitness(members: list[PopulationMember]) -> list[PopulationMember]:
    """Returns members sorted best-first (lowest fitness first -- fitness is
    an eval loss/perplexity, so lower is better). Raises if any member has no
    fitness recorded yet, since ranking an untrained member is a bug in the
    caller, not a case to silently paper over."""
    for m in members:
        if m.fitness is None:
            raise ConfigError(
                f"Cannot rank population member {m.member_id}: it has no "
                f"fitness recorded yet. Fix: evaluate every member before "
                f"calling rank_by_fitness."
            )
    return sorted(members, key=lambda m: m.fitness)  # type: ignore[arg-type,return-value]


def select_cull(
    ranked_members: list[PopulationMember], cull_fraction: float
) -> tuple[list[PopulationMember], list[PopulationMember]]:
    """Splits a best-first ranked population into (survivors, culled).
    `culled` is always at least 1 member and never the whole population,
    regardless of rounding, so a cull_fraction near 0 or 1 can't silently
    cull nobody or everybody."""
    n = len(ranked_members)
    n_cull = round(n * cull_fraction)
    n_cull = max(1, min(n - 1, n_cull))
    survivors = ranked_members[: n - n_cull]
    culled = ranked_members[n - n_cull :]
    return survivors, culled


def exploit_and_explore(
    survivors: list[PopulationMember],
    culled: list[PopulationMember],
    perturb_specs: list[PerturbSpec],
    rng: random.Random,
) -> list[tuple[PopulationMember, PopulationMember]]:
    """The PBT exploit/explore step. For each culled member, picks a random
    survivor as its new "parent", perturbs that survivor's hyperparameters,
    and mutates the culled member's `config`/`parent_id` in place to match --
    but keeps the culled member's own `checkpoint_dir` (its own directory on
    disk), since the ACTUAL weight file copy happens in the orchestrator, one
    layer up, and needs a source (the parent's checkpoint) and a stable
    destination path to write to.

    Returns a list of (member, source_member) pairs describing exactly which
    checkpoint to copy weights FROM for each culled member -- the
    orchestrator uses this to do the file I/O."""
    if not survivors:
        raise ConfigError(
            "exploit_and_explore got an empty survivors list -- nothing to "
            "exploit from. Fix: select_cull must leave at least one survivor."
        )
    assignments = []
    for member in culled:
        source = rng.choice(survivors)
        new_config = source.config
        if perturb_specs:
            new_config = perturb_config(new_config, perturb_specs, rng)
        # Keep this member's own checkpoint/output directory: it must not
        # start writing its checkpoints into the source member's directory.
        new_config = new_config.model_copy(
            update={"checkpoint": member.config.checkpoint}
        )
        member.config = ATSConfig.model_validate(new_config.model_dump())
        member.parent_id = source.member_id
        assignments.append((member, source))
    return assignments
