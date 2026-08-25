"""Pydantic schema for a Population Based Training (PBT) run.

Mirrors the style of ats.config.schema: every field is validated up front
with an actionable error message, so a bad PBT config fails loudly before any
population member starts training rather than partway through generation 3.
"""

from __future__ import annotations

from pydantic import BaseModel, Field, field_validator, model_validator

from ats.config.schema import ConfigError


class PerturbSpec(BaseModel):
    """One hyperparameter PBT is allowed to mutate on the losing half of the
    population during the exploit/explore step.

    `name` is a dotted "section.field" path into ATSConfig, e.g.
    "training.learning_rate" or "model.dropout" -- only top-level ATSConfig
    sections (training, model, optimizer, ...) are supported, not deeper
    nesting.

    Perturbation is multiplicative: the new value is
    current_value * U(factor_low, factor_high), clamped to
    [min_value, max_value] if given. With probability
    resample_probability, a fresh value is drawn uniformly from
    [min_value, max_value] instead (or from
    [current * factor_low, current * factor_high] if no explicit bounds are
    given) -- this "resample" step is what lets PBT escape a hyperparameter
    region the whole population has drifted into, matching the original PBT
    paper's explore step.
    """

    name: str
    factor_low: float = 0.8
    factor_high: float = 1.2
    min_value: float | None = None
    max_value: float | None = None
    resample_probability: float = 0.25

    @field_validator("name")
    @classmethod
    def _validate_name(cls, v: str) -> str:
        if "." not in v or v.count(".") != 1:
            raise ConfigError(
                f"PerturbSpec.name must be a 'section.field' dotted path into "
                f"ATSConfig (e.g. 'training.learning_rate'), got {v!r}."
            )
        return v

    @field_validator("factor_low", "factor_high")
    @classmethod
    def _validate_factors_positive(cls, v: float) -> float:
        if v <= 0:
            raise ConfigError(
                f"PerturbSpec factor_low/factor_high must be > 0 (they multiply "
                f"the current value), got {v}."
            )
        return v

    @field_validator("resample_probability")
    @classmethod
    def _validate_resample_probability(cls, v: float) -> float:
        if not 0.0 <= v <= 1.0:
            raise ConfigError(
                f"PerturbSpec.resample_probability must be in [0.0, 1.0], got {v}."
            )
        return v

    @model_validator(mode="after")
    def _check_ranges(self) -> PerturbSpec:
        if self.factor_low > self.factor_high:
            raise ConfigError(
                f"PerturbSpec({self.name!r}).factor_low ({self.factor_low}) must "
                f"be <= factor_high ({self.factor_high})."
            )
        if (
            self.min_value is not None
            and self.max_value is not None
            and self.min_value > self.max_value
        ):
            raise ConfigError(
                f"PerturbSpec({self.name!r}).min_value ({self.min_value}) must be "
                f"<= max_value ({self.max_value})."
            )
        return self


class PBTConfig(BaseModel):
    """Top-level configuration for a PBT run.

    steps_per_generation is applied as each member's training.max_steps for
    that generation (see ats.pbt.population.initialize_population) so the LR
    scheduler sees a complete warmup+decay cycle every generation, instead of
    seeing only the early, still-warming-up part of a schedule shaped for the
    full run.
    """

    population_size: int = 10
    generations: int = 10
    steps_per_generation: int = 100
    cull_fraction: float = 0.5
    output_dir: str = "./pbt_runs"
    seed: int = 0
    perturb: list[PerturbSpec] = Field(default_factory=list)

    @field_validator("population_size")
    @classmethod
    def _validate_population_size(cls, v: int) -> int:
        if v < 2:
            raise ConfigError(
                f"pbt.population_size must be >= 2 (PBT needs at least one "
                f"member to cull and one to survive), got {v}."
            )
        return v

    @field_validator("generations")
    @classmethod
    def _validate_generations(cls, v: int) -> int:
        if v < 1:
            raise ConfigError(f"pbt.generations must be >= 1, got {v}.")
        return v

    @field_validator("steps_per_generation")
    @classmethod
    def _validate_steps_per_generation(cls, v: int) -> int:
        if v < 1:
            raise ConfigError(f"pbt.steps_per_generation must be >= 1, got {v}.")
        return v

    @field_validator("cull_fraction")
    @classmethod
    def _validate_cull_fraction(cls, v: float) -> float:
        if not 0.0 < v < 1.0:
            raise ConfigError(f"pbt.cull_fraction must be in (0.0, 1.0), got {v}.")
        return v

    @model_validator(mode="after")
    def _check_cull_leaves_survivors(self) -> PBTConfig:
        n_cull = round(self.population_size * self.cull_fraction)
        n_cull = max(1, min(self.population_size - 1, n_cull))
        if n_cull >= self.population_size:
            raise ConfigError(
                f"pbt.cull_fraction={self.cull_fraction} against "
                f"population_size={self.population_size} would cull the entire "
                f"population, leaving no survivors to exploit. Fix: lower "
                f"cull_fraction or raise population_size."
            )
        return self
