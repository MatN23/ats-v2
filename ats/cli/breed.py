#!/usr/bin/env python
"""Entry point: python -m ats.cli.breed --config configs/debug.yaml \
    --population-size 10 --generations 5 --steps-per-generation 100 \
    --cull-fraction 0.5 --output-dir ./pbt_runs

Runs Population Based Training ("breeding"): trains `population_size`
independent copies of the model in `--config` side by side. After every
`steps_per_generation` steps, each member is evaluated on its own held-out
data (config.data.sources), the bottom `cull_fraction` are culled, and each
culled member's weights + a perturbed copy of a surviving winner's
hyperparameters take its place.

Cost warning: this trains population_size independent copies of the model,
so total compute is roughly population_size times a single run of the same
step count. Only use this with small model configs (configs/debug.yaml,
configs/125m.yaml, configs/350m.yaml, ...) -- see ats/pbt/orchestrator.py's
module docstring for the full reasoning. Nothing here stops you from
pointing --config at configs/7b.yaml or configs/70b.yaml, but doing so
multiplies an already-large job by population_size.

This CLI does not support resuming an interrupted breeding run across
process restarts (a fresh call always starts a new population at
generation 0); each member's own per-generation training does still
checkpoint normally under --output-dir.
"""

from __future__ import annotations

import argparse
import sys

from ats.config.loader import load_config
from ats.config.schema import ConfigError
from ats.pbt.orchestrator import PBTOrchestrator
from ats.pbt.schema import PBTConfig, PerturbSpec
from ats.utils.logging_utils import get_logger, setup_logging

logger = get_logger("ats.cli.breed")

# Defaults mirror the fields Population Based Training was originally
# demonstrated on (learning rate, regularization strength): perturbing
# architecture fields (hidden_size, use_moe, ...) would break the
# weights-only transplant between members, since source and destination
# would no longer have matching parameter shapes -- see
# ats.training.checkpoint.load_initial_weights.
_DEFAULT_PERTURB_SPECS = {
    "learning-rate": PerturbSpec(
        name="training.learning_rate", factor_low=0.8, factor_high=1.2, min_value=1e-7
    ),
    "weight-decay": PerturbSpec(
        name="training.weight_decay",
        factor_low=0.8,
        factor_high=1.2,
        min_value=0.0,
        max_value=1.0,
    ),
    "dropout": PerturbSpec(
        name="model.dropout",
        factor_low=0.8,
        factor_high=1.2,
        min_value=0.0,
        max_value=0.9,
    ),
}


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run Population Based Training ('breeding') over a population of "
        "ats-v2 model configs.",
    )
    parser.add_argument(
        "--config", required=True, help="Base YAML config every member starts from."
    )
    parser.add_argument(
        "--population-size",
        type=int,
        default=10,
        help="Number of members trained side by side (default: 10).",
    )
    parser.add_argument(
        "--generations",
        type=int,
        default=10,
        help="Number of train -> evaluate -> cull/clone rounds (default: 10).",
    )
    parser.add_argument(
        "--steps-per-generation",
        type=int,
        default=100,
        help="Training steps each member runs per generation (default: 100).",
    )
    parser.add_argument(
        "--cull-fraction",
        type=float,
        default=0.5,
        help="Fraction of the population replaced each generation (default: 0.5, "
        "i.e. the bottom half).",
    )
    parser.add_argument(
        "--output-dir",
        default="./pbt_runs",
        help="Where each member's configs and checkpoints are written (default: "
        "./pbt_runs).",
    )
    parser.add_argument(
        "--seed", type=int, default=0, help="Base RNG seed for the whole run."
    )
    for flag_name, spec in _DEFAULT_PERTURB_SPECS.items():
        parser.add_argument(
            f"--no-perturb-{flag_name}",
            dest=f"perturb_{flag_name.replace('-', '_')}",
            action="store_false",
            default=True,
            help=f"Don't perturb {spec.name} during the explore step (perturbed by "
            f"default).",
        )
    parser.add_argument(
        "--no-perturb",
        action="store_true",
        help="Disable all default hyperparameter perturbation -- members still "
        "start with different training.seed values, but the explore step becomes "
        "a no-op clone instead of a perturbed clone.",
    )
    return parser


def _resolve_perturb_specs(args: argparse.Namespace) -> list[PerturbSpec]:
    if args.no_perturb:
        return []
    specs = []
    for flag_name, spec in _DEFAULT_PERTURB_SPECS.items():
        if getattr(args, f"perturb_{flag_name.replace('-', '_')}"):
            specs.append(spec)
    return specs


def main(argv: list[str] | None = None) -> int:
    setup_logging()
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    try:
        base_config = load_config(args.config)
        perturb_specs = _resolve_perturb_specs(args)
        pbt_config = PBTConfig(
            population_size=args.population_size,
            generations=args.generations,
            steps_per_generation=args.steps_per_generation,
            cull_fraction=args.cull_fraction,
            output_dir=args.output_dir,
            seed=args.seed,
            perturb=perturb_specs,
        )
    except ConfigError as exc:
        logger.error("Config error: %s", exc)
        return 1

    logger.info(
        "Starting breeding run: population_size=%d generations=%d "
        "steps_per_generation=%d cull_fraction=%.2f perturbing=%s",
        pbt_config.population_size,
        pbt_config.generations,
        pbt_config.steps_per_generation,
        pbt_config.cull_fraction,
        [s.name for s in perturb_specs] or "nothing",
    )

    orchestrator = PBTOrchestrator(base_config, pbt_config)
    try:
        best = orchestrator.run()
    except (ConfigError, RuntimeError) as exc:
        logger.error("Breeding run failed: %s", exc)
        return 1

    logger.info(
        "Breeding complete after %d generations. Best member: member_%d "
        "(fitness=%.4f), checkpoint at %s",
        pbt_config.generations,
        best.member_id,
        best.fitness,
        best.checkpoint_dir,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
