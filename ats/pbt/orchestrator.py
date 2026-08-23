"""Orchestrates a PBT ("breeding") run: repeatedly trains every population
member for one generation, evaluates it, culls the worst
`pbt_config.cull_fraction`, and clones each culled member's weights +
perturbed hyperparameters from a surviving winner -- "snap the bottom
performers out of existence and replace them with the best performer" is
exactly the exploit/explore step from Jaderberg et al., 2017 ("Population
Based Training of Neural Networks").

Cost, stated plainly: this multiplies training cost by population_size.
Every member trains independently for steps_per_generation steps every
generation, so a 10-member, 5-generation run costs roughly 10x what a
single run of the same total step count would. This module is intended for
small models (configs/debug.yaml, configs/125m.yaml, configs/350m.yaml)
where that multiplier is affordable on one machine -- it deliberately does
NOT attempt to schedule population members across a cluster, and using it
with configs/7b.yaml or larger will just multiply an already-large job by
population_size.

Design choice worth calling out: every generation is a fresh, independent
training segment. A culled member doesn't resume its old optimizer state
(Adam's moments were accumulated under hyperparameters this member no
longer has) and a surviving member doesn't either (for consistency, and
because carrying DeepSpeed's own checkpoint/config_hash matching through a
run where sibling members' hyperparameters keep changing is far more
failure-prone than a plain weights-only reload). Every member's weights are
loaded fresh each generation via `ats.cli.train --init-weights` (see
ats.training.checkpoint.load_initial_weights), and the LR scheduler runs a
complete warmup+decay cycle over steps_per_generation every time -- see
ats.pbt.population.initialize_population's docstring.
"""

from __future__ import annotations

import random
import subprocess
import sys
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import yaml

from ats.config.schema import ATSConfig
from ats.pbt.population import (
    PopulationMember,
    exploit_and_explore,
    initialize_population,
    rank_by_fitness,
    select_cull,
)
from ats.pbt.schema import PBTConfig
from ats.utils.logging_utils import get_logger

logger = get_logger("ats.pbt.orchestrator")


def _write_config_yaml(config: ATSConfig, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(config.model_dump(), f, sort_keys=False)


def _latest_checkpoint_dir(output_dir: Path) -> Path:
    if not output_dir.exists():
        raise RuntimeError(
            f"Expected a checkpoint output directory at {output_dir} after "
            f"training, but it doesn't exist. The training subprocess may have "
            f"failed before writing any checkpoint."
        )
    step_dirs = sorted(
        (p for p in output_dir.glob("step_*") if p.is_dir()),
        key=lambda p: int(p.name.split("_")[1]),
    )
    if not step_dirs:
        raise RuntimeError(
            f"No step_* checkpoint directories found under {output_dir}. Check "
            f"that training.save_every is <= steps_per_generation, or the "
            f"training subprocess never got far enough to save a checkpoint."
        )
    return step_dirs[-1]


class MemberRunner(ABC):
    """Abstraction over "run one population member's training segment and
    evaluate its resulting checkpoint", so PBTOrchestrator's generation loop
    can be unit-tested with a fast fake instead of requiring a GPU, DeepSpeed,
    and real training data for every test."""

    @abstractmethod
    def train(
        self,
        member: PopulationMember,
        generation: int,
        steps: int,
        init_weights: str | None,
    ) -> str:
        """Trains `member` for `steps` steps this generation, loading initial
        weights from `init_weights` (a checkpoint dir or .safetensors file,
        or None for a fresh random init -- only valid at generation 0).
        Returns the path to the resulting checkpoint directory."""

    @abstractmethod
    def evaluate(self, member: PopulationMember, checkpoint_dir: str) -> float:
        """Returns this member's fitness (lower is better -- perplexity or
        held-out loss) for the checkpoint at `checkpoint_dir`."""


class ATSMemberRunner(MemberRunner):
    """Real MemberRunner: shells out to `python -m ats.cli.train` per member
    per generation (so members are isolated processes, matching how ats-v2
    is normally launched) and computes fitness in-process via
    ats.training.perplexity.compute_perplexity."""

    def __init__(self, python_executable: str | None = None) -> None:
        self.python_executable = python_executable or sys.executable

    def train(
        self,
        member: PopulationMember,
        generation: int,
        steps: int,
        init_weights: str | None,
    ) -> str:
        gen_output_dir = Path(member.checkpoint_dir) / f"gen_{generation}"
        config = member.config.model_copy(
            update={
                "checkpoint": member.config.checkpoint.model_copy(
                    update={"output_dir": str(gen_output_dir)}
                )
            }
        )
        config_path = Path(member.checkpoint_dir) / f"config_gen_{generation}.yaml"
        _write_config_yaml(config, config_path)

        cmd = [
            self.python_executable,
            "-m",
            "ats.cli.train",
            "--config",
            str(config_path),
            "--max-steps",
            str(steps),
        ]
        if init_weights is not None:
            cmd += ["--init-weights", str(init_weights)]

        logger.info(
            "member %d generation %d: %s", member.member_id, generation, " ".join(cmd)
        )
        result = subprocess.run(cmd, check=False)
        if result.returncode != 0:
            raise RuntimeError(
                f"Training failed for member {member.member_id}, generation "
                f"{generation} (exit code {result.returncode}). See the "
                f"subprocess output above for the actual error."
            )
        return str(_latest_checkpoint_dir(gen_output_dir))

    def evaluate(self, member: PopulationMember, checkpoint_dir: str) -> float:
        from ats.training.perplexity import compute_perplexity

        perplexity, _client_state = compute_perplexity(member.config, checkpoint_dir)
        return perplexity


class PBTOrchestrator:
    """Runs a full PBT breeding session: `pbt_config.generations` rounds of
    training every member for `pbt_config.steps_per_generation` steps,
    evaluating, and culling/cloning the bottom `pbt_config.cull_fraction`.
    """

    def __init__(
        self,
        base_config: ATSConfig,
        pbt_config: PBTConfig,
        runner: MemberRunner | None = None,
    ) -> None:
        self.base_config = base_config
        self.pbt_config = pbt_config
        self.runner = runner if runner is not None else ATSMemberRunner()
        self.rng = random.Random(pbt_config.seed)

        self.members: list[PopulationMember] = initialize_population(
            base_config=base_config,
            population_size=pbt_config.population_size,
            steps_per_generation=pbt_config.steps_per_generation,
            perturb_specs=pbt_config.perturb,
            output_dir=pbt_config.output_dir,
            seed=pbt_config.seed,
        )
        # None at generation 0 (fresh random init); after that, the path
        # (checkpoint dir or .safetensors file) each member should load its
        # weights from before training its next generation.
        self._pending_init_weights: dict[int, str | None] = {
            m.member_id: None for m in self.members
        }
        # The latest checkpoint produced by each member_id's most recently
        # completed generation, keyed by member_id (not by member identity,
        # since a culled member's slot is reused, not deleted).
        self._latest_checkpoint: dict[int, str] = {}
        self.generation = 0
        self.history: list[dict[str, Any]] = []

    def run(self) -> PopulationMember:
        """Runs every remaining generation and returns the single
        best-fitness member at the end."""
        while self.generation < self.pbt_config.generations:
            self.run_generation()
        return rank_by_fitness(self.members)[0]

    def run_generation(self) -> dict[str, Any]:
        """Runs exactly one generation (train + evaluate + cull/clone) and
        returns a summary dict, also appended to self.history."""
        generation = self.generation

        for member in self.members:
            init_weights = self._pending_init_weights[member.member_id]
            checkpoint_dir = self.runner.train(
                member,
                generation=generation,
                steps=self.pbt_config.steps_per_generation,
                init_weights=init_weights,
            )
            self._latest_checkpoint[member.member_id] = checkpoint_dir
            member.fitness = self.runner.evaluate(member, checkpoint_dir)
            member.generation = generation + 1
            member.history.append(
                {
                    "generation": generation,
                    "fitness": member.fitness,
                    "parent_id": member.parent_id,
                }
            )

        ranked = rank_by_fitness(self.members)
        survivors, culled = select_cull(ranked, self.pbt_config.cull_fraction)
        assignments = exploit_and_explore(
            survivors, culled, self.pbt_config.perturb, self.rng
        )

        next_init_weights: dict[int, str | None] = {
            member.member_id: self._latest_checkpoint[member.member_id]
            for member in survivors
        }
        for member, source in assignments:
            next_init_weights[member.member_id] = self._latest_checkpoint[
                source.member_id
            ]
        self._pending_init_weights = next_init_weights

        summary = {
            "generation": generation,
            "fitness": {m.member_id: m.fitness for m in self.members},
            "culled": [member.member_id for member, _source in assignments],
            "survivors": [member.member_id for member in survivors],
        }
        logger.info(
            "Generation %d complete: culled %s, best fitness %.4f (member %d)",
            generation,
            summary["culled"],
            ranked[0].fitness,
            ranked[0].member_id,
        )
        self.history.append(summary)
        self.generation += 1
        return summary
