"""Tests for ats.pbt: hyperparameter perturbation, population
initialization, fitness ranking/culling, the exploit/explore step, and the
PBTOrchestrator generation loop (via a FakeRunner so no GPU/DeepSpeed/real
training data is needed -- matching how tests/test_training.py avoids
DeepSpeed with a fake model engine)."""

from __future__ import annotations

import random

import pytest
from pydantic import ValidationError

from ats.config.schema import (
    ATSConfig,
    CheckpointConfig,
    ConfigError,
    DataConfig,
    DataSource,
    ModelConfig,
    TrainingConfig,
)
from ats.pbt.orchestrator import MemberRunner, PBTOrchestrator
from ats.pbt.population import (
    PopulationMember,
    exploit_and_explore,
    initialize_population,
    perturb_config,
    rank_by_fitness,
    select_cull,
)
from ats.pbt.schema import PBTConfig, PerturbSpec


def _base_config(tmp_path, **training_overrides) -> ATSConfig:
    model_config = ModelConfig(
        hidden_size=8,
        num_layers=1,
        num_heads=2,
        num_kv_heads=1,
        intermediate_size=16,
        vocab_size=20,
    )
    training_fields = {
        "max_steps": 999,
        "learning_rate": 1e-3,
        "warmup_steps": 1,
        "weight_decay": 0.1,
    }
    training_fields.update(training_overrides)
    return ATSConfig(
        model=model_config,
        training=TrainingConfig(**training_fields),
        data=DataConfig(sources=[DataSource(path="x.jsonl")], seq_length=8),
        checkpoint=CheckpointConfig(output_dir=str(tmp_path / "ckpts")),
    )


# --------------------------------------------------------------------------
# perturb_config
# --------------------------------------------------------------------------


def test_perturb_config_changes_the_target_field(tmp_path):
    config = _base_config(tmp_path)
    spec = PerturbSpec(
        name="training.learning_rate",
        factor_low=0.5,
        factor_high=0.5,
        resample_probability=0.0,
    )
    new_config = perturb_config(config, [spec], random.Random(0))
    assert new_config.training.learning_rate == pytest.approx(5e-4)
    # Untouched fields survive unchanged.
    assert new_config.training.weight_decay == config.training.weight_decay


def test_perturb_config_clamps_to_min_and_max(tmp_path):
    config = _base_config(tmp_path, weight_decay=0.9)
    spec = PerturbSpec(
        name="training.weight_decay",
        factor_low=2.0,
        factor_high=2.0,
        max_value=1.0,
        resample_probability=0.0,
    )
    new_config = perturb_config(config, [spec], random.Random(0))
    assert new_config.training.weight_decay == pytest.approx(1.0)


def test_perturb_config_min_clamp(tmp_path):
    config = _base_config(tmp_path)
    spec = PerturbSpec(
        name="training.learning_rate",
        factor_low=1e-9,
        factor_high=1e-9,
        min_value=1e-7,
        resample_probability=0.0,
    )
    new_config = perturb_config(config, [spec], random.Random(0))
    assert new_config.training.learning_rate == pytest.approx(1e-7)


def test_perturb_config_resample_draws_within_bounds(tmp_path):
    config = _base_config(tmp_path)
    spec = PerturbSpec(
        name="training.learning_rate",
        min_value=1e-5,
        max_value=1e-4,
        resample_probability=1.0,  # always resample, never multiply
    )
    rng = random.Random(0)
    for _ in range(20):
        new_config = perturb_config(config, [spec], rng)
        assert 1e-5 <= new_config.training.learning_rate <= 1e-4


def test_perturb_config_rejects_non_numeric_field(tmp_path):
    config = _base_config(tmp_path)
    spec = PerturbSpec(name="training.mixed_precision", resample_probability=0.0)
    with pytest.raises(ConfigError):
        perturb_config(config, [spec], random.Random(0))


def test_perturb_config_revalidates_and_rejects_bad_combination(tmp_path):
    # hidden_size (8) must be divisible by num_heads; perturbing num_heads
    # from 2 to 3 breaks that invariant and must fail loudly, not silently
    # produce a broken config (mirrors ats.cli.train's apply_cli_overrides
    # revalidation step).
    config = _base_config(tmp_path)
    spec = PerturbSpec(
        name="model.num_heads",
        factor_low=1.5,
        factor_high=1.5,
        resample_probability=0.0,
    )
    with pytest.raises(ConfigError, match="divisible"):
        perturb_config(config, [spec], random.Random(0))


# --------------------------------------------------------------------------
# initialize_population
# --------------------------------------------------------------------------


def test_initialize_population_creates_requested_size(tmp_path):
    config = _base_config(tmp_path)
    members = initialize_population(
        config,
        population_size=5,
        steps_per_generation=10,
        perturb_specs=[],
        output_dir=str(tmp_path / "pop"),
        seed=0,
    )
    assert len(members) == 5
    assert {m.member_id for m in members} == {0, 1, 2, 3, 4}


def test_initialize_population_forces_max_steps_to_steps_per_generation(tmp_path):
    config = _base_config(tmp_path)  # max_steps=999 in the base config
    members = initialize_population(
        config,
        population_size=3,
        steps_per_generation=25,
        perturb_specs=[],
        output_dir=str(tmp_path / "pop"),
        seed=0,
    )
    assert all(m.config.training.max_steps == 25 for m in members)


def test_initialize_population_gives_each_member_a_distinct_seed(tmp_path):
    # Each member's training.seed is base_config.training.seed + member_id
    # (population.py's initialize_population docstring) -- the `seed`
    # *argument* only drives the perturbation RNG, not training.seed.
    config = _base_config(tmp_path)  # default TrainingConfig.seed == 42
    members = initialize_population(
        config,
        population_size=4,
        steps_per_generation=10,
        perturb_specs=[],
        output_dir=str(tmp_path / "pop"),
        seed=100,
    )
    seeds = [m.config.training.seed for m in members]
    assert seeds == [42, 43, 44, 45]


def test_initialize_population_gives_each_member_its_own_checkpoint_dir(tmp_path):
    config = _base_config(tmp_path)
    members = initialize_population(
        config,
        population_size=3,
        steps_per_generation=10,
        perturb_specs=[],
        output_dir=str(tmp_path / "pop"),
        seed=0,
    )
    dirs = {m.checkpoint_dir for m in members}
    assert len(dirs) == 3
    for m in members:
        assert m.config.checkpoint.output_dir == m.checkpoint_dir


def test_initialize_population_with_perturb_specs_produces_diversity(tmp_path):
    config = _base_config(tmp_path)
    spec = PerturbSpec(
        name="training.learning_rate",
        factor_low=0.5,
        factor_high=1.5,
        resample_probability=0.0,
    )
    members = initialize_population(
        config,
        population_size=8,
        steps_per_generation=10,
        perturb_specs=[spec],
        output_dir=str(tmp_path / "pop"),
        seed=0,
    )
    lrs = {m.config.training.learning_rate for m in members}
    # Distinct per-member seeds -> perturbation shouldn't land on the exact
    # same float for every member.
    assert len(lrs) > 1


# --------------------------------------------------------------------------
# rank_by_fitness / select_cull
# --------------------------------------------------------------------------


def _members_with_fitness(fitnesses: list[float]) -> list[PopulationMember]:
    return [
        PopulationMember(member_id=i, config=None, checkpoint_dir=f"/tmp/m{i}", fitness=f)
        for i, f in enumerate(fitnesses)
    ]


def test_rank_by_fitness_sorts_ascending_lowest_first():
    members = _members_with_fitness([3.0, 1.0, 2.0])
    ranked = rank_by_fitness(members)
    assert [m.member_id for m in ranked] == [1, 2, 0]


def test_rank_by_fitness_raises_if_any_member_unevaluated():
    members = _members_with_fitness([1.0, 2.0])
    members[1].fitness = None
    with pytest.raises(ConfigError):
        rank_by_fitness(members)


def test_select_cull_default_half():
    ranked = _members_with_fitness([0.0, 1.0, 2.0, 3.0])  # already best-first
    survivors, culled = select_cull(ranked, cull_fraction=0.5)
    assert [m.member_id for m in survivors] == [0, 1]
    assert [m.member_id for m in culled] == [2, 3]


def test_select_cull_never_culls_everyone():
    ranked = _members_with_fitness([0.0, 1.0])
    survivors, culled = select_cull(ranked, cull_fraction=0.99)
    assert len(survivors) >= 1
    assert len(culled) >= 1
    assert len(survivors) + len(culled) == 2


def test_select_cull_never_culls_nobody_even_at_tiny_fraction():
    ranked = _members_with_fitness([0.0, 1.0, 2.0, 3.0, 4.0])
    survivors, culled = select_cull(ranked, cull_fraction=0.01)
    assert len(culled) >= 1


# --------------------------------------------------------------------------
# exploit_and_explore
# --------------------------------------------------------------------------


def test_exploit_and_explore_clones_survivor_config_into_culled_member(tmp_path):
    config = _base_config(tmp_path, learning_rate=1e-3)
    loser_own_checkpoint = config.checkpoint.model_copy(
        update={"output_dir": "/tmp/loser"}
    )
    loser_config = config.model_copy(update={"checkpoint": loser_own_checkpoint})
    winner_config = config.model_copy(
        update={
            "training": config.training.model_copy(update={"learning_rate": 5e-4}),
            "checkpoint": config.checkpoint.model_copy(
                update={"output_dir": "/tmp/survivor"}
            ),
        }
    )
    survivor = PopulationMember(
        member_id=0, config=winner_config, checkpoint_dir="/tmp/survivor"
    )
    loser = PopulationMember(
        member_id=1, config=loser_config, checkpoint_dir="/tmp/loser", fitness=999.0
    )
    assignments = exploit_and_explore(
        [survivor], [loser], perturb_specs=[], rng=random.Random(0)
    )
    assert len(assignments) == 1
    cloned_member, source = assignments[0]
    assert cloned_member is loser
    assert source is survivor
    assert cloned_member.config.training.learning_rate == pytest.approx(5e-4)
    # The culled member keeps its OWN checkpoint dir, not the source's --
    # the orchestrator needs a stable destination to write this member's
    # next checkpoint to.
    assert cloned_member.config.checkpoint.output_dir == "/tmp/loser"
    assert cloned_member.parent_id == 0


def test_exploit_and_explore_applies_perturbation(tmp_path):
    config = _base_config(tmp_path, learning_rate=1e-3)
    survivor = PopulationMember(member_id=0, config=config, checkpoint_dir="/tmp/s")
    loser = PopulationMember(member_id=1, config=config, checkpoint_dir="/tmp/l")
    spec = PerturbSpec(
        name="training.learning_rate",
        factor_low=0.5,
        factor_high=0.5,
        resample_probability=0.0,
    )
    exploit_and_explore([survivor], [loser], perturb_specs=[spec], rng=random.Random(0))
    assert loser.config.training.learning_rate == pytest.approx(5e-4)


def test_exploit_and_explore_raises_with_no_survivors(tmp_path):
    config = _base_config(tmp_path)
    loser = PopulationMember(member_id=1, config=config, checkpoint_dir="/tmp/l")
    with pytest.raises(ConfigError):
        exploit_and_explore([], [loser], perturb_specs=[], rng=random.Random(0))


# --------------------------------------------------------------------------
# PBTOrchestrator (via a fake MemberRunner -- no training/eval actually runs)
# --------------------------------------------------------------------------


class FakeRunner(MemberRunner):
    """Fitness is a fixed function of member_id (lower id = better), so
    exactly who gets culled/survives each generation is deterministic and
    assertable, regardless of how many generations run."""

    def __init__(self):
        self.train_calls: list[tuple[int, int, int, str | None]] = []

    def train(self, member, generation, steps, init_weights):
        self.train_calls.append((member.member_id, generation, steps, init_weights))
        return f"/fake/{member.member_id}/gen_{generation}/step_{steps}"

    def evaluate(self, member, checkpoint_dir) -> float:
        return float(member.member_id)


def _orchestrator(tmp_path, population_size=6, generations=3, cull_fraction=0.5):
    config = _base_config(tmp_path)
    pbt_config = PBTConfig(
        population_size=population_size,
        generations=generations,
        steps_per_generation=5,
        cull_fraction=cull_fraction,
        output_dir=str(tmp_path / "pbt"),
        seed=7,
        perturb=[
            PerturbSpec(
                name="training.learning_rate",
                factor_low=0.8,
                factor_high=1.2,
                min_value=1e-7,
            )
        ],
    )
    runner = FakeRunner()
    return PBTOrchestrator(config, pbt_config, runner=runner), runner


def test_orchestrator_generation_0_trains_every_member_from_scratch(tmp_path):
    orch, runner = _orchestrator(tmp_path, generations=1)
    orch.run_generation()
    gen0_calls = [c for c in runner.train_calls if c[1] == 0]
    assert len(gen0_calls) == 6
    assert all(init_weights is None for (_id, _gen, _steps, init_weights) in gen0_calls)


def test_orchestrator_culls_bottom_half_every_generation(tmp_path):
    orch, _runner = _orchestrator(tmp_path, population_size=6, generations=1)
    summary = orch.run_generation()
    # member ids 0..5, fitness == member_id (lower is better) -> worst three
    # (3, 4, 5) are culled, best three (0, 1, 2) survive.
    assert sorted(summary["culled"]) == [3, 4, 5]
    assert sorted(summary["survivors"]) == [0, 1, 2]


def test_orchestrator_next_generation_survivors_reload_their_own_checkpoint(tmp_path):
    orch, runner = _orchestrator(tmp_path, population_size=6, generations=2)
    orch.run_generation()  # generation 0
    orch.run_generation()  # generation 1
    gen1_calls = {c[0]: c[3] for c in runner.train_calls if c[1] == 1}
    # Survivor 0's generation-1 init_weights must be exactly its OWN
    # generation-0 checkpoint, not another member's.
    assert gen1_calls[0] == "/fake/0/gen_0/step_5"


def test_orchestrator_culled_members_load_a_survivors_checkpoint(tmp_path):
    orch, runner = _orchestrator(tmp_path, population_size=6, generations=2)
    orch.run_generation()
    orch.run_generation()
    gen1_calls = {c[0]: c[3] for c in runner.train_calls if c[1] == 1}
    # Member 5 was culled after generation 0 (worst fitness); its
    # generation-1 init_weights must point at ONE of the survivors'
    # generation-0 checkpoints (0, 1, or 2), never its own or a fellow
    # loser's (3, 4).
    assert gen1_calls[5] in {
        "/fake/0/gen_0/step_5",
        "/fake/1/gen_0/step_5",
        "/fake/2/gen_0/step_5",
    }


def test_orchestrator_run_executes_all_generations_and_returns_best(tmp_path):
    orch, runner = _orchestrator(tmp_path, population_size=6, generations=4)
    best = orch.run()
    assert best.member_id == 0
    assert best.fitness == 0.0
    assert orch.generation == 4
    assert len(orch.history) == 4
    # 6 members x 4 generations.
    assert len(runner.train_calls) == 24


def test_orchestrator_rejects_population_size_of_one(tmp_path):
    with pytest.raises(ValidationError, match="population_size"):
        PBTConfig(population_size=1, output_dir=str(tmp_path / "pbt"))


def test_orchestrator_history_records_per_generation_summaries(tmp_path):
    orch, _runner = _orchestrator(tmp_path, population_size=4, generations=2)
    orch.run()
    assert len(orch.history) == 2
    for entry in orch.history:
        assert set(entry.keys()) == {"generation", "fitness", "culled", "survivors"}
