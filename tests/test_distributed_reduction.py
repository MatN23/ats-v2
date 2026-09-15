"""Real multi-process regression test for BUG-107.

Runs two actual processes over the gloo backend on CPU -- no mocking of
torch.distributed -- and checks that the numbers fed to AdaptiveController
are identical on both ranks even when each rank's micro-batch loss differs.

Pre-fix, Trainer.train_step passed its own rank-local `ce_loss` straight
into TrainingMetrics, so rank 0 and rank 1 could take different
AdaptiveActions: different learning rates on ranks sharing all-reduced
gradients, and a `training_halt` that raises on one rank while the others
block forever on the next collective.
"""

from __future__ import annotations

import os

import pytest
import torch

pytestmark = pytest.mark.skipif(
    not (torch.distributed.is_available() and torch.distributed.is_gloo_available()),
    reason="torch.distributed with the gloo backend is required",
)


def _worker(rank: int, world_size: int, queue) -> None:  # pragma: no cover - subprocess
    import traceback

    import torch.distributed as dist

    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = "29517"
    dist.init_process_group(
        "gloo", rank=rank, world_size=world_size, init_method="env://"
    )
    try:
        from ats.training.trainer import _distributed_mean, _reduce_expert_utilization

        # Deliberately divergent per-rank values.
        local_loss = torch.tensor(1.0 if rank == 0 else 5.0)
        reduced = _distributed_mean(local_loss.clone())

        local_util = (
            {0: 1.0, 1: 0.0} if rank == 0 else {0: 0.0, 1: 1.0}
        )
        reduced_util = _reduce_expert_utilization(local_util, torch.device("cpu"))

        # And confirm the controller actually agrees given identical input.
        from ats.config.schema import AdaptiveConfig
        from ats.training.adaptive_controller import AdaptiveController, TrainingMetrics

        controller = AdaptiveController(
            AdaptiveConfig(enabled=True, grad_norm_threshold=1.0)
        )
        # grad_norm above the threshold on both ranks -> both must emit the
        # same emergency action.
        action = controller.step(
            TrainingMetrics(
                step=0,
                loss=float(reduced.item()),
                grad_norm=10.0,
                learning_rate=1e-3,
                expert_utilization=reduced_util,
            )
        )
        queue.put(
            (
                rank,
                float(reduced.item()),
                reduced_util,
                None if action is None else action.type,
            )
        )
    except Exception:
        queue.put((rank, None, None, "ERROR:\n" + traceback.format_exc()))
    finally:
        dist.destroy_process_group()


def test_controller_inputs_are_identical_across_ranks():
    import multiprocessing as mp

    # fork, not spawn: under pytest, spawn re-executes the pytest entry
    # point in the child (multiprocessing reconstructs __main__ from
    # sys.argv[0]), which fails before the worker ever runs. fork is
    # available on Linux and safe here because nothing CUDA is touched.
    if "fork" not in mp.get_all_start_methods():  # pragma: no cover
        pytest.skip("the fork start method is required for this test")
    ctx = mp.get_context("fork")
    queue = ctx.Queue()
    world_size = 2
    procs = [
        ctx.Process(target=_worker, args=(rank, world_size, queue))
        for rank in range(world_size)
    ]
    for p in procs:
        p.start()
    results = {}
    try:
        for _ in range(world_size):
            rank, loss, util, action = queue.get(timeout=120)
            results[rank] = (loss, util, action)
    finally:
        for p in procs:
            p.join(timeout=60)
            if p.is_alive():  # pragma: no cover - only on a hang
                p.terminate()

    assert set(results) == {0, 1}
    loss0, util0, action0 = results[0]
    loss1, util1, action1 = results[1]

    assert loss0 == pytest.approx(3.0), "mean of 1.0 and 5.0 should be 3.0"
    assert loss0 == pytest.approx(loss1), (
        f"ranks disagree on the loss fed to AdaptiveController: "
        f"{loss0} vs {loss1} (BUG-107)"
    )
    for key in util0:
        assert util0[key] == pytest.approx(util1[key]), (
            "ranks disagree on expert utilization fed to AdaptiveController"
        )
        assert util0[key] == pytest.approx(0.5)
    for rank, (_loss, _util, action) in results.items():
        assert action is None or not str(action).startswith("ERROR"), (
            f"rank {rank} raised:\n{action}"
        )
    assert action0 == action1 == "emergency_lr_cut", (
        "ranks took different adaptive actions from the same step"
    )


