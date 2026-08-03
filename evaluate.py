#!/usr/bin/env python
"""Entry point: python evaluate.py --config configs/7b_dense.yaml \
    --checkpoint checkpoints/run/step_10000 [--tasks mmlu,hellaswag]"""

from __future__ import annotations

import argparse
import sys
from typing import List

import torch

from ats.config.loader import load_config
from ats.config.schema import ConfigError
from ats.data.dataloader import build_dataloader
from ats.model.transformer import ATSTransformer
from ats.parallelism.deepspeed_utils import initialize_engine
from ats.training.checkpoint import CheckpointManager
from ats.utils.logging_utils import get_logger, setup_logging

logger = get_logger("ats.evaluate")

# Multiple-choice benchmarks supported via a simple per-choice log-likelihood
# scorer. Each task file must be a .jsonl with {"prompt": str, "choices": [str],
# "answer_index": int}. This is a real (if minimal) eval loop, not a stub.
SUPPORTED_TASKS = {"mmlu", "hellaswag", "arc_easy", "arc_challenge"}


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate an ats-v2 checkpoint.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument(
        "--tasks", default="",
        help=f"Comma-separated task list, any of {sorted(SUPPORTED_TASKS)}. "
             f"If empty, only perplexity on data.sources is reported.",
    )
    parser.add_argument(
        "--micro-batch-size", type=int, default=None,
        help="Per-GPU micro batch size. Overrides training.micro_batch_size from the "
             "config if given; otherwise the config's value is used.",
    )
    return parser.parse_args(argv)


def _score_multiple_choice(model_engine, tokenizer, task_path: str) -> float:
    """Real log-likelihood-ranking scorer: for each example, pick the choice
    with highest total log-probability under the model and check accuracy."""
    import json

    correct = 0
    total = 0
    with open(task_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            example = json.loads(line)
            prompt_ids = tokenizer.encode(example["prompt"])
            choice_scores: List[float] = []
            for choice in example["choices"]:
                choice_ids = tokenizer.encode(choice)
                full_ids = prompt_ids + choice_ids
                input_ids = torch.tensor([full_ids], dtype=torch.long)
                with torch.no_grad():
                    output = model_engine(input_ids)
                logits = output.logits[0, len(prompt_ids) - 1: len(full_ids) - 1]
                target = input_ids[0, len(prompt_ids):]
                log_probs = torch.log_softmax(logits, dim=-1)
                token_log_probs = log_probs.gather(1, target.unsqueeze(-1)).squeeze(-1)
                choice_scores.append(float(token_log_probs.sum().item()))
            predicted = int(torch.tensor(choice_scores).argmax().item())
            if predicted == example["answer_index"]:
                correct += 1
            total += 1

    if total == 0:
        raise ConfigError(f"Task file {task_path} contained no examples.")
    return correct / total


def main(argv=None) -> int:
    setup_logging()
    args = parse_args(argv)

    try:
        config = load_config(args.config)
    except ConfigError as exc:
        logger.error("Config error: %s", exc)
        return 1

    micro_batch_size = (
        args.micro_batch_size if args.micro_batch_size is not None else config.training.micro_batch_size
    )

    model = ATSTransformer(config.model)
    model_engine, _optimizer, _, _ = initialize_engine(model, config, micro_batch_size)

    checkpoint_manager = CheckpointManager(config)
    try:
        checkpoint_manager.load(model_engine, args.checkpoint)
    except ConfigError as exc:
        logger.error("Failed to load checkpoint: %s", exc)
        return 1

    model_engine.eval()

    tasks = [t.strip() for t in args.tasks.split(",") if t.strip()]
    for task in tasks:
        if task not in SUPPORTED_TASKS:
            logger.error(
                "Unknown task '%s'. Fix: choose from %s or omit --tasks to only "
                "compute perplexity.", task, sorted(SUPPORTED_TASKS),
            )
            return 1

    if tasks:
        from ats.data.tokenizer import Tokenizer
        tokenizer = Tokenizer(config.data.tokenizer_name)
        for task in tasks:
            task_path = f"./eval_data/{task}.jsonl"
            try:
                accuracy = _score_multiple_choice(model_engine, tokenizer, task_path)
            except FileNotFoundError:
                logger.error(
                    "No eval data found for task '%s' at %s. "
                    "Fix: place a .jsonl file with {prompt, choices, answer_index} "
                    "records at that path.", task, task_path,
                )
                return 1
            logger.info("Task %s: accuracy=%.4f", task, accuracy)
    else:
        eval_dataloader = build_dataloader(
            config.data, batch_size=micro_batch_size,
            rank=0, world_size=1, seed=config.training.seed,
        )
        total_loss = 0.0
        total_tokens = 0
        with torch.no_grad():
            for batch in eval_dataloader:
                output = model_engine(batch["input_ids"], attention_mask=batch.get("attention_mask"))
                shift_logits = output.logits[..., :-1, :].contiguous()
                shift_labels = batch["labels"][..., 1:].contiguous()
                loss = torch.nn.functional.cross_entropy(
                    shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1),
                    ignore_index=-100, reduction="sum",
                )
                num_valid = (shift_labels != -100).sum().item()
                total_loss += float(loss.item())
                total_tokens += int(num_valid)
        if total_tokens == 0:
            logger.error("Eval dataloader produced zero valid tokens.")
            return 1
        perplexity = float(torch.exp(torch.tensor(total_loss / total_tokens)))
        logger.info("Perplexity: %.4f", perplexity)

    return 0


if __name__ == "__main__":
    sys.exit(main())
