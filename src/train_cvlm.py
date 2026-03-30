# Full CVLM training entrypoint (reference: src/train.py — DDP, accum, cosine LR, ckpt).

from __future__ import annotations

import argparse
import os
import random
import time
from typing import List, Optional

import numpy as np
import torch
import torch.distributed as dist
from cvlm_dataset import CvlmTrainDataset, make_collate_fn
from modeling import CVLM, ModelArguments, TrainingArguments
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from train_utils import (
    AverageMeter,
    cleanup_distributed,
    find_latest_checkpoint,
    load_cvlm_checkpoint,
    save_cvlm_checkpoint,
    setup_distributed,
    unwrap_model,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train CVLM")
    p.add_argument("--output_dir", type=str, required=True)
    p.add_argument(
        "--embeddings_path",
        type=str,
        default=os.path.join(
            os.path.dirname(__file__),
            "..",
            "data",
            "dataset_100_samples",
            "embeddings.npy",
        ),
    )
    p.add_argument("--dataset_name", type=str, default="sggetao/PwC")
    p.add_argument("--dataset_split", type=str, default="train")
    p.add_argument("--model_name_or_path", type=str, default=None)
    p.add_argument("--vision_encoder_name", type=str, default=None)
    p.add_argument(
        "--max_prompt_len",
        type=int,
        default=512,
        help="Keep only samples with prompt token count <= this (no truncation).",
    )
    p.add_argument(
        "--max_answer_len",
        type=int,
        default=2048,
        help="Keep only samples with answer token count <= this (no truncation).",
    )
    p.add_argument(
        "--max_vision_len",
        type=int,
        default=512,
        help="Keep only samples with vision seq length (embedding rows) <= this (no truncation).",
    )
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--batch_size", type=int, default=2)
    p.add_argument("--lr", type=float, default=1e-5)
    p.add_argument("--adamw_beta1", type=float, default=0.9)
    p.add_argument("--adamw_beta2", type=float, default=0.95)
    p.add_argument("--grad_clip", type=float, default=1.0)
    p.add_argument("--gradient_accumulation_steps", type=int, default=1)
    p.add_argument("--min_lr", type=float, default=0.0)
    p.add_argument("--enable_warmup", action="store_true")
    p.add_argument("--warmup_ratio", type=int, default=0)
    p.add_argument("--warmup_steps", type=int, default=0)
    p.add_argument("--log_interval", type=int, default=10)
    p.add_argument(
        "--save_interval_steps",
        type=int,
        default=500,
        help=(
            "Save every N optimizer steps (sync points). "
            "Set to 0 to save only at the end of each epoch (no extra mid-epoch saves)."
        ),
    )
    p.add_argument(
        "--resume_dir",
        type=str,
        default="",
        help="Directory with model_step_*.safetensors + trainer_step_*.pt; picks latest.",
    )
    p.add_argument("--restore_from", type=str, default="", help="HF-style; sets TrainingArguments.restore_from for init only.")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--no_bf16",
        action="store_true",
        help="Disable bf16 (use float32 on CUDA).",
    )
    p.add_argument(
        "--tensorboard_dir",
        type=str,
        default="",
        help="Directory for TensorBoard logs. Defaults to <output_dir>/tb if not set.",
    )
    return p.parse_args()


def set_seed(seed: int, rank: int) -> None:
    random.seed(seed + rank)
    np.random.seed(seed + rank)
    torch.manual_seed(seed + rank)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed + rank)


def move_batch(batch: dict, device: torch.device, dtype: torch.dtype) -> dict:
    out = {}
    for k, v in batch.items():
        if k == "input_embeds":
            out[k] = v.to(device=device, dtype=dtype)
        else:
            out[k] = v.to(device=device)
    return out


def main() -> None:
    args = parse_args()
    rank, world_size, local_rank, device, use_ddp = setup_distributed()
    is_master = rank == 0
    set_seed(args.seed, rank)

    use_bf16 = device.type == "cuda" and not args.no_bf16

    model_args = ModelArguments()
    if args.model_name_or_path:
        model_args.model_name_or_path = args.model_name_or_path
    if args.vision_encoder_name:
        model_args.vision_encoder_name = args.vision_encoder_name

    training_args = TrainingArguments(output_dir=args.output_dir)
    training_args.bf16 = bool(use_bf16)
    training_args.restore_from = args.restore_from or ""

    model = CVLM(model_args, training_args, None)
    model.to(device)

    tok_pad = model.tokenizer.pad_token_id
    if tok_pad is None:
        raise ValueError("Tokenizer must define pad_token for batching")

    collate = make_collate_fn(tok_pad)
    dataset = CvlmTrainDataset(
        embeddings_path=os.path.normpath(args.embeddings_path),
        hf_dataset_name=args.dataset_name,
        hf_split=args.dataset_split,
        tokenizer_name=model_args.model_name_or_path,
        max_prompt_len=args.max_prompt_len,
        max_answer_len=args.max_answer_len,
        max_vision_len=args.max_vision_len,
    )

    sampler: Optional[DistributedSampler] = None
    if use_ddp:
        sampler = DistributedSampler(dataset, shuffle=True, drop_last=False)

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=sampler is None,
        sampler=sampler,
        num_workers=0,
        collate_fn=collate,
        drop_last=False,
    )

    if use_ddp:
        model = DDP(
            model,
            device_ids=[local_rank] if device.type == "cuda" else None,
            find_unused_parameters=True,
        )

    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(
        trainable,
        lr=args.lr,
        betas=(args.adamw_beta1, args.adamw_beta2),
        weight_decay=0.0,
    )
    grad_clip = args.grad_clip

    grad_accum_steps = args.gradient_accumulation_steps
    batch_size_per_rank = args.batch_size
    eff_batch = args.batch_size * world_size * grad_accum_steps

    steps_per_epoch = max(len(loader) // grad_accum_steps, 1)
    total_steps = max(steps_per_epoch * args.epochs, 1)

    if args.enable_warmup:
        warmup_steps = int(
            max(args.warmup_ratio * total_steps // 100, args.warmup_steps)
        )
    else:
        warmup_steps = 0

    init_lrs: List[float] = [pg["lr"] for pg in optimizer.param_groups]
    lr_scheduler = CosineAnnealingLR(
        optimizer, T_max=total_steps, eta_min=args.min_lr
    )

    start_epoch = 1
    global_step = 1
    resume_from_dir = args.resume_dir.strip()
    if resume_from_dir:
        if not os.path.isdir(resume_from_dir):
            raise FileNotFoundError(resume_from_dir)
        found = find_latest_checkpoint(resume_from_dir)
        if found is None:
            raise FileNotFoundError(
                f"No model_step_*.safetensors + trainer_step_*.pt pair in {resume_from_dir}"
            )
        mp, tp, _ = found
        start_epoch, global_step = load_cvlm_checkpoint(
            mp, tp, model, optimizer, lr_scheduler, device
        )
        if use_ddp:
            dist.barrier()
        if is_master:
            print(
                f"Resumed from {resume_from_dir} "
                f"next_start_epoch={start_epoch} global_step={global_step}"
            )

    if is_master:
        print(optimizer)
        print("init_lrs:", init_lrs)
        print(
            f"total training steps (approx): {total_steps}\n"
            f"  epochs: {args.epochs}\n"
            f"  steps_per_epoch (accum): {steps_per_epoch}\n"
            f"  dataloader len: {len(loader)}\n"
            f"  world_size: {world_size}\n"
            f"  grad_accum: {grad_accum_steps}\n"
            f"  batch_size / rank: {batch_size_per_rank}\n"
            f"  effective batch (approx): {eff_batch}\n"
            f"  warmup_steps: {warmup_steps}\n"
            f"  grad_clip: {grad_clip}\n"
        )

    writer = None
    if is_master:
        from torch.utils.tensorboard import SummaryWriter
        tb_dir = args.tensorboard_dir.strip() or os.path.join(args.output_dir, "tb")
        os.makedirs(tb_dir, exist_ok=True)
        writer = SummaryWriter(log_dir=tb_dir)

    model.train()
    dtype = torch.bfloat16 if use_bf16 else torch.float32
    pgs = -1

    if start_epoch > args.epochs:
        if is_master:
            print(f"No epochs to run (start_epoch={start_epoch} > epochs={args.epochs}). Exiting.")
        cleanup_distributed(use_ddp)
        return

    for local_epoch in range(start_epoch, args.epochs + 1):
        if sampler is not None:
            sampler.set_epoch(local_epoch)

        optimizer.zero_grad(set_to_none=True)
        running_avg_batch_time = AverageMeter("time", ":6.3f")
        running_avg_loss_value = AverageMeter("loss", ":6.3f")
        grad_norm = 0.0
        t0 = time.perf_counter()
        curr_lrs: List[float] = []

        for local_step, batch_data in enumerate(loader, start=1):
            data_time = time.perf_counter() - t0
            batch_cpu = batch_data
            batch_data = move_batch(batch_cpu, device, dtype)

            if (
                local_step == 1
                or local_step % grad_accum_steps == 0
            ):
                curr_lrs = []
                lr_scheduler.step()

                if global_step > warmup_steps:
                    for param_group in optimizer.param_groups:
                        curr_lrs.append(param_group["lr"])
                else:
                    if warmup_steps > 0:
                        for group_id, param_group in enumerate(optimizer.param_groups):
                            curr_lr = init_lrs[group_id] * global_step / warmup_steps
                            param_group["lr"] = curr_lr
                            curr_lrs.append(curr_lr)
                    else:
                        for param_group in optimizer.param_groups:
                            curr_lrs.append(param_group["lr"])

            out = model(
                batch_data["input_embeds"],
                batch_data["prompt_ids"],
                batch_data["answer_ids"],
                answer_labels=batch_data["answer_labels"],
            )
            loss = out["loss"]
            loss = loss / grad_accum_steps

            sync_step = (
                local_step % grad_accum_steps == 0 or local_step == len(loader)
            )

            if use_ddp and not sync_step:
                with model.no_sync():
                    loss.backward()
            elif not use_ddp and not sync_step:
                loss.backward()
            else:
                loss.backward()
                if grad_clip > 0.0:
                    params = unwrap_model(model).parameters()
                    grad_norm = float(
                        torch.nn.utils.clip_grad_norm_(params, grad_clip)
                    )
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            t1 = time.perf_counter()
            batch_time = t1 - t0
            running_avg_batch_time.update(batch_time)
            curr_loss = loss.item() * grad_accum_steps
            running_avg_loss_value.update(curr_loss)

            if is_master and global_step % args.log_interval == 0:
                rp = global_step / max(total_steps, 1) * 100
                curr_lrs_str = " ' ".join(f"{lr:.10f}" for lr in curr_lrs)
                gn = f"grad norm: {grad_norm:>6.3f} " if grad_norm > 0 else ""
                print(
                    f"epoch: {local_epoch} step: {global_step} ({total_steps}) | {rp:5.2f}% ",
                    f"lr: {curr_lrs_str} loss: {curr_loss:>6.3f} ({running_avg_loss_value.avg:>6.3f}) ",
                    gn,
                    f"data: {data_time:>7.5f}s batch: {batch_time:>6.3f}s",
                )
                if writer is not None:
                    writer.add_scalar("train/loss", curr_loss, global_step)
                    writer.add_scalar("train/loss_avg", running_avg_loss_value.avg, global_step)
                    writer.add_scalar("train/lr", curr_lrs[0] if curr_lrs else args.lr, global_step)
                    if grad_norm > 0:
                        writer.add_scalar("train/grad_norm", grad_norm, global_step)
                    writer.add_scalar("train/batch_time", batch_time, global_step)

            if (
                is_master
                and args.save_interval_steps > 0
                and global_step % args.save_interval_steps == 0
                and sync_step
            ):
                if global_step != pgs:
                    save_cvlm_checkpoint(
                        args.output_dir,
                        model,
                        optimizer,
                        lr_scheduler,
                        next_start_epoch=local_epoch,
                        global_step=global_step,
                        is_master=True,
                    )
                    pgs = global_step

            t0 = time.perf_counter()

        # Without step interval, checkpoint once per epoch. With N>0, rely on
        # save_interval_steps only (otherwise every epoch end also saves, which is
        # often much more frequent than N when epochs are short).
        if is_master and args.save_interval_steps <= 0:
            if global_step != pgs:
                save_cvlm_checkpoint(
                    args.output_dir,
                    model,
                    optimizer,
                    lr_scheduler,
                    next_start_epoch=local_epoch + 1,
                    global_step=global_step,
                    is_master=True,
                )
                pgs = global_step

    if is_master and args.save_interval_steps > 0 and global_step != pgs:
        save_cvlm_checkpoint(
            args.output_dir,
            model,
            optimizer,
            lr_scheduler,
            next_start_epoch=args.epochs + 1,
            global_step=global_step,
            is_master=True,
        )

    if is_master:
        print("Training finished.")

    if writer is not None:
        writer.close()

    cleanup_distributed(use_ddp)


if __name__ == "__main__":
    main()
