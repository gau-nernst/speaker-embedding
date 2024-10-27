import argparse
import json
import logging
import math
import os
import sys
import time
from datetime import datetime
from pathlib import Path

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import numpy as np
import timm.optim
import torch
import wandb
from sklearn.metrics import roc_curve
from torch import Tensor
from torch.utils.data import DataLoader
from tqdm import tqdm

from data import Vox1OClean, build_train_dloader
from modelling import SpeakerModel

logger = logging.getLogger()
logger.setLevel(logging.INFO)
_LOGGING_FORMATTER = logging.Formatter("[%(asctime)s] %(levelname)s [%(name)s:%(lineno)d] %(message)s")
_stdout_handler = logging.StreamHandler(sys.stdout)
_stdout_handler.setFormatter(_LOGGING_FORMATTER)
logger.addHandler(_stdout_handler)


class CosineSchedule:
    def __init__(self, lr: float, total_steps: int, warmup: float = 0.05, decay_multiplier: float = 1e-2) -> None:
        self.lr = lr
        self.final_lr = lr * decay_multiplier
        self.total_steps = total_steps
        self.warmup_steps = round(total_steps * warmup)

    def get_lr(self, step: int) -> float:
        if step < self.warmup_steps:
            return self.lr * step / self.warmup_steps
        if step < self.total_steps:
            progress = (step - self.warmup_steps) / (self.total_steps - self.warmup_steps)
            return self.final_lr + 0.5 * (self.lr - self.final_lr) * (1 + math.cos(progress * math.pi))
        return self.final_lr

    def set_lr(self, step: int, optim):
        lr = self.get_lr(step)
        for group in optim.param_groups:
            group_lr = lr * group.get("lr_multiplier", 1)
            if isinstance(group["lr"], Tensor):
                group["lr"].copy_(group_lr)
            else:
                group["lr"] = group_lr


def build_optim(model: SpeakerModel, optim: str, lr: float, weight_decay: float, **kwargs):
    _globals = dict(torch=torch, timm=timm)
    try:
        import torchao.prototype.low_bit_optim

        _globals["torchao"] = torchao
    except ImportError:
        pass
    optim_cls = eval(optim, _globals)
    return optim_cls(model.parameters(), lr=lr, weight_decay=weight_decay, **kwargs)


def eer_score(y_true: np.ndarray, y_score: np.ndarray):
    fpr, tpr, thresholds = roc_curve(y_true, y_score)
    fnr = 1 - tpr

    eer_idx = np.argmin(np.abs((fpr - fnr)))
    eer = (fpr[eer_idx] + fnr[eer_idx]) / 2
    th = thresholds[eer_idx]
    return dict(eer=eer, eer_th=th)


@torch.no_grad()
def evalute_model(model: SpeakerModel, vox1_test_dir: str, duration: float, batch_size: int, bf16_amp: bool = False):
    model.eval()

    ds = Vox1OClean(vox1_test_dir, duration=duration)
    dloader = DataLoader(ds, batch_size, num_workers=4)

    all_labels = []
    all_scores = []

    for labels, audio1, audio2 in tqdm(dloader, desc="Evaluate", dynamic_ncols=True):
        all_labels.append(labels)
        with torch.autocast("cuda", torch.bfloat16, enabled=bf16_amp):
            embs1 = model(audio1)
            embs2 = model(audio2)
        all_scores.append((embs1.float() * embs2.float()).sum(-1))

    all_labels = torch.stack(all_labels, dim=0).numpy()
    all_scores = torch.stack(all_scores, dim=0).numpy()
    metrics = eer_score(all_labels, all_scores)
    return metrics


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--backbone", required=True)
    parser.add_argument("--backbone_kwargs", type=json.loads, default=dict())
    parser.add_argument("--n_classes", type=int, default=6000)
    parser.add_argument("--embed_dim", type=int, default=256)
    parser.add_argument("--loss", default="cosface")

    parser.add_argument("--bf16_model", action="store_true")
    parser.add_argument("--bf16_amp", action="store_true")
    parser.add_argument("--channels_last", action="store_true")
    parser.add_argument("--compile", action="store_true")

    parser.add_argument("--n_steps", type=int, default=1000)
    parser.add_argument("--eval_interval", type=int, default=1000)

    parser.add_argument("--ds_path", required=True)
    parser.add_argument("--augmentations", nargs="+")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--n_workers", type=int, default=4)

    parser.add_argument("--vox1_test_dir", required=True)
    parser.add_argument("--test_duration", type=float, default=4.0)

    parser.add_argument("--optim", default="torch.optim.AdamW")
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-3)
    parser.add_argument("--optim_kwargs", type=json.loads, default=dict())

    parser.add_argument("--clip_grad_norm", type=float)
    parser.add_argument("--warmup", type=float, default=0.05)
    parser.add_argument("--decay_multiplier", type=float, default=0.01)
    parser.add_argument("--grad_accum", type=int, default=1)

    parser.add_argument("--run_name", default="debug")
    parser.add_argument("--resume")
    return parser


if __name__ == "__main__":
    args = get_parser().parse_args()
    if args.bf16_amp and args.bf16_model:
        raise ValueError("AMP should not be used when using BF16 model")
    args.torch_version = torch.__version__
    assert args.batch_size % args.grad_accum == 0

    for k, v in vars(args).items():
        logger.info(f"{k}: {v}")

    time_now = datetime.now().strftime("%Y%m%d_%H%M%S")
    CKPT_DIR = Path("checkpoints") / f"{args.run_name}_{time_now}"
    assert not CKPT_DIR.exists()
    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    wandb.init(project="Speaker embedding", name=args.run_name, config=args, dir="/tmp")

    batch_size = args.batch_size // args.grad_accum
    dloader, train_size = build_train_dloader(
        args.ds_path,
        batch_size,
        args.augmentations,
        n_workers=args.n_workers,
    )
    logger.info(f"Train dataset: {train_size:,} images")
    logger.info(f"{args.n_steps / (train_size // args.batch_size):.2f} epochs")

    model = SpeakerModel(
        backbone=args.backbone,
        backbone_kwargs=args.backbone_kwargs,
        embed_dim=args.embed_dim,
        n_classes=args.n_classes,
        loss=args.loss,
    )
    if args.bf16_model:
        for p in model.parameters():
            p.data = p.detach().bfloat16()  # only cast params, don't cast buffers
    model.cuda()
    if args.channels_last:
        model.to(memory_format=torch.channels_last)
    if args.compile:
        model.compile()
    logger.info("Model parameters:")
    logger.info(f"  Backbone: {sum(p.numel() for p in model.backbone.parameters()):,}")
    logger.info(f"  Head: {model.weight.numel():,}")

    optim = build_optim(model, args.optim, args.lr, args.weight_decay, args.param_groups, **args.optim_kwargs)
    lr_schedule = CosineSchedule(args.lr, args.n_steps, warmup=args.warmup, decay_multiplier=args.decay_multiplier)
    step = 0

    if args.resume is not None:
        logger.info(f"Resume from {args.resume}")
        ckpt = torch.load(args.resume)
        step = ckpt["step"]
        model.load_state_dict(ckpt["model"])
        optim.load_state_dict(ckpt["optim"])

    pbar = tqdm(total=args.n_steps, dynamic_ncols=True, initial=step)
    model.train()
    time0 = time.perf_counter()
    log_interval = 100

    while step < args.n_steps:
        for _ in range(args.grad_accum):
            audio, labels = next(dloader)
            with torch.autocast("cuda", torch.bfloat16, enabled=args.bf16_amp):
                loss, norms = model(audio.cuda(), labels.cuda())
            (loss / args.grad_accum).backward()

        lr_schedule.set_lr(step, optim)
        grad_norm = None
        if args.clip_grad_norm is not None:
            # gradient clipping with BF16 gradients might be problematic
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad_norm)

        if step % log_interval == 0:
            loss = loss.detach()
            if grad_norm is None:
                grads = [p.grad.detach() for p in model.parameters() if p.grad is not None]
                grad_norms = torch._foreach_norm(grads)
                grad_norm = torch.linalg.vector_norm(torch.stack(grad_norms, dim=0))
            norms = norms.detach().cpu().numpy()
            log_dict = dict(
                loss=loss.item(),
                norm_hist=wandb.Histogram(norms),
                norm_mean=norms.mean(),
                grad_norm=grad_norm.item(),
            )
            for param_group in optim.param_groups:
                log_dict[f"lr/{param_group['prefix']}"] = param_group["lr"]
            wandb.log(log_dict, step=step)

        optim.step()
        optim.zero_grad()
        step += 1
        pbar.update()

        if step % log_interval == 0:
            time1 = time.perf_counter()
            log_dict = dict(
                max_memory_allocated=torch.cuda.max_memory_allocated(),
                imgs_seen_millions=args.batch_size * step / 1e6,
                imgs_per_second=args.batch_size * log_interval / (time1 - time0),
            )
            wandb.log(log_dict, step=step)
            time0 = time1

        if step % args.eval_interval == 0:
            checkpoint = dict(
                step=step,
                model=model.state_dict(),
            )
            torch.save(checkpoint, CKPT_DIR / f"step_{step}.pth")
            checkpoint.update(optim=optim.state_dict())
            torch.save(checkpoint, CKPT_DIR / "last.pth")  # for resume, w/ optim states

            metrics = evalute_model(model, args.vox1_test_dir, args.test_duration, batch_size, args.bf16_amp)
            wandb.log(metrics, step=step)

            model.train()
