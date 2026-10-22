#!/usr/bin/env python3
"""Train audited block EBD on the existing matched ViT/data protocol."""
import argparse
from collections import defaultdict
from dataclasses import asdict
import hashlib
import inspect
import json
from pathlib import Path
import sys
import time

import torch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/"src"))
from experiment_logging import ExperimentLogger
from reproducibility import configure_reproducibility
from methods.transformer_ebd.configs import ViTConfig
from methods.transformer_ebd.trainer import build_loaders, str2bool
from methods.grape_vit_audit.model import AuditedViT
from methods.component_scaling.data import record_split_provenance
from methods.component_scaling.trainer import (move, capture_rng, restore_rng,
    atomic_checkpoint, source_hashes as common_source_hashes)
from methods.ebd_vit.learner import EBDConfig,EBDLearner,supervised_objective
from methods.ebd_vit.objectives import EBDObjectiveConfig


def parser():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument("--method",choices=["block_ebd"],default="block_ebd")
    p.add_argument("--dataset",choices=["cifar10","cifar100","tiny_imagenet"],default="cifar10")
    p.add_argument("--data-root")
    p.add_argument("--norm-layout",choices=["post"],default="post")
    p.add_argument("--device",choices=["cpu","cuda"],default="cuda")
    p.add_argument("--dtype",choices=["float32","float64"],default="float32")
    p.add_argument("--seed",type=int,default=0)
    p.add_argument("--feedback-seed",type=int,default=100)
    p.add_argument("--epochs",type=int,default=100)
    p.add_argument("--stop-after-epoch",type=int,default=0)
    p.add_argument("--batch-size",type=int,default=64)
    p.add_argument("--train-subset-samples",type=int,default=5000)
    p.add_argument("--validation-samples",type=int,default=1000)
    p.add_argument("--validation-split-seed",type=int,default=2027)
    p.add_argument("--num-workers",type=int,default=2)
    p.add_argument("--threads",type=int,default=2)
    p.add_argument("--augmentation",type=str2bool,default=True)
    p.add_argument("--evaluate-test-final",type=str2bool,default=False)
    p.add_argument("--image-size",type=int,default=32)
    p.add_argument("--patch-size",type=int,default=4)
    p.add_argument("--embed-dim",type=int,default=128)
    p.add_argument("--depth",type=int,default=4)
    p.add_argument("--num-heads",type=int,default=4)
    p.add_argument("--ffn-dim",type=int,default=256)
    p.add_argument("--loss",choices=["cross_entropy","mse"],default="cross_entropy")
    p.add_argument("--parameter-scope",choices=["all"],default="all")
    p.add_argument("--optimizer",choices=["adam","sgd"],default="adam")
    p.add_argument("--learning-rate",type=float,default=1e-4)
    p.add_argument("--momentum",type=float,default=.9)
    p.add_argument("--weight-decay",type=float,default=1e-4)
    p.add_argument("--run-tag",default="ebd_vit_corrected")
    p.add_argument("--require-clean-git",type=str2bool,default=False)
    p.add_argument("--jobs-per-gpu",type=int,default=1)
    p.add_argument("--checkpoint-every",type=int,default=5)
    p.add_argument("--resume")
    for name,value in asdict(EBDObjectiveConfig()).items():
        options = {"choices":["author_sample","paper_batch"]} if name == "power_mode" else {"type":float}
        p.add_argument("--ebd-"+name.replace("_","-"),default=value,**options)
    return p


def protocol(args):
    ignored = {"stop_after_epoch","resume","run_tag","require_clean_git","jobs_per_gpu","checkpoint_every"}
    return {k:v for k,v in vars(args).items() if k not in ignored}


def source_hashes():
    hashes = common_source_hashes()
    for p in Path(__file__).parent.glob("*.py"):
        hashes[str(p.relative_to(ROOT))] = hashlib.sha256(p.read_bytes()).hexdigest()
    return hashes


def load_checkpoint(path):
    # Locally created full training checkpoints include RNG/optimizer objects.
    # PyTorch 1.12 has no weights_only argument; newer releases require it
    # explicitly for this format. Inspect the API instead of masking load errors.
    options = {"weights_only":False} if "weights_only" in inspect.signature(torch.load).parameters else {}
    return torch.load(path,map_location="cpu",**options)


@torch.no_grad()
def evaluate(model,loader,device,dtype,loss):
    model.eval()
    count = correct = 0
    total = 0.
    for batch in loader:
        x,y = move(batch,device,dtype)
        logits = model(x)
        value = supervised_objective(logits,y,loss)
        if not torch.isfinite(value):
            raise FloatingPointError("Nonfinite evaluation loss")
        count += len(y)
        correct += int((logits.argmax(-1) == y).sum())
        total += float(value)*len(y)
    return correct/count,total/count


def main(argv=None):
    args = parser().parse_args(argv)
    stop = args.stop_after_epoch or args.epochs
    if not 1 <= stop <= args.epochs or args.checkpoint_every < 1 or args.batch_size < 1:
        raise ValueError("Invalid training schedule")
    if args.evaluate_test_final and stop != args.epochs:
        raise ValueError("Development stage cannot access final test data")
    if args.image_size != (64 if args.dataset == "tiny_imagenet" else 32):
        raise ValueError("Image size does not match the dataset")
    local_cfg = EBDObjectiveConfig(**{n:getattr(args,"ebd_"+n) for n in EBDObjectiveConfig.__dataclass_fields__})
    learning_cfg = EBDConfig(**{n:getattr(args,n) for n in EBDConfig.__dataclass_fields__ if n != "objective"},objective=local_cfg)
    torch.set_num_threads(args.threads)
    reproducibility = configure_reproducibility(args.seed)
    device,dtype = torch.device(args.device),getattr(torch,args.dtype)
    train,train_eval,validation,test,data = build_loaders(args)
    data.update(record_split_provenance(train,validation))
    cfg = ViTConfig(image_size=args.image_size,patch_size=args.patch_size,num_classes=data["num_classes"],
        embed_dim=args.embed_dim,depth=args.depth,num_heads=args.num_heads,ffn_dim=args.ffn_dim)
    model = AuditedViT(cfg,norm_layout="post").to(device=device,dtype=dtype)
    learner = EBDLearner(model,learning_cfg)
    auxiliary_generator = torch.Generator().manual_seed(args.seed+71831)
    hashes = source_hashes()
    records,first,elapsed = [],1,0.
    checkpoint = None
    if args.resume:
        checkpoint = load_checkpoint(args.resume)
        if checkpoint["protocol"] != protocol(args) or checkpoint["data_info"] != data:
            raise ValueError("Resume scientific protocol or data split differs")
        if checkpoint["source_sha256"] != hashes:
            raise ValueError("Resume implementation differs from checkpoint")
        learner.load_state_dict(checkpoint["learner"])
        records = checkpoint["records"]
        first = checkpoint["epoch"]+1
        elapsed = checkpoint["training_wall_clock_sec"]
        if first > stop:
            raise ValueError("Checkpoint has already reached the requested epoch")
    run_cfg = {"args":vars(args),"model":cfg.to_dict(),"learning":asdict(learning_cfg),
        "data":data,"reproducibility":reproducibility,"source_sha256":hashes,
        "fidelity":"EBD local objectives adapted at matched ViT block boundaries; not author-supplied ViT code",
        "correlation":"E.T @ flattened block output / B; raw EMA, no matrix normalization",
        "entropy":"sum of per-token channel logdet objectives; block-diagonal covariance approximation",
        "power":"author sample-wise quartic or paper batch-power objective, selected explicitly",
        "scope":"all parameters; tokenizer grouped with first block; exact supervised readout",
        "local_derivatives":"analytic exact within-block VJPs, no cross-block reverse transport",
        "normalization":"No explicit gradient RMS or Frobenius feedback normalization; common optimizer",
        "loss_definition":"mean CE" if args.loss == "cross_entropy" else "mean half sum of squared class residuals",
        "bp_calibration_batches":0,"optimizer_schedule":"constant",
        "concurrency_policy":f"up to {args.jobs_per_gpu} assigned jobs per GPU; shared timing; not efficiency evidence",
        "parent_checkpoint":args.resume}
    logger = ExperimentLogger(ROOT,args.dataset,f"vit_postln_d{cfg.depth}_p{cfg.patch_size}_e{cfg.embed_dim}",
        "block_ebd",args.seed,run_cfg,run_tag=args.run_tag,require_clean_git=args.require_clean_git)
    logger.reset_peak_cuda_memory()
    if checkpoint is not None:
        restore_rng(checkpoint["rng"],train,auxiliary_generator)
    started = time.perf_counter()
    for epoch in range(first,stop+1):
        epoch_start = time.perf_counter()
        count = correct = 0
        total = 0.
        sums = defaultdict(float)
        try:
            for batch in train:
                x,y = move(batch,device,dtype)
                logits,loss,diagnostics = learner.step(x,y)
                count += len(y)
                correct += int((logits.argmax(1) == y).sum())
                total += loss*len(y)
                for key,value in diagnostics.items():
                    sums[key] += value*len(y)
            va,vl = evaluate(model,validation,device,dtype,args.loss)
            record = logger.log_epoch(epoch=epoch,train_loss=total/count,train_accuracy=correct/count,
                test_loss=None,test_accuracy=None,epoch_time_sec=time.perf_counter()-epoch_start,
                learning_rate=args.learning_rate,extra={"validation_accuracy":va,"validation_loss":vl,
                    "test_accessed":False,"bp_calibration_batches":0,"training_steps":learner.steps,
                    "diagnostics":{k:v/count for k,v in sums.items()}})
            records.append(record)
            print(json.dumps({k:record[k] for k in ("epoch","train_loss","train_accuracy","validation_loss","validation_accuracy","epoch_time_sec")}),flush=True)
            if epoch % args.checkpoint_every == 0 or epoch == stop:
                atomic_checkpoint(logger.artifacts_dir/"last_checkpoint.pt",{
                    "epoch":epoch,"protocol":protocol(args),"data_info":data,"source_sha256":hashes,
                    "learner":learner.state_dict(),"records":records,"rng":capture_rng(train,auxiliary_generator),
                    "training_wall_clock_sec":elapsed+time.perf_counter()-started})
        except Exception as error:
            logger.log_epoch(epoch=epoch,train_loss=None,train_accuracy=None,test_loss=None,test_accuracy=None,
                epoch_time_sec=time.perf_counter()-epoch_start,learning_rate=args.learning_rate,failed=True,
                failure_reason=repr(error),extra={"test_accessed":False})
            raise
    ta,tl = evaluate(model,train_eval,device,dtype,args.loss)
    test_accuracy,test_loss = (None,None) if test is None else evaluate(model,test,device,dtype,args.loss)
    summary = {"status":"completed" if stop == args.epochs else "development_stage_completed",
        "completed_epoch":stop,"scheduled_epochs":args.epochs,"dataset":args.dataset,"data_info":data,
        "method":args.method,"seed":args.seed,"train_accuracy":ta,"train_loss":tl,
        "validation_accuracy":records[-1]["validation_accuracy"],"validation_loss":records[-1]["validation_loss"],
        "test_accuracy":test_accuracy,"test_loss":test_loss,"test_accessed":test is not None,
        "bp_calibration_batches":0,"parameter_scope":"all","norm_layout":"post",
        "wall_clock_time_sec":elapsed+time.perf_counter()-started,"peak_gpu_memory_mb":logger.peak_cuda_memory_mb(),
        "last_checkpoint":str(logger.artifacts_dir/"last_checkpoint.pt"),
        "comparison_warning":"Block EBD ViT adaptation; data scope and test access explicit; shared-GPU timing"}
    (logger.artifacts_dir/"summary.json").write_text(json.dumps(summary,indent=2)+"\n")
    print(json.dumps(summary),flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
