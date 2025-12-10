# -*- coding: utf-8 -*-
"""
Created: Fall 2025

@author: jrhoo

This script has k-fold orchestration, checkpointing, ROC/AUC calculation & plots
saves per-fold loss plots and ROC plots

K-fold SGAN training controlled entirely by config.yaml.
Saves generator and discriminator checkpoints each epoch and records per-fold metrics.
"""
import os, sys, traceback
print("=== DEBUG BOOTSTRAP ===")
print("sys.executable:", sys.executable)
print("cwd:", os.getcwd())
print("PYTHONPATH:", os.environ.get("PYTHONPATH"))
print("PATH (start):", os.environ.get("PATH")[:400])
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
try:
    import torch
    print("import torch OK:", torch.__version__, "cuda:", torch.version.cuda)
except Exception:
    print("ERROR importing torch in train_cv.py")
    traceback.print_exc()
    sys.exit(1)
print("=== DEBUG BOOTSTRAP COMPLETE ===\n")

import json, time, yaml
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import precision_score, recall_score, confusion_matrix
from torch.utils.data import DataLoader
from data import DefDataset
from models import UNetG, D
from trainer import Trainer
from utils import plot_losses, plot_roc

def load_config(path="config.yaml"):
    with open(path, "r") as f:
        return yaml.safe_load(f)

def ensure_keys(cfg, defaults):
    for k, v in defaults.items():
        if k not in cfg:
            cfg[k] = v
    return cfg

def run_kfold(cfg):
    defaults = {
        'data_dir': './data/images',
        'labels_csv': './data/labels.csv',
        'out_dir': './cv_results',
        'k': 5,
        'epochs': 15,
        'batch_size': 8,
        'seed': 42,
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        'num_workers': 4,
        'base_channels': 64,
        'img_shape': [1, 256, 256],
        'resume_epoch': 0,
        'max_folds': 0
    }
    cfg = ensure_keys(cfg, defaults)

    data_dir = cfg['data_dir']
    labels_csv = cfg['labels_csv']
    out_dir = cfg['out_dir']
    k = int(cfg['k'])
    epochs = int(cfg['epochs'])
    batch_size = int(cfg['batch_size'])
    seed = int(cfg['seed'])
    device = cfg['device']
    num_workers = int(cfg['num_workers'])
    base_channels = cfg['base_channels']
    img_shape = tuple(cfg['img_shape'])
    resume_epoch_cfg = int(cfg.get('resume_epoch', 0) or 0)
    max_folds = int(cfg.get('max_folds', 0) or 0)

    # safe: on cpu-only force num_workers=0 and no pinned memory
    if not torch.cuda.is_available():
        num_workers = 0
        pin_memory = False
    else:
        pin_memory = True

    os.makedirs(out_dir, exist_ok=True)

    df = pd.read_csv(labels_csv)
    filenames = df['filename'].values
    labels = df['label'].values

    if k < 2:
        from sklearn.model_selection import train_test_split
        idxs = np.arange(len(filenames))
        train_idx, val_idx = train_test_split(idxs, test_size=0.2, stratify=labels, random_state=seed)
        splits = [(train_idx, val_idx)]
    else:
        skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=seed)
        splits = list(skf.split(filenames, labels))

    fold_results = []

    for fold, (train_idx, val_idx) in enumerate(splits, start=1):
        start_fold = time.time()
        fold_dir = os.path.join(out_dir, f"fold{fold}")
        os.makedirs(fold_dir, exist_ok=True)
        print(f"[FOLD {fold}] start (pid={os.getpid()})", flush=True)

        train_df = df.iloc[train_idx].reset_index(drop=True)
        val_df = df.iloc[val_idx].reset_index(drop=True)
        train_csv = os.path.join(fold_dir, f"train_fold{fold}.csv")
        val_csv = os.path.join(fold_dir, f"val_fold{fold}.csv")
        train_df.to_csv(train_csv, index=False)
        val_df.to_csv(val_csv, index=False)
        print(f"[FOLD {fold}] splits written (train={len(train_df)} val={len(val_df)})", flush=True)

        train_ds = DefDataset(data_dir, train_csv)
        val_ds = DefDataset(data_dir, val_csv)

        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                                  num_workers=num_workers, pin_memory=pin_memory)
        val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                                num_workers=num_workers, pin_memory=pin_memory)

        gen = UNetG(in_ch=1, base_ch=base_channels)
        disc = D(img_shape=img_shape)
        trainer = Trainer(gen, disc, device=device)

        # resume support: load weights from resume_epoch-1 to continue at resume_epoch
        resume_epoch = resume_epoch_cfg
        start_epoch = 1
        if resume_epoch and resume_epoch > 1:
            prev_ep = resume_epoch - 1
            gen_prev = os.path.join(fold_dir, f"gen_ep{prev_ep}.pth")
            disc_prev = os.path.join(fold_dir, f"disc_ep{prev_ep}.pth")
            if os.path.exists(gen_prev) and os.path.exists(disc_prev):
                try:
                    gen.load_state_dict(torch.load(gen_prev, map_location=device))
                    disc.load_state_dict(torch.load(disc_prev, map_location=device))
                    print(f"[FOLD {fold}] loaded weights from ep {prev_ep} -> resuming at epoch {resume_epoch}", flush=True)
                    start_epoch = resume_epoch
                except Exception as e:
                    print(f"[FOLD {fold}] failed loading weights for resume: {e}; starting from epoch 1", flush=True)
            else:
                print(f"[FOLD {fold}] checkpoint files for epoch {prev_ep} not found; starting from epoch 1", flush=True)
        else:
            start_epoch = 1

        # load existing per-fold epoch CSV rows so we do not duplicate them
        epoch_csv = os.path.join(fold_dir, "epoch_metrics.csv")
        existing_epochs = set()
        if os.path.exists(epoch_csv):
            try:
                with open(epoch_csv, "r") as f:
                    next(f)
                    for line in f:
                        parts = line.strip().split(",")
                        if parts and parts[0].isdigit():
                            existing_epochs.add(int(parts[0]))
            except Exception:
                existing_epochs = set()

        history_path = os.path.join(fold_dir, "history.npz")
        if os.path.exists(history_path):
            try:
                old = np.load(history_path)
                history = {
                    "G_loss": old["G_loss"].tolist() if "G_loss" in old else [],
                    "D_loss": old["D_loss"].tolist() if "D_loss" in old else [],
                    "val_auc": old["val_auc"].tolist() if "val_auc" in old else []
                }
            except Exception:
                history = {"G_loss": [], "D_loss": [], "val_auc": []}
        else:
            history = {"G_loss": [], "D_loss": [], "val_auc": []}

        for ep in range(start_epoch, epochs + 1):
            t0 = time.time()
            stats = trainer.train_epoch(train_loader)
            t1 = time.time()
            eval_res = trainer.evaluate_on_loader(val_loader)
            t2 = time.time()

            history["G_loss"].append(stats.get("G_loss", float("nan")))
            history["D_loss"].append(stats.get("D_loss", float("nan")))
            history["val_auc"].append(eval_res.get("auc", float("nan")))

            # per-fold CSV (guarded)
            if not os.path.exists(epoch_csv):
                with open(epoch_csv, "w") as f:
                    f.write("epoch,G_loss,D_loss,val_auc\n")
            if ep not in existing_epochs:
                with open(epoch_csv, "a") as f:
                    f.write(f"{ep},{stats.get('G_loss',float('nan'))},{stats.get('D_loss',float('nan'))},{eval_res.get('auc',float('nan'))}\n")
                existing_epochs.add(ep)

            # combined CSV (guarded)
            combined_csv = os.path.join(out_dir, "all_epoch_metrics.csv")
            if not os.path.exists(combined_csv):
                with open(combined_csv, "w") as f:
                    f.write("fold,epoch,G_loss,D_loss,val_auc\n")
            append_combined = True
            if os.path.exists(combined_csv):
                try:
                    with open(combined_csv, "r") as f:
                        lines = f.readlines()
                        if len(lines) > 1:
                            last = lines[-1].strip().split(",")
                            if len(last) >= 2 and last[0].isdigit() and int(last[0]) == fold and int(last[1]) == ep:
                                append_combined = False
                except Exception:
                    append_combined = True
            if append_combined:
                with open(combined_csv, "a") as f:
                    f.write(f"{fold},{ep},{stats.get('G_loss',float('nan'))},{stats.get('D_loss',float('nan'))},{eval_res.get('auc',float('nan'))}\n")

            # save models + history
            gen_path = os.path.join(fold_dir, f"gen_ep{ep}.pth")
            disc_path = os.path.join(fold_dir, f"disc_ep{ep}.pth")
            torch.save(gen.state_dict(), gen_path)
            torch.save(disc.state_dict(), disc_path)
            np.savez(history_path, **history)

            print(f"[FOLD {fold}] Epoch {ep}/{epochs}  G_loss={stats.get('G_loss',float('nan')):.4f}  "
                  f"D_loss={stats.get('D_loss',float('nan')):.4f}  val_auc={eval_res.get('auc',float('nan')):.4f}  "
                  f"(train_t={t1-t0:.1f}s val_t={t2-t1:.1f}s)", flush=True)

        y_true = eval_res["labels"]
        y_score = eval_res["scores"]
        y_pred = (y_score >= 0.5).astype(int)

        precision = precision_score(y_true, y_pred, zero_division=0)
        recall = recall_score(y_true, y_pred, zero_division=0)
        cm = confusion_matrix(y_true, y_pred)

        metrics = {
            "fold": fold,
            "auc": float(eval_res.get("auc", float("nan"))),
            "precision": float(precision),
            "recall": float(recall),
            "confusion_matrix": cm.tolist(),
        }
        with open(os.path.join(fold_dir, "metrics.json"), "w") as f:
            json.dump(metrics, f, indent=2)

        plot_losses({"G_loss": history["G_loss"], "D_loss": history["D_loss"]},
                    os.path.join(fold_dir, f"losses.png"))
        plot_roc(y_true, y_score, os.path.join(fold_dir, f"roc.png"))

        fold_results.append(metrics)
        print(f"[FOLD {fold}] done in {time.time()-start_fold:.1f}s", flush=True)

        if max_folds and fold >= max_folds:
            print(f"max_folds reached ({max_folds}) — stopping early.", flush=True)
            break

    pd.DataFrame(fold_results).to_csv(os.path.join(out_dir, "folds_summary.csv"), index=False)
    return fold_results

if __name__ == "__main__":
    cfg = load_config("config.yaml")
    try:
        import torch.multiprocessing as mp
        mp.set_start_method("spawn", force=True)
    except Exception:
        pass
    run_kfold(cfg)
