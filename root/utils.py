# -*- coding: utf-8 -*-
"""
Created: Fall 2025

@author: jrhoo

This script plots ROC, loss/epoch, and has save/load helpers
"""
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
import numpy as np

def plot_losses(history, out_path):
    plt.figure(figsize=(6,4))
    for k,v in history.items():
        plt.plot(v, label=k)
    plt.xlabel("Epoch"); plt.ylabel("Loss"); plt.legend(); plt.tight_layout()
    plt.savefig(out_path, dpi=200); plt.close()

def plot_roc(labels, scores, out_path):
    fpr, tpr, _ = roc_curve(labels, scores)
    roc_auc = auc(fpr, tpr)
    plt.figure(figsize=(5,5))
    plt.plot(fpr, tpr, label=f"AUC={roc_auc:.3f}")
    plt.plot([0,1],[0,1],"--", alpha=0.4)
    plt.xlabel("FPR"); plt.ylabel("TPR"); plt.legend(); plt.tight_layout()
    plt.savefig(out_path, dpi=200); plt.close()
    return roc_auc

