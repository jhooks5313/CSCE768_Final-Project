# -*- coding: utf-8 -*-
"""
Created: Fall 2025

@author: jrhoo

This script holds the trainer class
Deals with train/val steps and metric collection
returns avg losses/epoch and computes ROC/AUC
"""
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score
import numpy as np
from tqdm import tqdm

class Trainer:
    def __init__(self, gen, disc, device, lr_G=2e-4, lr_D=1e-4, lamb=20.0, lamb_fm=0.0, gamma_sup=1.0):
        self.gen = gen.to(device); self.disc = disc.to(device)
        self.dev = device
        self.adv_loss = nn.BCEWithLogitsLoss()
        self.sup_loss = nn.CrossEntropyLoss()
        self.rec_loss = nn.L1Loss()
        self.optim_G = torch.optim.Adam(gen.parameters(), lr=lr_G, betas=(0.5,0.999))
        self.optim_D = torch.optim.Adam(disc.parameters(), lr=lr_D, betas=(0.5,0.999))
        self.lamb = lamb; self.lamb_fm = lamb_fm; self.gamma_sup = gamma_sup

    def train_epoch(self, loader, extra_g_when_strong=True):
        self.gen.train(); self.disc.train()
        logs = {"G_loss":[], "D_loss":[], "sup_acc":[], "G_rec":[], "G_adv":[]}
        pbar = tqdm(loader, desc="train", leave=False)
        for imgs, labs in pbar:
            imgs = imgs.to(self.dev); labs = labs.to(self.dev)
            b = imgs.size(0)
            # D step
            self.optim_D.zero_grad()
            out_real = self.disc(imgs)
            real_val, real_logits = out_real
            real_gt = torch.ones(b,1,device=self.dev)*0.92
            d_loss_real = self.adv_loss(real_val, real_gt)
            d_loss_cls = self.sup_loss(real_logits, labs)
            reconst = self.gen(imgs)
            fk_val, _ = self.disc(reconst.detach())
            fk_gt = torch.zeros(b,1,device=self.dev)
            d_loss_fk = self.adv_loss(fk_val, fk_gt)
            D_loss = d_loss_real + d_loss_cls + d_loss_fk
            D_loss.backward(); self.optim_D.step()
            # G step
            predcts = torch.argmax(real_logits, dim=1)
            acc = (predcts==labs).float().mean().item()
            g_steps = 2 if (extra_g_when_strong and acc>0.75) else 1
            for _ in range(g_steps):
                self.optim_G.zero_grad()
                reconst = self.gen(imgs)
                val, cls_logits = self.disc(reconst)
                G_adv = self.adv_loss(val, real_gt)
                G_rec = self.rec_loss(reconst, imgs)
                G_sup = self.sup_loss(cls_logits, torch.zeros(b, dtype=torch.long, device=self.dev))
                G_loss = G_adv + self.lamb*G_rec + self.gamma_sup*G_sup
                if self.lamb_fm>0:
                    pass
                G_loss.backward(); self.optim_G.step()
            logs["G_loss"].append(G_loss.item()); logs["D_loss"].append(D_loss.item())
            logs["sup_acc"].append(acc); logs["G_rec"].append(G_rec.item()); logs["G_adv"].append(G_adv.item())
            pbar.set_postfix({"G":f"{np.mean(logs['G_loss']):.4f}", "D":f"{np.mean(logs['D_loss']):.4f}"})
        pbar.close()
        out = {k: float(np.mean(v)) for k,v in logs.items()}
        return out

    def evaluate_on_loader(self, loader):
        self.gen.eval(); self.disc.eval()
        all_labels=[]; all_scores=[]
        with torch.no_grad():
            pbar = tqdm(loader, desc="eval", leave=False)
            for imgs, labs in pbar:
                imgs = imgs.to(self.dev)
                _, logits = self.disc(imgs)
                probs = nn.functional.softmax(logits, dim=1)[:,1].cpu().numpy()
                all_scores.extend(probs.tolist())
                all_labels.extend(labs.numpy().tolist())
            pbar.close()
        auc = roc_auc_score(all_labels, all_scores) if len(set(all_labels))>1 else float('nan')
        return {"auc": float(auc), "labels": np.array(all_labels), "scores": np.array(all_scores)}
