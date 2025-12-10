# -*- coding: utf-8 -*-
"""
Created: Fall 2025

@author: jrhoo

This script handles inference + heatmap generation
"""
import os, torch
from PIL import Image
from data import transform
from models import UNetG
import matplotlib.pyplot as plt
import numpy as np
import time

def load_gen(checkpoint_path, device):
    gen = UNetG(in_ch=1, base_ch=64)
    gen.load_state_dict(torch.load(checkpoint_path, map_location=device))
    gen.to(device); gen.eval()
    return gen

def gen_heat(og_img_tensor, generator, save_dir, sev_meth="topk", topk_pct=0.012, device='cpu'):
    generator.eval()
    with torch.no_grad():
        og = og_img_tensor.to(device)
        recon = generator(og)
        diff = torch.abs(og-recon).squeeze().cpu().numpy()
    diff_n = (diff-diff.min())/(diff.max()-diff.min()+1e-8)
    if sev_meth == 'mean': sev = float(diff_n.mean())
    elif sev_meth == 'max': sev = float(diff_n.max())
    elif sev_meth == 'topk':
        flat = diff_n.flatten()
        n = flat.size
        k = max(1, int(round(topk_pct*n)))
        idx = np.argpartition(flat, -k)[-k:]
        topk_vals = flat[idx] if k < n else flat
        sev = float(topk_vals.mean())
    else:
        raise ValueError("unknown sev_meth")
    og_np = og.squeeze().cpu().numpy()
    og_vis = (og_np+1)/2
    plt.imshow(og_vis, cmap='gray'); plt.imshow(diff_n, cmap="plasma", alpha=0.5)
    plt.colorbar(label="Reconstruction Difference")
    plt.title(f"Severity: {sev:.4f}")
    plt.axis("off")
    os.makedirs(save_dir, exist_ok=True)
    out_path = os.path.join(save_dir, f"heatmap_{int(time.time())}.png")
    plt.savefig(out_path, bbox_inches="tight", dpi=300); plt.close()
    return sev, out_path

if __name__ == "__main__":
    #change to paths used
    ckpt = "cv_results/fold1/gen_ep150.pth"
    img_path = "test_data/test_img0003.jpg"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    gen = load_gen(ckpt, device)
    pil = Image.open(img_path).convert("L")
    tensor = transform(pil).unsqueeze(0)  # [1,1,H,W]
    sev, saved = gen_heat(tensor, gen, "results", sev_meth="topk", device=device)
    print("Saved heatmap:", saved, "Severity:", sev)

