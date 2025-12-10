# -*- coding: utf-8 -*-
"""
Created: Fall 2025

@author: jrhoo

This script holds the UNet Generator and discriminator classes
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

#UNetG 
class UNetG(nn.Module):
    def __init__(self, in_ch=1, base_ch=64):
        super().__init__()
        #encoder blocks
        def enc_blk(in_c, out_c):
            return nn.Sequential(
                nn.Conv2d(in_c, out_c, 4, 2, 1),
                nn.BatchNorm2d(out_c),
                nn.LeakyReLU(0.2, inplace=True)
            )
        self.e1 = nn.Sequential(nn.Conv2d(in_ch, base_ch, 4, 2, 1),
                                 nn.LeakyReLU(0.2, inplace=True))
        self.e2 = enc_blk(base_ch, base_ch*2)
        self.e3 = enc_blk(base_ch*2, base_ch*4)
        self.e4 = enc_blk(base_ch*4, base_ch*8)
        self.bottle = nn.Sequential(nn.Conv2d(base_ch*8, base_ch*8, 3, 1, 1),
                                    nn.ReLU(inplace=True))
        #decoder
        def dec_blk(in_c, out_c):
            return nn.Sequential(
                nn.Conv2d(in_c, out_c, 3, padding=1),
                nn.BatchNorm2d(out_c),
                nn.ReLU(inplace=True)
            )
        self.dc4 = dec_blk(base_ch*8, base_ch*4)
        self.dc3 = dec_blk(base_ch*8, base_ch*2)
        self.dc2 = dec_blk(base_ch*4, base_ch)
        self.dc1 = dec_blk(base_ch*2, base_ch)
        self.final = nn.Sequential(nn.Conv2d(base_ch, in_ch, 3, padding=1), nn.Tanh())

    def _crop_match(self, enc, dec):
        _,_,h_t,w_t = dec.size()
        _,_,h_e,w_e = enc.size()
        if h_e == h_t and w_e == w_t: return enc
        sh = max((h_e - h_t)//2, 0)
        sw = max((w_e - w_t)//2, 0)
        return enc[:,:,sh:sh+h_t, sw:sw+w_t]

    def forward(self, x):
        e1 = self.e1(x); e2 = self.e2(e1); e3 = self.e3(e2); e4 = self.e4(e3)
        b = self.bottle(e4)
        d4 = F.interpolate(b, size=(e3.size(2), e3.size(3)), mode='bilinear', align_corners=False)
        d4 = self.dc4(d4); e3c = self._crop_match(e3, d4); d4 = torch.cat([d4,e3c], dim=1)
        d3 = F.interpolate(d4, size=(e2.size(2), e2.size(3)), mode='bilinear', align_corners=False)
        d3 = self.dc3(d3); e2c = self._crop_match(e2, d3); d3 = torch.cat([d3,e2c], dim=1)
        d2 = F.interpolate(d3, size=(e1.size(2), e1.size(3)), mode='bilinear', align_corners=False)
        d2 = self.dc2(d2); e1c = self._crop_match(e1, d2); d2 = torch.cat([d2,e1c], dim=1)
        d1 = F.interpolate(d2, size=(x.size(2), x.size(3)), mode='bilinear', align_corners=False)
        d1 = self.dc1(d1)
        return self.final(d1)
#Discriminator
class D(nn.Module):
    def __init__(self, num_cls=2, img_shape=(1,256,256)):
        super().__init__()
        self.c_model = nn.Sequential(
            nn.Conv2d(1,32,3,stride=2,padding=1),
            nn.LeakyReLU(0.2,inplace=True),
            nn.Dropout2d(0.25),
            nn.Conv2d(32,64,3,stride=2,padding=1),
            nn.BatchNorm2d(64,momentum=0.8),
            nn.LeakyReLU(0.2,inplace=True),
            nn.Dropout2d(0.25),
            nn.Conv2d(64,128,3,stride=2,padding=1),
            nn.BatchNorm2d(128,momentum=0.8),
            nn.LeakyReLU(0.2,inplace=True),
            nn.Dropout2d(0.25),
        )
        dwns_size = img_shape[1] // 8
        self.rf_layr = nn.Linear(128*dwns_size*dwns_size, 1)
        self.dfnd_layr = nn.Linear(128*dwns_size*dwns_size, num_cls)

    def forward(self, img):
        out = self.c_model(img).view(img.size(0), -1)
        val = self.rf_layr(out)
        label = self.dfnd_layr(out)
        return val, label
