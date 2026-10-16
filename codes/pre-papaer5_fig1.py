#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 16 17:02:59 2026

@author: sadrasaleh
"""

import pandas as pd
import matplotlib.pyplot as plt
import os

BASE = "/E/20_papars/DataBase/Nasa/selected/"
OUT = "/E/20_papars/Paper/conference/figures_v2/"
os.makedirs(OUT, exist_ok=True)

FILES = {
    "B0005": "B0005_D2_3_1_Discharge_Degradation_SOH_RUL_fixed_filter_epsilon_0.005.csv",
    "B0006": "B0006_D2_3_1_Discharge_Degradation_SOH_RUL_fixed_filter_epsilon_0.005.csv",
}

data = {}
for name, f in FILES.items():
    df = pd.read_csv(os.path.join(BASE, f))
    data[name] = df

fig, ax = plt.subplots(figsize=(8, 5))
for name, df in data.items():
    ax.plot(df["Cycle_ID"], df["SOH"], marker="o", markersize=3, label=name)
ax.set_xlabel("Cycle Number")
ax.set_ylabel("State of Health (SOH %)")
ax.set_title("Battery SOH Degradation")
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(OUT, "fig1_soh.png"), dpi=600)
plt.close()
print("Saved: fig1_soh.png at 600 dpi")