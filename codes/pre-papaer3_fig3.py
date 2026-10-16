#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 16 17:00:29 2026

@author: sadrasaleh
"""

import pandas as pd
import numpy as np
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
    print(f"{name}: {len(df)} rows")

fig, ax = plt.subplots(figsize=(8, 5))
for name, df in data.items():
    ax.plot(df["Cycle_ID"], df["RUL"], marker="o", markersize=3, label=name)
ax.set_xlabel("Cycle Number")
ax.set_ylabel("Remaining Useful Life (Cycles)")
ax.set_title("Battery RUL")
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(OUT, "fig3_rul.png"), dpi=600)
plt.close()
print("Saved: fig3_rul.png at 600 dpi")