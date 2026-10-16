#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 16 16:51:12 2026

@author: sadrasaleh
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
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

df_all = pd.concat([df for df in data.values()], ignore_index=True)

X = df_all[["Cycle_ID", "Discharge_Capacity", "SOH"]].values
y = df_all["RUL"].values

scaler_X = MinMaxScaler()
scaler_y = MinMaxScaler()
X_scaled = scaler_X.fit_transform(X)
y_scaled = scaler_y.fit_transform(y.reshape(-1, 1)).flatten()

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y_scaled, test_size=0.2, random_state=42
)

rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

importances = rf.feature_importances_
features = ["Cycle_ID", "Discharge_Capacity", "SOH"]

fig, ax = plt.subplots(figsize=(7, 5))
ax.barh(features, importances, color="steelblue")
ax.set_xlabel("Importance")
ax.set_title("Feature Importance (Random Forest)")
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(OUT, "fig5_feature_importance.png"), dpi=600)
plt.close()
print("Saved to:", OUT)