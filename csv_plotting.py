# This Python file uses the following encoding: utf-8

# if __name__ == "__main__":
#     pass

# csv_plotting.py
# Visualize decision boundary and sample size bounds

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from glob import glob
import csv

sns.set(style="whitegrid")

def load_params_dict(params_csv):
    with open(params_csv, newline='') as csvfile:
        reader = csv.reader(csvfile)
        return {row[0].strip(): float(row[1]) for row in reader if len(row) == 2}

def find_risks_from_summary(summary_csv, tag):
    if not os.path.exists(summary_csv):
        return None, None
    df = pd.read_csv(summary_csv)
    row = df[df.apply(lambda r: tag in f"p{int(r.p*10)}_q{int(r.q*10)}_e{int(r.epsilon*100)}_h{int(r.eta*100)}", axis=1)]
    if not row.empty:
        return row.iloc[0]["train_risk"], row.iloc[0]["test_risk"]
    return None, None

def plot_decision_boundary(train_csv, test_csv, params_csv, summary_csv="experiment_summary.csv"):
    train_df = pd.read_csv(train_csv)
    test_df = pd.read_csv(test_csv)
    params = load_params_dict(params_csv)

    a1 = params["a1"]
    a2 = params["a2"]
    theta = params["theta"]

    tag = params_csv.replace("_params.csv", "")
    train_risk, test_risk = find_risks_from_summary(summary_csv, tag)

    fig, ax = plt.subplots(figsize=(6, 6))
    sns.scatterplot(x="x1", y="x2", hue="label", data=train_df, style=train_df["label"], palette="coolwarm", ax=ax, legend=False)
    sns.scatterplot(x="x1", y="x2", data=test_df, marker="x", color="black", alpha=0.3, ax=ax, legend=False)

    x_vals = np.linspace(-3, 3, 100)
    if a2 != 0:
        y_vals = (theta - a1 * x_vals) / a2
        ax.plot(x_vals, y_vals, 'k--', label=f"Boundary: {a1:.2f}x + {a2:.2f}y = {theta:.2f}")
    else:
        x_line = theta / a1 if a1 != 0 else 0.0
        ax.axvline(x=x_line, color='k', linestyle='--', label=f"x = {x_line:.2f}")

    subtitle = f"Train risk: {train_risk:.3f}, Test risk: {test_risk:.3f}" if train_risk is not None else ""
    ax.set_title(f"Decision Boundary\n{subtitle}")
    ax.set_xlim(-3, 3)
    ax.set_ylim(-3, 3)
    ax.legend()
    plt.tight_layout()
    plt.show()

def plot_bounds_vs_epsilon(csv_file):
    df = pd.read_csv(csv_file)
    plt.figure(figsize=(8, 5))
    plt.plot(df["epsilon"], df["l_simp"], label="Simplified Bound", marker="o")
    plt.plot(df["epsilon"], df["l_vc"], label="VC-Theoretic Bound", marker="s")
    plt.xlabel("Epsilon (ε)")
    plt.ylabel("Sample Size (l)")
    plt.title("Sample Size vs ε")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    csv_group = glob("*_train.csv")
    if csv_group:
        base = csv_group[0].replace("_train.csv", "")
        plot_decision_boundary(base + "_train.csv", base + "_test.csv", base + "_params.csv")

    if os.path.exists("epsilon_vs_bounds.csv"):
        plot_bounds_vs_epsilon("epsilon_vs_bounds.csv")
