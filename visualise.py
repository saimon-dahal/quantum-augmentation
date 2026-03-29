import ast
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import entropy

OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

df = pd.read_csv("results.csv")


def parse_array(col):
    return np.array(df[col].apply(ast.literal_eval).tolist())


noisy = parse_array("noisy")
ideal = parse_array("ideal")
mitigated = parse_array("mitigated")

mae_noisy = np.mean(np.abs(noisy - ideal))
mae_mitigated = np.mean(np.abs(mitigated - ideal))

plt.figure()
plt.bar(["Noisy", "Mitigated"], [mae_noisy, mae_mitigated], color=["orange", "green"])
plt.ylabel("Mean Absolute Error")
plt.title("MAE Comparison")
plt.savefig(f"{OUTPUT_DIR}/mae_comparison.png", dpi=300)
plt.close()

ideal_mean = ideal.mean(axis=0)
noisy_mean = noisy.mean(axis=0)
mitigated_mean = mitigated.mean(axis=0)

x = np.arange(len(ideal_mean))

plt.figure()
plt.plot(x, ideal_mean, marker="o", label="Ideal")
plt.plot(x, noisy_mean, marker="o", label="Noisy")
plt.plot(x, mitigated_mean, marker="o", label="Mitigated")
plt.xticks(x, ["00", "01", "10", "11"])
plt.ylabel("Average Probability")
plt.title("Average Probability Distribution")
plt.legend()
plt.savefig(f"{OUTPUT_DIR}/distribution_comparison.png", dpi=300)
plt.close()

error_noisy = np.mean(np.abs(noisy - ideal), axis=1)
error_mitigated = np.mean(np.abs(mitigated - ideal), axis=1)

plt.figure()
plt.scatter(error_noisy, error_mitigated, alpha=0.6)
plt.plot(
    [0, max(error_noisy.max(), error_mitigated.max())],
    [0, max(error_noisy.max(), error_mitigated.max())],
    linestyle="--",
    color="red",
)
plt.xlabel("Noisy Error")
plt.ylabel("Mitigated Error")
plt.title("Error Scatter Plot")
plt.savefig(f"{OUTPUT_DIR}/error_scatter.png", dpi=300)
plt.close()

ideal_entropy = [entropy(p) for p in ideal]
noisy_entropy = [entropy(p) for p in noisy]
mitigated_entropy = [entropy(p) for p in mitigated]

plt.figure()
plt.bar(
    ["Ideal", "Noisy", "Mitigated"],
    [np.mean(ideal_entropy), np.mean(noisy_entropy), np.mean(mitigated_entropy)],
    color=["blue", "orange", "green"],
)
plt.ylabel("Average Shannon Entropy")
plt.title("Entropy of Output Distributions")
plt.savefig(f"{OUTPUT_DIR}/entropy_comparison.png", dpi=300)
plt.close()
