import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import sys

# Load the data
df = pd.read_csv(sys.argv[1], delimiter=" ")
df = df[df["percent_planar"] >= 0.0]

# Plot the data
fig, ax = plt.subplots(1, 3, dpi=300, figsize=(15, 5), layout="constrained")

values = ["mean_edge", "mean_planar", "error"]
names = ["Mean # of edges features", "Mean # of planar features", "Error"]

print(df["error"].mean())

for i, val in enumerate(values):
    df_err = df.pivot(index="percent_edge", columns="percent_planar", values=val)
    ax[i].set_title(names[i])
    fmt = ".1f" if val == "error" else ".0f"
    vmax = df["error"].mean() if val == "error" else None
    sns.heatmap(
        df_err,
        ax=ax[i],
        cmap="rocket",
        annot=True,
        fmt=fmt,
        annot_kws={"fontsize": 7.5},
        vmax=vmax,
    )
    ax[i].invert_yaxis()

plt.show()
