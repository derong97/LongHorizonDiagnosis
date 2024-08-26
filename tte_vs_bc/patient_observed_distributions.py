import matplotlib.pyplot as plt

plt.style.use("seaborn-v0_8-darkgrid")

import pandas as pd
import numpy as np
from toml import load as toml_load

from my_config import MyConfig
from embedding_model.config import Config

my_config = MyConfig(**toml_load("my_config.toml"))
config = Config(**toml_load("config.toml"))

labels = list(my_config.dict().keys())
DATASET = "data/dataset/analytic(14-22).parquet"

df = pd.read_parquet(DATASET)
df = df[df["date_of_birth"].dt.year <= 2020]

fig, axes = plt.subplots(2, 4, figsize=(20, 8))
for index, label in enumerate(labels):
    myconfig_phenotype = getattr(my_config, label)
    config_phenotype = getattr(config.phenotypes, label)

    observed = (
        df[df[label] == 1][["mrn", f"{label}_time_to_event"]].groupby("mrn").first()
    )
    print(np.mean(observed))

    # By year
    ax = axes[0, index]
    ax.hist(
        observed,
        bins=np.arange(11),
        color=myconfig_phenotype.color,
    )
    ax.axvline(x=config_phenotype.age_cutoff, color="r")
    ax.set_xlabel("Diagnosis age (years)")
    ax.set_ylabel("Frequency")
    ax.xaxis.set_ticks(np.arange(0, 11, 1))
    ax.set_title(myconfig_phenotype.name, fontsize=20)

    """
    for p in ax.patches:
        ax.annotate(
            f"{p.get_height()}",
            (p.get_x() + p.get_width() / 2.0, p.get_height()),
            ha="center",
            va="center",
            fontsize=9,
            color="black",
            xytext=(0, 5),
            textcoords="offset points",
        )
    """

    # By month
    ax = axes[1, index]
    ax.hist(
        observed * 12,
        bins=np.arange(25),
        color=myconfig_phenotype.color,
    )
    ax.axvline(x=config_phenotype.age_cutoff * 12, color="r")
    ax.set_xlabel("Diagnosis age (months)")
    ax.set_ylabel("Frequency")
    ax.xaxis.set_ticks(np.arange(0, 25, 2))

    """
    for p in ax.patches:
        ax.annotate(
            f"{p.get_height()}",
            (p.get_x() + p.get_width() / 2.0, p.get_height()),
            ha="center",
            va="center",
            fontsize=9,
            color="black",
            xytext=(0, 5),
            textcoords="offset points",
        )
    """

fig.suptitle(f"Distribution of Observed Diagnosis Age", y=0.04, fontsize=20)
fig.tight_layout(rect=[0, 0.05, 1, 1])
plt.savefig("data/results/patient_observed_distribution.png")
