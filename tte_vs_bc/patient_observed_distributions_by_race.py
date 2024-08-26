import matplotlib.pyplot as plt
import seaborn as sns

plt.style.use("seaborn-v0_8-darkgrid")

import pandas as pd
import numpy as np
from utils import get_race_dfs
from toml import load as toml_load

from my_config import MyConfig


DATASET = "data/dataset/analytic(14-22).parquet"


def get_observed(df, label):
    return (
        df[df[label] == 1][["mrn", f"{label}_time_to_event"]]
        .groupby("mrn")
        .first()
        .squeeze()
    )


my_config = MyConfig(**toml_load("my_config.toml"))
labels = list(my_config.dict().keys())
df = pd.read_parquet(path=DATASET)
df = df[df["date_of_birth"].dt.year <= 2020]

fig, axes = plt.subplots(1, 4, figsize=(20, 5))
for index, label in enumerate(labels):
    print(f"Processing {label}...")

    myconfig_phenotype = getattr(my_config, label)
    dfs = get_race_dfs(df)

    for race, race_df in dfs.items():
        observed = get_observed(race_df, label)
        try:
            print(f"race {race}: {np.median(observed)}")
            sns.kdeplot(observed, label=race, ax=axes[index])
        except:
            print(f"race {race}: no data")

    axes[index].set_xlabel("Diagnosis age (years)")
    axes[index].set_ylabel("Density")
    axes[index].xaxis.set_ticks(np.arange(0, 11, 1))
    axes[index].set_title(myconfig_phenotype.name)

    axes[index].legend()

plt.tight_layout()
plt.savefig("data/results/patient_observed_distribution_by_race.png")
