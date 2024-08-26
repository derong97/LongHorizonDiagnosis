from toml import load as toml_load
import argparse
import matplotlib.pyplot as plt

plt.style.use("seaborn-v0_8-darkgrid")

import pandas as pd
from utils import (
    get_df,
    exclude_df_by_dob,
    collate_x,
    plot_x_distributions,
    format_model_config,
    get_yob_dfs,
)
from my_config import MyConfig
from embedding_model.config import Config


config = Config(**toml_load("config.toml"))
my_config = MyConfig(**toml_load("my_config.toml"))

parser = argparse.ArgumentParser(description="Get plots by YOB")
parser.add_argument(
    "--model",
    type=str,
    choices=["DTNN", "DCPH", "BC"],
    required=True,
)
parser.add_argument(
    "--yob_cutoff",
    type=int,
    choices=[2020, 2018],
    required=True,
)
parser.add_argument(
    "--followup_cutoff",
    type=int,
    choices=[0, 5],
    required=True,
)

args = parser.parse_args()
labels = list(my_config.dict().keys())
X_LABEL = "Year of Birth"


# fig, axes = plt.subplots(5, len(labels), figsize=(20, 20))
fig, axes = plt.subplots(3, len(labels), figsize=(20, 12))
for index, label in enumerate(labels):
    print(f"Processing {label}...")
    myconfig_phenotype = getattr(my_config, label)
    config_phenotype = getattr(config.phenotypes, label)

    bin_boundaries = config_phenotype.bin_boundaries
    model = myconfig_phenotype.load_model(
        args.model, args.yob_cutoff, args.followup_cutoff, len(bin_boundaries) - 1
    )

    df = get_df(
        myconfig_phenotype.filepaths.test1422,
        2020,
        0,
        config_phenotype.age_cutoff,
    )
    if args.followup_cutoff == 5:
        df_2014_2018 = df[df["censoring_age"] > args.followup_cutoff]  # ID
        df_2019_2020 = df[df["date_of_birth"].dt.year >= 2019]  # OOD
        df = pd.concat([df_2014_2018, df_2019_2020], ignore_index=True)
    if args.yob_cutoff == 2018:
        df = exclude_df_by_dob(df, "2018-06-02", "2018-12-31")  # make 2018 ID year
    dfs = get_yob_dfs(df)
    x_labels, x_event_times, x_predictions = collate_x(
        model, dfs, myconfig_phenotype.vocab, config.model.seq_threshold, label
    )

    years = list(dfs.keys())
    axes[0, index].set_title(myconfig_phenotype.name, fontsize=20)
    plot_x_distributions(
        axes[:, index],
        years,
        x_labels,
        x_event_times,
        x_predictions,
        bin_boundaries[1:],
        myconfig_phenotype.calibrated_time,
        myconfig_phenotype.color,
    )

fig.suptitle(
    f"Sub-group: {X_LABEL}; Set-up: {format_model_config(args.model, args.yob_cutoff, args.followup_cutoff)}",
    y=0.04,
    fontsize=20,
)
fig.tight_layout(rect=[0, 0.05, 1, 1])
plt.savefig(
    f"data/results/subgroup_yob_{args.model}_{args.yob_cutoff}_{args.followup_cutoff}.png"
)
