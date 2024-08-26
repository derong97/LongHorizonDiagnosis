from toml import load as toml_load
import argparse
import matplotlib.pyplot as plt

plt.style.use("seaborn-v0_8-darkgrid")

from utils import (
    get_df,
    collate_x,
    plot_x_distributions,
    format_model_config,
    get_sex_dfs,
)
from my_config import MyConfig
from embedding_model.config import Config


config = Config(**toml_load("config.toml"))
my_config = MyConfig(**toml_load("my_config.toml"))

parser = argparse.ArgumentParser(description="Get plots by sex")
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
X_LABEL = "Sex"


fig, axes = plt.subplots(5, len(labels), figsize=(20, 20))
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
        args.yob_cutoff,
        args.followup_cutoff,
        config_phenotype.age_cutoff,
    )
    dfs = get_sex_dfs(df)
    x_labels, x_event_times, x_predictions = collate_x(
        model, dfs, myconfig_phenotype.vocab, config.model.seq_threshold, label
    )

    sexes = list(dfs.keys())
    axes[0, index].set_title(myconfig_phenotype.name, fontsize=20)
    plot_x_distributions(
        axes[:, index],
        sexes,
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
    f"data/results/subgroup_sex_{args.model}_{args.yob_cutoff}_{args.followup_cutoff}.png"
)
