from toml import load as toml_load
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import textwrap

plt.style.use("seaborn-v0_8-darkgrid")

from utils import (
    get_df,
    collate_x,
)
from my_config import MyConfig
from embedding_model.config import Config


def get_sex_dfs(df):
    sex_dfs = {}
    for sex, sex_df in df.groupby(df["sex"]):
        sex_dfs[sex] = sex_df

    order = ["Male", "Female"]
    sex_dfs_sorted = {key: sex_dfs[key] for key in order if key in sex_dfs}

    return sex_dfs_sorted


def map_race(race):
    if race == "Caucasian/White":
        return "White or Caucasian"
    elif race in ["Black or African American", "Asian", "Unavailable"]:
        return race
    else:
        return "Others"


def get_race_dfs(df):
    df["mapped_race"] = df["race"].apply(map_race)
    race_dfs = {}

    for race, race_df in df.groupby(df["mapped_race"]):
        race_dfs[race] = race_df

    order = [
        "White or Caucasian",
        "Black or African American",
        "Asian",
        "Unavailable",
        "Others",
    ]
    race_dfs_sorted = {key: race_dfs[key] for key in order if key in race_dfs}

    return race_dfs_sorted


def map_insurance(insurance):
    public_insurance = [
        "Medicaid",
        "OOS Medicaid",
        "Medicaid Pending",
        "Other Government",
        "NC Medicaid Managed Care",
        "NC MEDICAID",
    ]
    private_insurance = [
        "Managed Care",
        "OOS Blue Cross",
        "Commercial",
        "Medicare Advantage",
        "NC Blue Cross",
    ]

    if insurance in public_insurance:
        return "Public"
    elif insurance in private_insurance:
        return "Private"
    else:
        return "Others"


def get_insurance_dfs(df):
    df["mapped_insurance"] = df["financial_class"].apply(map_insurance)
    insurance_dfs = {}

    for insurance, insurance_df in df.groupby(df["mapped_insurance"]):
        insurance_dfs[insurance] = insurance_df

    order = ["Public", "Private", "Others"]
    insurance_dfs_sorted = {
        key: insurance_dfs[key] for key in order if key in insurance_dfs
    }

    return insurance_dfs_sorted


config = Config(**toml_load("config.toml"))
my_config = MyConfig(**toml_load("my_config.toml"))

model_type = "DTNN"
yob_cutoff = 2020
followup_cutoff = 0
labels = list(my_config.dict().keys())

X_LABELS = ["Sex", "Race", "Insurance"]
X_FUNCS = [get_sex_dfs, get_race_dfs, get_insurance_dfs]

fig, axes = plt.subplots(len(X_LABELS), len(labels), figsize=(20, 12))

for i, label in enumerate(labels):
    print(f"Processing {label}...")
    myconfig_phenotype = getattr(my_config, label)
    config_phenotype = getattr(config.phenotypes, label)

    bin_boundaries = config_phenotype.bin_boundaries
    model = myconfig_phenotype.load_model(
        model_type, yob_cutoff, followup_cutoff, len(bin_boundaries) - 1
    )

    df = get_df(
        myconfig_phenotype.filepaths.test1422,
        yob_cutoff,
        followup_cutoff,
        config_phenotype.age_cutoff,
    )

    for j, x_func in enumerate(X_FUNCS):
        dfs = x_func(df)
        x_labels, x_event_times, x_predictions = collate_x(
            model, dfs, myconfig_phenotype.vocab, config.model.seq_threshold, label
        )

        x_data = list(dfs.keys())

        x_pred_risks = [
            np.cumsum(predictions, axis=1)[:, :-1] for predictions in x_predictions
        ]
        probability_tmax = [pred_risk[:, -1] for pred_risk in x_pred_risks]

        sns.boxplot(
            data=probability_tmax,
            color=myconfig_phenotype.color,
            ax=axes[j, i],
            showfliers=False,
            boxprops=dict(linewidth=0),
            saturation=1,
        )
        x_positions = np.arange(len(x_data))
        axes[j, i].set_xticks(x_positions)
        axes[j, i].set_xticklabels(
            [textwrap.fill(str(data), width=12) for data in x_data]
        )

        axes[j, i].set_ylabel("Predicted probability")


for i, label in enumerate(labels):
    axes[0, i].spines["top"].set_visible(False)
    myconfig_phenotype = getattr(my_config, label)
    axes[0, i].text(
        0.5,
        1.05,
        myconfig_phenotype.name,
        rotation=0,
        ha="center",
        va="center",
        transform=axes[0, i].transAxes,
        fontsize=20,
    )

for j, x_label in enumerate(X_LABELS):
    axes[j, 0].spines["left"].set_visible(False)
    axes[j, 0].text(
        -0.2,
        0.5,
        x_label,
        rotation=90,
        ha="center",
        va="center",
        transform=axes[j, 0].transAxes,
        fontsize=20,
    )

plt.tight_layout(rect=[0.02, 0, 1, 1])
plt.savefig(f"data/results/dtnn_demographics_prob_pred.png")
