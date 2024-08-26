from toml import load as toml_load
import matplotlib.pyplot as plt

plt.style.use("seaborn-v0_8-darkgrid")

import seaborn as sns
import pandas as pd
import numpy as np
from scipy.special import expit
from utils import (
    get_df,
    collate_x,
    format_model_config,
    get_insurance_dfs,
    filter_df_by_yob,
)
from my_config import MyConfig
from embedding_model.config import Config

config = Config(**toml_load("config.toml"))
my_config = MyConfig(**toml_load("my_config.toml"))

LABELS = list(my_config.dict().keys())
CONFIGURATIONS = [
    ("DTNN", 2020, 0),
    ("DCPH", 2020, 0),
    ("BC", 2020, 0),
    ("BC", 2018, 0),
    ("BC", 2020, 5),
]

fig, axes = plt.subplots(2, 2, figsize=(20, 12))
axes = axes.flatten()

for i, label in enumerate(LABELS):
    print(f"Processing {label}...")
    myconfig_phenotype = getattr(my_config, label)
    config_phenotype = getattr(config.phenotypes, label)

    bin_boundaries = config_phenotype.bin_boundaries

    df = get_df(
        myconfig_phenotype.filepaths.test1422,
        2020,
        0,
        config_phenotype.age_cutoff,
    )
    df_size = len(CONFIGURATIONS) * len(df)
    df_boxplot = pd.DataFrame(
        index=range(df_size),
        columns=["Insurance", "Model Config", "Predicted Probability"],
    )

    times = bin_boundaries[1:]

    idx = 0
    for j, configuration in enumerate(CONFIGURATIONS):
        print(configuration)

        model_type, yob_cutoff, followup_cutoff = configuration
        model = myconfig_phenotype.load_model(
            model_type, yob_cutoff, followup_cutoff, len(bin_boundaries) - 1
        )
        model_config = format_model_config(
            model_type, yob_cutoff, followup_cutoff, myconfig_phenotype.name
        )

        df_final = filter_df_by_yob(df, yob_cutoff)
        df_final = df_final[df_final["censoring_age"] >= followup_cutoff]
        dfs = get_insurance_dfs(df_final)

        x_labels, x_event_times, x_predictions = collate_x(
            model, dfs, myconfig_phenotype.vocab, config.model.seq_threshold, label
        )

        insurancees = list(dfs.keys())

        # Get predicted probability
        x_pred_risks = []
        for labels, event_times, predictions in zip(
            x_labels, x_event_times, x_predictions
        ):
            if model_type == "BC":
                assert predictions.ndim == 1
                cumulative_predicted_risk = np.tile(
                    expit(predictions[:, np.newaxis]), (1, len(times))
                )
            elif model_type == "DTNN":
                assert predictions.ndim == 2
                cumulative_predicted_risk = np.cumsum(predictions, axis=1)[:, :-1]
            elif model_type == "DCPH":
                assert predictions.ndim == 1
                baseline_hazards = model.compute_baseline_hazards(
                    predictions, labels, event_times
                )
                predicted_cumulative_hazards = model.predict_cumulative_hazards(
                    baseline_hazards, predictions
                )
                predicted_cumulative_probs = 1 - np.exp(-predicted_cumulative_hazards)
                cumulative_predicted_risk = predicted_cumulative_probs.values.T

            x_pred_risks.append(cumulative_predicted_risk)
        probability_tmax = [pred_risk[:, -1] for pred_risk in x_pred_risks]

        for insurance, insurance_probs in zip(insurancees, probability_tmax):
            for indiv_prob in insurance_probs:
                df_boxplot.loc[idx] = [insurance, model_config, indiv_prob]
                idx += 1

    sns.boxplot(
        data=df_boxplot,
        x="Insurance",
        y="Predicted Probability",
        hue="Model Config",
        showfliers=False,
        boxprops=dict(linewidth=0),
        saturation=1,
        ax=axes[i],
        palette="muted",
    )

    axes[i].set_title(myconfig_phenotype.name, fontsize=15)
    handles, labels = axes[i].get_legend_handles_labels()
    axes[i].legend(handles=handles, labels=labels)

plt.tight_layout()
plt.savefig(f"data/results/subgroup_insurance_grouped_probability.png")
